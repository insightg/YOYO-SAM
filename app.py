#!/usr/bin/env python3
"""
SAM3 Object Detection Web Interface
FastAPI backend for the SAM3 detection tool
"""

import os
import sys
import io
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Thread pool for parallel preprocessing
_preprocess_executor = ThreadPoolExecutor(max_workers=4)

# Add local modules to path
from geolocate import get_poses, geolocate_detections
from deep_analyze import is_deep_class, deep_analyze_detection
from qwen_analyze import qwen_analyze_detection
from local_analyze import is_local_class, local_analyze_detection, batch_analyze_detections
from crack_analyze import is_crack_class
from reconstruct_3d import (
    generate_3d_from_detection, cleanup_cache as cleanup_3d_cache,
    check_sam3d_available, get_cache_status, CACHE_DIR as CACHE_3D_DIR
)

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent / "sam3"))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Configuration
IMAGES_DIR = Path("/home/giobbe/tools/data/cities/Piacenza/run1/Panoramas/original")
CLASS_LISTS_DIR = Path(__file__).parent / "class_lists"
CLASS_LISTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "test_out"
OUTPUT_DIR.mkdir(exist_ok=True)
DETECTIONS_DIR = Path(__file__).parent / "detections"
DETECTIONS_DIR.mkdir(exist_ok=True)
THRESHOLDS_FILE = Path(__file__).parent / "thresholds.json"
THUMBNAIL_SIZE = 150
DEFAULT_CLASSES = """30,Cartello_stradale,road sign,gtsrb
40,Dare_precedenza,yield sign,
40,Limite_velocita,speed limit sign,
30,Semaforo,traffic light,
,Lampione,street light pole,
,Palo_luce,light pole,
,Tombino,manhole cover,
,Segnaletica_orizzontale,road marking,
,Strisce_pedonali,crosswalk,
,Attraversamento,pedestrian crossing,
,Divieto_sosta,no parking sign,
,Senso_unico,one way sign,
,Parcheggio,parking sign,
,Idrante,fire hydrant,
,Cono_stradale,traffic cone,
,Barriera,road barrier,
,Guardrail,guardrail,"""

# Global model (loaded once)
_processor: Optional[Sam3Processor] = None
_model_loading = False
_text_embeddings_cache: dict = {}  # Cache for text embeddings


def parse_csv_class(csv_line: str) -> dict:
    """
    Parse CSV class format: threshold,label,description,modules

    Examples:
        "30,Passo_carrabile,no parking sign,ocr"
        "50,Limite_velocita,speed limit sign,ocr,deep"
        ",Street_light,street light pole,"  (no threshold, no modules)

    Returns:
        {
            "threshold": 0.30 or None,
            "label": "Passo_carrabile",
            "description": "no parking sign",
            "modules": ["ocr"] or []
        }
    """
    parts = [p.strip() for p in csv_line.split(',')]

    # Minimo 3 campi: threshold, label, description
    if len(parts) < 3:
        # Fallback: tratta come formato legacy
        return parse_legacy_class(csv_line)

    result = {
        "threshold": None,
        "label": parts[1] if parts[1] else parts[2].replace(' ', '_'),
        "description": parts[2],
        "modules": []
    }

    # Parse threshold (campo 1)
    if parts[0] and parts[0].isdigit():
        result["threshold"] = int(parts[0]) / 100.0

    # Parse modules (campo 4+)
    if len(parts) > 3:
        modules = [m.strip().lower() for m in parts[3:] if m.strip()]
        result["modules"] = modules

    return result


def parse_legacy_class(class_str: str) -> dict:
    """Fallback per formato legacy (30 road sign - GTSRB)"""
    from local_analyze import parse_threshold_prefix, is_local_class
    from deep_analyze import is_deep_class

    name, threshold = parse_threshold_prefix(class_str)

    is_local, base, module, _ = is_local_class(class_str)
    if is_local:
        return {
            "threshold": threshold,
            "label": base,
            "description": base,
            "modules": [module.lower()] if module else []
        }

    is_deep, base, _ = is_deep_class(class_str)
    if is_deep:
        return {
            "threshold": threshold,
            "label": base,
            "description": base,
            "modules": ["deep"]
        }

    return {
        "threshold": threshold,
        "label": name,
        "description": name,
        "modules": []
    }


def get_processor() -> Sam3Processor:
    """Get or create the SAM3 processor (singleton)."""
    global _processor, _model_loading

    if _processor is None and not _model_loading:
        _model_loading = True
        print("Loading SAM3 model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path)

        # Apply torch.compile for faster inference
        print("Applying torch.compile optimization...")
        try:
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead")
            print("torch.compile applied successfully")
        except Exception as e:
            print(f"torch.compile failed (falling back to eager mode): {e}")

        _processor = Sam3Processor(model, confidence_threshold=0.1)

        print("SAM3 model loaded successfully")
        _model_loading = False

    return _processor


def get_text_embedding(processor, class_name: str, device: str = "cuda"):
    """Get cached text embedding for a class name."""
    global _text_embeddings_cache

    if class_name not in _text_embeddings_cache:
        text_outputs = processor.model.backbone.forward_text([class_name], device=device)
        _text_embeddings_cache[class_name] = {
            k: v.clone() if hasattr(v, 'clone') else v
            for k, v in text_outputs.items()
        }

    return _text_embeddings_cache[class_name]


def set_text_prompt_cached(processor, state: dict, class_name: str):
    """
    Apply cached text embedding and run inference.
    Faster than set_text_prompt() for repeated class names.
    """
    if "backbone_out" not in state:
        raise ValueError("You must call set_image before set_text_prompt_cached")

    # Get cached text embedding
    cached_text = get_text_embedding(processor, class_name)

    # Apply to state (clone to avoid modifying cache)
    for k, v in cached_text.items():
        state["backbone_out"][k] = v.clone() if hasattr(v, 'clone') else v

    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    # Run forward grounding
    return processor._forward_grounding(state)


def precompute_text_embeddings(processor, classes: list[str]):
    """Pre-compute and cache all text embeddings for the given classes."""
    for class_name in classes:
        get_text_embedding(processor, class_name)


def process_tiles_batch(processor, tiles_data: list, classes: list[str],
                        confidence: float, y_crop_offset: int,
                        original_width: int, original_height: int) -> list[dict]:
    """
    Process multiple tiles with optimized batching.
    Uses CUDA streams for better GPU utilization.
    """
    all_detections = []

    # Pre-compute all text embeddings
    precompute_text_embeddings(processor, classes)

    # Create CUDA stream for async operations
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    for tile, x_offset, y_offset in tiles_data:
        # Calculate backbone (most expensive operation)
        with torch.cuda.stream(stream) if stream else torch.no_grad():
            inference_state = processor.set_image(tile)

        # Process all classes for this tile
        for class_name in classes:
            processor.reset_all_prompts(inference_state)
            inference_state = set_text_prompt_cached(processor, inference_state, class_name)

            if "scores" in inference_state and len(inference_state["scores"]) > 0:
                scores = inference_state["scores"]
                boxes = inference_state["boxes"]

                for i, score in enumerate(scores):
                    score_val = score.item() if hasattr(score, "item") else float(score)
                    if score_val >= confidence:
                        box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                        x1, y1, x2, y2 = box

                        x1_adj = float(x1) + x_offset
                        y1_adj = float(y1) + y_offset + y_crop_offset
                        x2_adj = float(x2) + x_offset
                        y2_adj = float(y2) + y_offset + y_crop_offset

                        all_detections.append({
                            "class": class_name,
                            "score": round(score_val, 4),
                            "bbox": [x1_adj, y1_adj, x2_adj, y2_adj],
                            "bbox_normalized": [
                                x1_adj / original_width,
                                y1_adj / original_height,
                                x2_adj / original_width,
                                y2_adj / original_height
                            ]
                        })

    # Sync CUDA stream
    if stream:
        stream.synchronize()

    return all_detections


# Panorama preprocessing settings
CROP_TOP_PERCENT = 0.20    # Remove 20% from top (sky/roof)
CROP_BOTTOM_PERCENT = 0.25  # Remove 25% from bottom (car hood)


def crop_panorama(image: Image.Image) -> tuple[Image.Image, int]:
    """
    Crop top and bottom of panoramic image to remove car/sky.

    Returns:
        (cropped_image, y_offset) - y_offset is used to adjust bbox coordinates
    """
    width, height = image.size

    y_start = int(height * CROP_TOP_PERCENT)
    y_end = int(height * (1 - CROP_BOTTOM_PERCENT))

    cropped = image.crop((0, y_start, width, y_end))

    return cropped, y_start


def calculate_optimal_tiles(width: int, height: int, min_tile_size: int = 1500) -> tuple[int, int, int]:
    """
    Calculate optimal tile layout based on image dimensions.

    Args:
        width: Image width
        height: Image height (after crop)
        min_tile_size: Minimum tile dimension for good detection

    Returns:
        (num_tiles, cols, rows)
    """
    # Calculate max tiles that maintain minimum size
    max_cols = width // min_tile_size
    max_rows = height // min_tile_size

    # Common tile layouts
    layouts = [
        (6, 3, 2),    # 3x2
        (8, 4, 2),    # 4x2
        (9, 3, 3),    # 3x3
        (12, 4, 3),   # 4x3
        (16, 4, 4),   # 4x4
        (20, 5, 4),   # 5x4
        (24, 6, 4),   # 6x4
        (30, 6, 5),   # 6x5
    ]

    # Find best layout that fits constraints
    best_layout = (6, 3, 2)  # Default
    for num, cols, rows in layouts:
        tile_w = width // cols
        tile_h = height // rows

        # Check if tiles are big enough
        if tile_w >= min_tile_size and tile_h >= min_tile_size:
            # Prefer more tiles for better coverage
            if cols <= max_cols and rows <= max_rows:
                best_layout = (num, cols, rows)

    return best_layout


# Import detection functions from detect.py
def split_image_into_tiles(image: Image.Image, num_tiles: int = 6, custom_layout: tuple = None):
    """
    Split an image into tiles.

    Args:
        image: PIL Image
        num_tiles: Number of tiles (6, 8, 9, 12, 16, 20, 24, 30)
        custom_layout: Optional (cols, rows) tuple to override default

    Returns:
        List of (tile_image, x_offset, y_offset)
    """
    width, height = image.size

    if custom_layout:
        cols, rows = custom_layout
    elif num_tiles == 6:
        cols, rows = 3, 2
    elif num_tiles == 8:
        cols, rows = 4, 2
    elif num_tiles == 9:
        cols, rows = 3, 3
    elif num_tiles == 12:
        cols, rows = 4, 3
    elif num_tiles == 16:
        cols, rows = 4, 4
    elif num_tiles == 20:
        cols, rows = 5, 4
    elif num_tiles == 24:
        cols, rows = 6, 4
    elif num_tiles == 30:
        cols, rows = 6, 5
    else:
        cols, rows = 3, 2

    tile_width = width // cols
    tile_height = height // rows

    tiles = []
    for row in range(rows):
        for col in range(cols):
            x_offset = col * tile_width
            y_offset = row * tile_height
            x_end = width if col == cols - 1 else x_offset + tile_width
            y_end = height if row == rows - 1 else y_offset + tile_height
            tile = image.crop((x_offset, y_offset, x_end, y_end))
            tiles.append((tile, x_offset, y_offset))

    return tiles


def process_tile(processor, tile, classes, confidence_threshold, x_offset, y_offset, original_width, original_height):
    """Process a single tile and return detections."""
    all_detections = []

    # OPTIMIZATION: Calculate backbone ONCE per tile, reuse for all classes
    inference_state = processor.set_image(tile)

    for class_name in classes:
        processor.reset_all_prompts(inference_state)
        # OPTIMIZATION: Use cached text embeddings
        inference_state = set_text_prompt_cached(processor, inference_state, class_name)

        if "scores" in inference_state and len(inference_state["scores"]) > 0:
            scores = inference_state["scores"]
            boxes = inference_state["boxes"]

            for i, score in enumerate(scores):
                score_val = score.item() if hasattr(score, "item") else float(score)
                if score_val >= confidence_threshold:
                    box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                    x1, y1, x2, y2 = box

                    x1_adj = float(x1) + x_offset
                    y1_adj = float(y1) + y_offset
                    x2_adj = float(x2) + x_offset
                    y2_adj = float(y2) + y_offset

                    all_detections.append({
                        "class": class_name,
                        "score": round(score_val, 4),
                        "bbox": [x1_adj, y1_adj, x2_adj, y2_adj],
                        "bbox_normalized": [
                            x1_adj / original_width,
                            y1_adj / original_height,
                            x2_adj / original_width,
                            y2_adj / original_height
                        ]
                    })

    return all_detections


def detect_objects(image_path: Path, classes: list[str], confidence: float, tiles: int) -> list[dict]:
    """Run object detection on an image."""
    processor = get_processor()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    all_detections = []

    if tiles > 0:
        tile_list = split_image_into_tiles(image, tiles)
        for tile, x_offset, y_offset in tile_list:
            tile_detections = process_tile(
                processor, tile, classes, confidence,
                x_offset, y_offset, width, height
            )
            all_detections.extend(tile_detections)
    else:
        # OPTIMIZATION: Calculate backbone ONCE, reuse for all classes
        inference_state = processor.set_image(image)

        for class_name in classes:
            processor.reset_all_prompts(inference_state)
            # OPTIMIZATION: Use cached text embeddings
            inference_state = set_text_prompt_cached(processor, inference_state, class_name)

            if "scores" in inference_state and len(inference_state["scores"]) > 0:
                scores = inference_state["scores"]
                boxes = inference_state["boxes"]

                for i, score in enumerate(scores):
                    score_val = score.item() if hasattr(score, "item") else float(score)
                    if score_val >= confidence:
                        box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                        x1, y1, x2, y2 = box

                        all_detections.append({
                            "class": class_name,
                            "score": round(score_val, 4),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "bbox_normalized": [
                                float(x1) / width,
                                float(y1) / height,
                                float(x2) / width,
                                float(y2) / height
                            ]
                        })

    # Sort by class name, then by score descending
    all_detections.sort(key=lambda x: (x["class"], -x["score"]))

    return all_detections


# FastAPI app
app = FastAPI(title="SAM3 Object Detection")

# Create static directory if not exists
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class DetectRequest(BaseModel):
    image_name: str
    classes: list[str]
    confidence: float = 0.3
    tiles: int = 24


@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(static_dir / "index.html")


@app.get("/api/images")
async def list_images(limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0)):
    """List available images with pagination."""
    if not IMAGES_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Images directory not found: {IMAGES_DIR}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_images = []

    for ext in extensions:
        all_images.extend(IMAGES_DIR.glob(f"*{ext}"))
        all_images.extend(IMAGES_DIR.glob(f"*{ext.upper()}"))

    # Get list of images that have CSV detections
    csv_stems = {f.stem for f in DETECTIONS_DIR.glob("*.csv")} if DETECTIONS_DIR.exists() else set()

    # Sort: images with CSV first, then alphabetically
    all_images = sorted(all_images, key=lambda x: (0 if x.stem in csv_stems else 1, x.name))
    total = len(all_images)

    # Apply pagination
    images = all_images[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "images": [img.name for img in images],
        "has_csv": [img.stem in csv_stems for img in images]
    }


@app.get("/api/thumbnail/{image_name}")
async def get_thumbnail(image_name: str):
    """Get a thumbnail of an image."""
    image_path = IMAGES_DIR / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Generate thumbnail
    img = Image.open(image_path)
    img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)

    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


@app.get("/api/image/{image_name}")
async def get_image(image_name: str, max_size: int = Query(1600, ge=100, le=4000)):
    """Get an image, optionally resized."""
    image_path = IMAGES_DIR / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img = Image.open(image_path)

    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


@app.get("/api/image-info/{image_name}")
async def get_image_info(image_name: str):
    """Get image metadata."""
    image_path = IMAGES_DIR / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img = Image.open(image_path)

    return {
        "name": image_name,
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "size_bytes": image_path.stat().st_size
    }


@app.post("/api/detect")
async def detect(request: DetectRequest):
    """Run object detection on an image (non-streaming version)."""
    image_path = IMAGES_DIR / request.image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    if not request.classes:
        raise HTTPException(status_code=400, detail="No classes provided")

    try:
        # Parse classes (support CSV and legacy formats)
        local_classes = {}   # label -> module (OCR, GTSRB, RDD)
        deep_classes = {}    # label -> True
        qwen_classes = {}    # label -> True (local Qwen VL analysis)
        class_thresholds = {}  # label -> min_threshold
        detection_classes = []  # descriptions for SAM3
        label_to_description = {}  # label -> description mapping

        for cls_config in request.classes:
            parsed = parse_csv_class(cls_config)

            label = parsed["label"]
            description = parsed["description"]
            modules = parsed["modules"]
            threshold = parsed["threshold"]

            # Store mapping label -> description (for SAM3 prompt)
            label_to_description[label] = description

            # Route to appropriate module
            if "ocr" in modules:
                local_classes[label] = "OCR"
            elif "gtsrb" in modules:
                local_classes[label] = "GTSRB"
            elif "rdd" in modules:
                local_classes[label] = "RDD"
            elif "local" in modules:
                local_classes[label] = "auto"

            if "deep" in modules:
                deep_classes[label] = True

            if "qwen" in modules:
                qwen_classes[label] = True

            # Use DESCRIPTION for SAM3 detection, not label
            detection_classes.append(description)

            if threshold is not None:
                class_thresholds[label] = threshold

        # Create reverse mapping: description -> label
        description_to_label = {v: k for k, v in label_to_description.items()}

        # Run detection with descriptions (for SAM3)
        detections = detect_objects(
            image_path,
            detection_classes,
            request.confidence,
            request.tiles
        )

        # Map back: detection["class"] = description -> label
        for det in detections:
            desc = det["class"]
            if desc in description_to_label:
                det["class"] = description_to_label[desc]
                det["description"] = desc

        # Perform local analysis for marked classes (GTSRB/RDD model) - apply class threshold
        if local_classes:
            for det in detections:
                if det["class"] in local_classes:
                    # Use class-specific threshold if set, otherwise no minimum
                    min_threshold = class_thresholds.get(det["class"])
                    if min_threshold is not None and det.get("score", 0) < min_threshold:
                        continue  # Skip detections below threshold
                    det["original_class"] = det["class"]
                    module = local_classes[det["class"]]  # Get explicit module
                    detailed_class, local_confidence = local_analyze_detection(
                        image_path,
                        det,
                        det["class"],
                        module
                    )
                    det["class"] = detailed_class
                    det["local_analyzed"] = True
                    det["local_confidence"] = round(local_confidence, 4)
                    det["local_module"] = module if module != "auto" else ("RDD" if is_crack_class(det["original_class"]) else "GTSRB")

        # Perform deep analysis for marked classes (LLM) - apply class threshold or default 0.5
        if deep_classes:
            for det in detections:
                # Use original_class (label) if available, otherwise current class
                label = det.get("original_class", det["class"])
                if label in deep_classes:
                    # Use class-specific threshold if set, otherwise default to 0.5
                    min_threshold = class_thresholds.get(label, 0.5)
                    if det.get("score", 0) >= min_threshold:
                        if "original_class" not in det:
                            det["original_class"] = det["class"]
                        detailed_class = deep_analyze_detection(
                            image_path,
                            det,
                            label
                        )
                        det["class"] = detailed_class
                        det["deep_analyzed"] = True

        # Perform Qwen analysis for marked classes (local VLM) - apply class threshold or default 0.5
        if qwen_classes:
            for det in detections:
                label = det.get("original_class", det["class"])
                if label in qwen_classes:
                    min_threshold = class_thresholds.get(label, 0.5)
                    if det.get("score", 0) >= min_threshold:
                        if "original_class" not in det:
                            det["original_class"] = det["class"]
                        detailed_class = qwen_analyze_detection(
                            image_path,
                            det,
                            label
                        )
                        det["class"] = detailed_class
                        det["qwen_analyzed"] = True

        # Get image dimensions for frontend
        img = Image.open(image_path)

        # Add GPS coordinates to detections
        poses = get_poses()
        if request.image_name in poses:
            geolocated = geolocate_detections(request.image_name, detections, poses)
            # Merge geolocation data into detections
            for det, geo in zip(detections, geolocated):
                det["latitude"] = geo.get("latitude")
                det["longitude"] = geo.get("longitude")
                det["distance_m"] = geo.get("distance_m")
                det["bearing_deg"] = geo.get("bearing_deg")
                det["geo_confidence"] = geo.get("confidence")

            # Get camera info
            pose = poses[request.image_name]
            camera_info = {
                "latitude": pose.latitude,
                "longitude": pose.longitude,
                "heading": pose.heading,
                "altitude": pose.altitude
            }
        else:
            camera_info = None

        # Filter detections by per-class thresholds
        if class_thresholds:
            filtered_detections = []
            for det in detections:
                # Get the base class for threshold lookup
                base_cls = det.get("original_class", det["class"])
                # For nested classes like "road_sign.limite_30", use the base part
                if "." in base_cls:
                    base_cls = base_cls.split(".")[0].replace("_", " ")
                min_threshold = class_thresholds.get(base_cls)
                if min_threshold is None:
                    # No threshold specified, keep detection
                    filtered_detections.append(det)
                elif det.get("score", 0) >= min_threshold:
                    filtered_detections.append(det)
            detections = filtered_detections

        return {
            "status": "success",
            "image_name": request.image_name,
            "image_width": img.width,
            "image_height": img.height,
            "camera": camera_info,
            "parameters": {
                "confidence": request.confidence,
                "tiles": request.tiles,
                "classes_count": len(request.classes)
            },
            "detections_count": len(detections),
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def detect_with_progress(image_name: str, classes: list[str], confidence: float, tiles: int) -> AsyncGenerator[dict, None]:
    """Generator that yields progress updates during detection."""
    image_path = IMAGES_DIR / image_name

    if not image_path.exists():
        yield {"type": "error", "message": "Image not found"}
        return

    if not classes:
        yield {"type": "error", "message": "No classes provided"}
        return

    try:
        # Step 1: Initialize
        yield {"type": "progress", "stage": "init", "message": "Inizializzazione...", "percent": 0}
        await asyncio.sleep(0.05)  # Small delay to allow UI update

        # Parse classes (support CSV and legacy formats)
        local_classes = {}   # label -> module (OCR, GTSRB, RDD)
        deep_classes = {}    # label -> True
        qwen_classes = {}    # label -> True (local Qwen VL analysis)
        class_thresholds = {}  # label -> min_threshold
        detection_classes = []  # descriptions for SAM3
        label_to_description = {}  # label -> description mapping

        for cls_config in classes:
            parsed = parse_csv_class(cls_config)

            label = parsed["label"]
            description = parsed["description"]
            modules = parsed["modules"]
            threshold = parsed["threshold"]

            # Store mapping label -> description (for SAM3 prompt)
            label_to_description[label] = description

            # Route to appropriate module
            if "ocr" in modules:
                local_classes[label] = "OCR"
            elif "gtsrb" in modules:
                local_classes[label] = "GTSRB"
            elif "rdd" in modules:
                local_classes[label] = "RDD"
            elif "local" in modules:
                local_classes[label] = "auto"

            if "deep" in modules:
                deep_classes[label] = True

            if "qwen" in modules:
                qwen_classes[label] = True

            # Use DESCRIPTION for SAM3 detection, not label
            detection_classes.append(description)

            if threshold is not None:
                class_thresholds[label] = threshold

        # Create reverse mapping: description -> label
        description_to_label = {v: k for k, v in label_to_description.items()}

        # Step 2: Load model if needed
        yield {"type": "progress", "stage": "model", "message": "Caricamento modello SAM3...", "percent": 5}
        await asyncio.sleep(0.05)

        processor = get_processor()

        # Pre-compute all text embeddings (runs in parallel with image loading)
        yield {"type": "progress", "stage": "text_cache", "message": "Pre-computing text embeddings...", "percent": 6}
        precompute_text_embeddings(processor, detection_classes)

        # Step 3: Load and preprocess image
        yield {"type": "progress", "stage": "image", "message": "Caricamento immagine...", "percent": 8}
        await asyncio.sleep(0.05)

        original_image = Image.open(image_path).convert("RGB")
        original_width, original_height = original_image.size

        # Step 3b: Crop panorama to remove car/sky
        yield {
            "type": "progress",
            "stage": "crop",
            "message": f"Taglio panorama (top {int(CROP_TOP_PERCENT*100)}%, bottom {int(CROP_BOTTOM_PERCENT*100)}%)...",
            "percent": 10
        }
        await asyncio.sleep(0.05)

        image, y_crop_offset = crop_panorama(original_image)
        width, height = image.size

        # Step 3c: Calculate optimal tiles
        if tiles == 0:
            # No tiling requested
            optimal_tiles, opt_cols, opt_rows = 0, 1, 1
        else:
            optimal_tiles, opt_cols, opt_rows = calculate_optimal_tiles(width, height)

        yield {
            "type": "progress",
            "stage": "tiles_calc",
            "message": f"Immagine: {width}x{height} → {optimal_tiles} tiles ({opt_cols}x{opt_rows})",
            "percent": 12
        }
        await asyncio.sleep(0.05)

        all_detections = []

        if tiles > 0:
            # Step 4: Split into tiles (use optimal layout)
            yield {
                "type": "progress",
                "stage": "tiles",
                "message": f"Suddivisione in {optimal_tiles} tiles ({opt_cols}x{opt_rows})...",
                "percent": 15
            }
            await asyncio.sleep(0.05)

            tile_list = split_image_into_tiles(image, optimal_tiles, custom_layout=(opt_cols, opt_rows))
            total_tiles = len(tile_list)
            total_classes = len(detection_classes)

            # Step 5: Process each tile
            # OPTIMIZATION: Calculate backbone ONCE per tile, reuse for all classes
            for tile_idx, (tile, x_offset, y_offset) in enumerate(tile_list):
                base_percent = 15 + (tile_idx / total_tiles) * 60  # 15-75%

                # Calculate backbone once for this tile
                inference_state = processor.set_image(tile)

                for class_idx, class_name in enumerate(detection_classes):
                    class_percent = base_percent + ((class_idx / total_classes) * (60 / total_tiles))
                    yield {
                        "type": "progress",
                        "stage": "detecting",
                        "message": f"Tile {tile_idx + 1}/{total_tiles} - Classe: {class_name}",
                        "percent": int(class_percent),
                        "tile": tile_idx + 1,
                        "total_tiles": total_tiles,
                        "current_class": class_name
                    }
                    await asyncio.sleep(0.01)

                    # Reuse backbone, use cached text embeddings
                    processor.reset_all_prompts(inference_state)
                    inference_state = set_text_prompt_cached(processor, inference_state, class_name)

                    if "scores" in inference_state and len(inference_state["scores"]) > 0:
                        scores = inference_state["scores"]
                        boxes = inference_state["boxes"]

                        for i, score in enumerate(scores):
                            score_val = score.item() if hasattr(score, "item") else float(score)
                            if score_val >= confidence:
                                box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                                x1, y1, x2, y2 = box

                                x1_adj = float(x1) + x_offset
                                y1_adj = float(y1) + y_offset + y_crop_offset  # Add crop offset
                                x2_adj = float(x2) + x_offset
                                y2_adj = float(y2) + y_offset + y_crop_offset  # Add crop offset

                                all_detections.append({
                                    "class": class_name,
                                    "score": round(score_val, 4),
                                    "bbox": [x1_adj, y1_adj, x2_adj, y2_adj],
                                    "bbox_normalized": [
                                        x1_adj / original_width,
                                        y1_adj / original_height,
                                        x2_adj / original_width,
                                        y2_adj / original_height
                                    ]
                                })
        else:
            # Process whole image (no tiles)
            # OPTIMIZATION: Calculate backbone ONCE, reuse for all classes
            total_classes = len(detection_classes)
            inference_state = processor.set_image(image)

            for class_idx, class_name in enumerate(detection_classes):
                class_percent = 15 + ((class_idx / total_classes) * 60)
                yield {
                    "type": "progress",
                    "stage": "detecting",
                    "message": f"Analisi classe: {class_name}",
                    "percent": int(class_percent),
                    "current_class": class_name
                }
                await asyncio.sleep(0.01)

                # Reuse backbone, use cached text embeddings
                processor.reset_all_prompts(inference_state)
                inference_state = set_text_prompt_cached(processor, inference_state, class_name)

                if "scores" in inference_state and len(inference_state["scores"]) > 0:
                    scores = inference_state["scores"]
                    boxes = inference_state["boxes"]

                    for i, score in enumerate(scores):
                        score_val = score.item() if hasattr(score, "item") else float(score)
                        if score_val >= confidence:
                            box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                            x1, y1, x2, y2 = box

                            # Add crop offset to y coordinates
                            y1_adj = float(y1) + y_crop_offset
                            y2_adj = float(y2) + y_crop_offset

                            all_detections.append({
                                "class": class_name,
                                "score": round(score_val, 4),
                                "bbox": [float(x1), y1_adj, float(x2), y2_adj],
                                "bbox_normalized": [
                                    float(x1) / original_width,
                                    y1_adj / original_height,
                                    float(x2) / original_width,
                                    y2_adj / original_height
                                ]
                            })

        yield {
            "type": "progress",
            "stage": "detection_done",
            "message": f"Detection completata: {len(all_detections)} oggetti trovati",
            "percent": 75,
            "found_count": len(all_detections)
        }
        await asyncio.sleep(0.05)

        # Map detections from description back to label
        for det in all_detections:
            desc = det["class"]
            if desc in description_to_label:
                det["class"] = description_to_label[desc]
                det["description"] = desc

        # Step 6: Local analysis (GTSRB/RDD) - BATCH PROCESSING - apply class threshold
        if local_classes:
            local_items = []
            for d in all_detections:
                if d["class"] in local_classes:
                    # Use class-specific threshold if set, otherwise no minimum
                    min_threshold = class_thresholds.get(d["class"])
                    if min_threshold is None or d.get("score", 0) >= min_threshold:
                        local_items.append(d)
            total_local = len(local_items)
            if total_local > 0:
                yield {
                    "type": "progress",
                    "stage": "local_analysis",
                    "message": f"Analisi locale batch: {total_local} items",
                    "percent": 76
                }
                await asyncio.sleep(0.05)

                # Load image ONCE for all detections
                local_image = Image.open(image_path).convert("RGB")

                # Group detections by class for efficient batching
                by_class = {}
                for det in local_items:
                    cls = det["class"]
                    if cls not in by_class:
                        by_class[cls] = []
                    by_class[cls].append(det)

                # Process each class group in batch
                processed = 0
                for cls, dets in by_class.items():
                    module = local_classes.get(cls, "auto")  # Get module name for this class
                    yield {
                        "type": "progress",
                        "stage": "local_analysis",
                        "message": f"Analisi batch ({module}): {cls} ({len(dets)} items)",
                        "percent": 76 + int((processed / total_local) * 8),
                        "current_class": cls
                    }
                    await asyncio.sleep(0.01)

                    # Batch inference on GPU with explicit module
                    results = batch_analyze_detections(local_image, dets, cls, module)

                    # Apply results - format: base_class.label
                    base_clean = cls.replace(" ", "_").lower()
                    for det, (label, conf) in zip(dets, results):
                        det["original_class"] = det["class"]
                        det["class"] = f"{base_clean}.{label}"
                        det["local_analyzed"] = True
                        det["local_confidence"] = round(conf, 4)
                        det["local_module"] = module if module != "auto" else ("RDD" if is_crack_class(cls) else "GTSRB")

                    processed += len(dets)

        # Step 7: Deep analysis (LLM) - apply class threshold or default 0.5
        if deep_classes:
            deep_items = []
            for d in all_detections:
                base_cls = d.get("original_class", d["class"])
                if base_cls in deep_classes and not d.get("local_analyzed"):
                    # Use class-specific threshold if set, otherwise default to 0.5
                    min_threshold = class_thresholds.get(base_cls, 0.5)
                    if d.get("score", 0) >= min_threshold:
                        deep_items.append(d)
            total_deep = len(deep_items)
            if total_deep > 0:
                yield {
                    "type": "progress",
                    "stage": "deep_analysis",
                    "message": f"Analisi LLM (Gemini): 0/{total_deep}",
                    "percent": 84
                }
                await asyncio.sleep(0.05)

                for idx, det in enumerate(deep_items):
                    yield {
                        "type": "progress",
                        "stage": "deep_analysis",
                        "message": f"Analisi LLM (Gemini): {idx + 1}/{total_deep}",
                        "percent": 84 + int((idx / total_deep) * 10),
                        "current_item": idx + 1,
                        "total_items": total_deep
                    }
                    await asyncio.sleep(0.01)

                    label = det.get("original_class", det["class"])
                    if "original_class" not in det:
                        det["original_class"] = det["class"]
                    detailed_class = deep_analyze_detection(image_path, det, label)
                    det["class"] = detailed_class
                    det["deep_analyzed"] = True

        # Step 7b: Qwen analysis (local VLM)
        if qwen_classes:
            qwen_items = []
            for d in all_detections:
                label = d.get("original_class", d["class"])
                if label in qwen_classes:
                    threshold = class_thresholds.get(label, 0.5)
                    if d.get("score", 0) >= threshold:
                        qwen_items.append(d)

            total_qwen = len(qwen_items)
            if total_qwen > 0:
                yield {
                    "type": "progress",
                    "stage": "qwen_analysis",
                    "message": f"Analisi Qwen VL (locale): 0/{total_qwen}",
                    "percent": 90
                }
                await asyncio.sleep(0.05)

                for idx, det in enumerate(qwen_items):
                    yield {
                        "type": "progress",
                        "stage": "qwen_analysis",
                        "message": f"Analisi Qwen VL (locale): {idx + 1}/{total_qwen}",
                        "percent": 90 + int((idx / total_qwen) * 4),
                        "current_item": idx + 1,
                        "total_items": total_qwen
                    }
                    await asyncio.sleep(0.01)

                    label = det.get("original_class", det["class"])
                    if "original_class" not in det:
                        det["original_class"] = det["class"]
                    detailed_class = qwen_analyze_detection(image_path, det, label)
                    det["class"] = detailed_class
                    det["qwen_analyzed"] = True

        # Step 8: Geolocation
        yield {"type": "progress", "stage": "geolocation", "message": "Calcolo coordinate GPS...", "percent": 95}
        await asyncio.sleep(0.05)

        poses = get_poses()
        camera_info = None
        if image_name in poses:
            geolocated = geolocate_detections(image_name, all_detections, poses)
            for det, geo in zip(all_detections, geolocated):
                det["latitude"] = geo.get("latitude")
                det["longitude"] = geo.get("longitude")
                det["distance_m"] = geo.get("distance_m")
                det["bearing_deg"] = geo.get("bearing_deg")
                det["geo_confidence"] = geo.get("confidence")

            pose = poses[image_name]
            camera_info = {
                "latitude": pose.latitude,
                "longitude": pose.longitude,
                "heading": pose.heading,
                "altitude": pose.altitude
            }

        # Filter detections by per-class thresholds
        if class_thresholds:
            filtered_detections = []
            for det in all_detections:
                # Get the base class for threshold lookup
                base_cls = det.get("original_class", det["class"])
                # For nested classes like "road_sign.limite_30", use the base part
                if "." in base_cls:
                    base_cls = base_cls.split(".")[0].replace("_", " ")
                min_threshold = class_thresholds.get(base_cls)
                if min_threshold is None:
                    # No threshold specified, keep detection
                    filtered_detections.append(det)
                elif det.get("score", 0) >= min_threshold:
                    filtered_detections.append(det)
            all_detections = filtered_detections

        # Sort by class name, then by score descending
        all_detections.sort(key=lambda x: (x["class"], -x["score"]))

        # Assign sequential IDs (after sorting)
        for idx, det in enumerate(all_detections, start=1):
            det["id"] = idx

        # Step 9: Done
        yield {"type": "progress", "stage": "complete", "message": "Completato!", "percent": 100}
        await asyncio.sleep(0.05)

        # Final result - use original dimensions (pre-crop)
        yield {
            "type": "result",
            "status": "success",
            "image_name": image_name,
            "image_width": original_width,
            "image_height": original_height,
            "crop_info": {
                "top_percent": CROP_TOP_PERCENT,
                "bottom_percent": CROP_BOTTOM_PERCENT,
                "y_offset": y_crop_offset,
                "cropped_height": height
            },
            "camera": camera_info,
            "parameters": {
                "confidence": confidence,
                "tiles": optimal_tiles,
                "tiles_layout": f"{opt_cols}x{opt_rows}",
                "classes_count": len(classes)
            },
            "detections_count": len(all_detections),
            "detections": all_detections
        }

    except Exception as e:
        yield {"type": "error", "message": str(e)}


@app.get("/api/detect-stream")
async def detect_stream(
    image_name: str = Query(...),
    classes: str = Query(...),
    confidence: float = Query(0.1),
    tiles: int = Query(24)
):
    """Run object detection with SSE progress updates."""
    class_list = [c.strip() for c in classes.split("|") if c.strip()]

    async def event_generator():
        async for progress in detect_with_progress(image_name, class_list, confidence, tiles):
            yield {"event": progress.get("type", "message"), "data": json.dumps(progress)}

    return EventSourceResponse(event_generator())


@app.get("/api/default-classes")
async def get_default_classes():
    """Get default class list."""
    return {"classes": DEFAULT_CLASSES.strip().split("\n")}


# Class Lists Management API
class ClassListRequest(BaseModel):
    name: str
    classes: list[str]


@app.get("/api/class-lists")
async def list_class_lists():
    """List all saved class lists."""
    lists = []
    for file_path in CLASS_LISTS_DIR.glob("*.txt"):
        lists.append({
            "name": file_path.stem,
            "file": file_path.name
        })
    lists.sort(key=lambda x: x["name"])
    return {"lists": lists}


@app.get("/api/class-lists/{list_name}")
async def get_class_list(list_name: str):
    """Get a specific class list."""
    file_path = CLASS_LISTS_DIR / f"{list_name}.txt"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Class list not found")

    content = file_path.read_text().strip()
    classes = [c.strip() for c in content.split("\n") if c.strip()]

    return {
        "name": list_name,
        "classes": classes
    }


@app.post("/api/class-lists")
async def save_class_list(request: ClassListRequest):
    """Save a class list."""
    # Sanitize name (remove special characters)
    safe_name = "".join(c for c in request.name if c.isalnum() or c in "._- ").strip()
    safe_name = safe_name.replace(" ", "_")

    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid list name")

    file_path = CLASS_LISTS_DIR / f"{safe_name}.txt"

    # Write classes to file
    content = "\n".join(request.classes)
    file_path.write_text(content)

    return {
        "status": "success",
        "name": safe_name,
        "classes_count": len(request.classes)
    }


@app.delete("/api/class-lists/{list_name}")
async def delete_class_list(list_name: str):
    """Delete a class list."""
    file_path = CLASS_LISTS_DIR / f"{list_name}.txt"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Class list not found")

    file_path.unlink()

    return {"status": "success", "deleted": list_name}


# Save Image with Detections
class SaveImageRequest(BaseModel):
    image_name: str
    detections: list[dict]
    confidence_threshold: float


@app.post("/api/save-image")
async def save_image_with_detections(request: SaveImageRequest):
    """Save image with bounding boxes drawn."""
    image_path = IMAGES_DIR / request.image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Load image with OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    height, width = img.shape[:2]

    # Color palette (BGR for OpenCV)
    colors_bgr = [
        (96, 69, 233), (128, 222, 74), (36, 191, 251), (246, 130, 59), (247, 85, 168),
        (153, 78, 236), (166, 184, 20), (22, 151, 249), (212, 182, 6), (246, 92, 139),
        (68, 68, 239), (94, 197, 34), (8, 179, 234), (241, 99, 99), (239, 70, 217)
    ]

    # Get unique classes for color assignment
    unique_classes = list(set(d["class"] for d in request.detections))
    class_colors = {cls: colors_bgr[i % len(colors_bgr)] for i, cls in enumerate(unique_classes)}

    # Draw detections
    for det in request.detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = class_colors[det["class"]]
        score = det["score"]

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Draw label background
        label = f"{det['class']}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    # Generate output filename
    base_name = Path(request.image_name).stem
    output_name = f"{base_name}_det_{request.confidence_threshold:.2f}.jpg"
    output_path = OUTPUT_DIR / output_name

    # Save image
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "status": "success",
        "output_path": str(output_path),
        "output_name": output_name,
        "detections_count": len(request.detections)
    }


class SaveDetectionsRequest(BaseModel):
    image_name: str
    detections: list[dict]
    camera: Optional[dict] = None


@app.post("/api/save-detections")
async def save_detections_csv(request: SaveDetectionsRequest):
    """Save detections as CSV file on server."""
    import csv

    image_stem = Path(request.image_name).stem
    csv_path = DETECTIONS_DIR / f"{image_stem}.csv"

    headers = [
        'id', 'image', 'class', 'original_class', 'score',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_width', 'bbox_height',
        'gps_lat', 'gps_lon', 'camera_lat', 'camera_lon', 'camera_heading',
        'deep_analyzed', 'local_analyzed', 'local_confidence'
    ]

    camera = request.camera or {}

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for det in request.detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            gps = det.get('gps', {})

            row = [
                det.get('id', ''),
                request.image_name,
                det.get('class', ''),
                det.get('original_class', ''),
                f"{det.get('score', 0):.4f}",
                int(x1), int(y1), int(x2), int(y2),
                int(x2 - x1), int(y2 - y1),
                f"{gps.get('lat', ''):.6f}" if gps.get('lat') else '',
                f"{gps.get('lon', ''):.6f}" if gps.get('lon') else '',
                f"{camera.get('lat', ''):.6f}" if camera.get('lat') else '',
                f"{camera.get('lon', ''):.6f}" if camera.get('lon') else '',
                f"{camera.get('heading', ''):.1f}" if camera.get('heading') else '',
                'true' if det.get('deep_analyzed') else 'false',
                'true' if det.get('local_analyzed') else 'false',
                f"{det.get('local_confidence', ''):.4f}" if det.get('local_confidence') else ''
            ]
            writer.writerow(row)

    return {"status": "success", "csv_path": str(csv_path), "detections_count": len(request.detections)}


@app.get("/api/load-detections/{image_name}")
async def load_detections_csv(image_name: str):
    """Load detections from CSV file if exists."""
    import csv

    image_stem = Path(image_name).stem
    csv_path = DETECTIONS_DIR / f"{image_stem}.csv"

    if not csv_path.exists():
        return {"exists": False, "detections": [], "camera": None}

    detections = []
    camera = None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            det = {
                'id': int(row['id']) if row['id'] else None,
                'class': row['class'],
                'original_class': row['original_class'] if row['original_class'] else None,
                'score': float(row['score']) if row['score'] else 0,
                'bbox': [float(row['bbox_x1']), float(row['bbox_y1']),
                         float(row['bbox_x2']), float(row['bbox_y2'])],
                'deep_analyzed': row['deep_analyzed'] == 'true',
                'local_analyzed': row['local_analyzed'] == 'true',
            }
            if row['gps_lat'] and row['gps_lon']:
                det['gps'] = {'lat': float(row['gps_lat']), 'lon': float(row['gps_lon'])}
            if row['local_confidence']:
                det['local_confidence'] = float(row['local_confidence'])
            detections.append(det)

            if camera is None and row['camera_lat'] and row['camera_lon']:
                camera = {
                    'lat': float(row['camera_lat']),
                    'lon': float(row['camera_lon']),
                    'heading': float(row['camera_heading']) if row['camera_heading'] else None
                }

    return {"exists": True, "detections": detections, "camera": camera}


@app.get("/api/thresholds")
async def get_thresholds():
    """Load saved default thresholds."""
    if not THRESHOLDS_FILE.exists():
        return {"thresholds": {}}

    with open(THRESHOLDS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {"thresholds": data}


class SaveThresholdsRequest(BaseModel):
    thresholds: dict


@app.post("/api/thresholds")
async def save_thresholds(request: SaveThresholdsRequest):
    """Save default thresholds."""
    with open(THRESHOLDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(request.thresholds, f, indent=2)
    return {"status": "success", "count": len(request.thresholds)}


@app.get("/api/model-status")
async def model_status():
    """Check if model is loaded."""
    return {
        "loaded": _processor is not None,
        "loading": _model_loading,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


# ============ 3D Reconstruction Endpoints ============

class Generate3DRequest(BaseModel):
    image_name: str
    bbox: list[float]
    detection_id: str = "0"
    force: bool = False  # If True, regenerate even if cached


class Check3DRequest(BaseModel):
    image_name: str
    bbox: list[float]
    detection_id: str = "0"


def get_ply_cache_path(image_path: str, bbox: list, detection_id: str) -> Path:
    """Get the cache path for a PLY file without generating it."""
    import hashlib
    cache_key = hashlib.md5(f"{image_path}_{bbox}_{detection_id}".encode()).hexdigest()
    return CACHE_3D_DIR / f"{cache_key}.ply"


@app.post("/api/check-3d")
async def check_3d_exists(request: Check3DRequest):
    """
    Check if a 3D reconstruction already exists in cache.

    Returns: { exists: bool, ply_url: str|null }
    """
    image_path = IMAGES_DIR / request.image_name
    if not image_path.exists():
        raise HTTPException(404, "Image not found")

    ply_path = get_ply_cache_path(str(image_path), request.bbox, request.detection_id)

    if ply_path.exists():
        return {
            "exists": True,
            "ply_url": f"/api/ply/{ply_path.name}"
        }
    else:
        return {
            "exists": False,
            "ply_url": None
        }


@app.get("/api/3d-status")
async def get_3d_status():
    """Check SAM-3D availability and cache status."""
    available, message = check_sam3d_available()
    cache = get_cache_status()
    return {
        "available": available,
        "message": message,
        "cache": cache
    }


@app.post("/api/generate-3d")
async def generate_3d(request: Generate3DRequest):
    """
    Generate 3D reconstruction for a detection.

    Body: { image_name, bbox: [x1,y1,x2,y2], detection_id, force: bool }
    Returns: { ply_url, cached: bool }
    """
    image_path = IMAGES_DIR / request.image_name
    if not image_path.exists():
        raise HTTPException(404, "Image not found")

    # Check if cached version exists
    ply_path = get_ply_cache_path(str(image_path), request.bbox, request.detection_id)

    # If cached and not forcing regeneration, return cached
    if ply_path.exists() and not request.force:
        return {
            "ply_url": f"/api/ply/{ply_path.name}",
            "cached": True
        }

    # Need to generate - check if SAM-3D is available
    available, message = check_sam3d_available()
    if not available:
        raise HTTPException(
            status_code=503,
            detail=f"SAM-3D not available: {message}"
        )

    # If forcing, delete existing cache
    if request.force and ply_path.exists():
        try:
            ply_path.unlink()
        except Exception:
            pass

    try:
        ply_path = generate_3d_from_detection(
            str(image_path),
            request.bbox,
            request.detection_id
        )
        return {
            "ply_url": f"/api/ply/{ply_path.name}",
            "cached": False
        }
    except Exception as e:
        raise HTTPException(500, f"3D generation failed: {str(e)}")


@app.get("/api/ply/{filename}")
async def serve_ply(filename: str):
    """Serve PLY file from cache."""
    # Validate filename (prevent path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    ply_path = CACHE_3D_DIR / filename
    if not ply_path.exists():
        raise HTTPException(404, "PLY not found")

    return FileResponse(
        ply_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )


@app.post("/api/cleanup-3d-cache")
async def cleanup_3d_cache_endpoint():
    """Clean up old PLY files from cache."""
    removed = cleanup_3d_cache()
    cache = get_cache_status()
    return {"removed": removed, "cache": cache}


# ============ Panoramic 360 Viewer Endpoints ============

@app.get("/api/gps-trajectory")
async def get_gps_trajectory():
    """Get full GPS trajectory for minimap."""
    poses = get_poses()
    if not poses:
        return {"count": 0, "bounds": None, "points": []}

    trajectory = []
    for name, pose in sorted(poses.items(), key=lambda x: x[0]):
        trajectory.append({
            "name": name,
            "lat": pose.latitude,
            "lon": pose.longitude,
            "heading": pose.heading,
            "altitude": pose.altitude,
            "timestamp": pose.timestamp
        })

    # Calculate bounding box for map centering
    lats = [p["lat"] for p in trajectory]
    lons = [p["lon"] for p in trajectory]
    bounds = {
        "minLat": min(lats), "maxLat": max(lats),
        "minLon": min(lons), "maxLon": max(lons),
        "centerLat": sum(lats) / len(lats),
        "centerLon": sum(lons) / len(lons)
    }

    return {"count": len(trajectory), "bounds": bounds, "points": trajectory}


@app.get("/api/panorama/{image_name}")
async def get_panorama_texture(
    image_name: str,
    resolution: str = Query("medium", regex="^(low|medium|high|full)$")
):
    """
    Get panorama image for 360 viewer with configurable resolution.

    Resolutions:
      - low: 2048x1024 (~200KB) - initial fast load
      - medium: 4096x2048 (~800KB) - default quality
      - high: 8192x4096 (~3MB) - high quality
      - full: 16000x8000 (~12MB) - original
    """
    image_path = IMAGES_DIR / image_name
    if not image_path.exists():
        raise HTTPException(404, "Image not found")

    resolution_sizes = {
        "low": 2048,
        "medium": 4096,
        "high": 8192,
        "full": 16000
    }
    max_width = resolution_sizes.get(resolution, 4096)

    img = Image.open(image_path)

    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    quality = 85 if resolution in ("low", "medium") else 92

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"}  # Cache 24h
    )


@app.get("/api/panorama-info/{image_name}")
async def get_panorama_info(image_name: str):
    """Get panorama metadata including GPS position and navigation."""
    image_path = IMAGES_DIR / image_name
    if not image_path.exists():
        raise HTTPException(404, "Image not found")

    poses = get_poses()
    pose = poses.get(image_name)

    # Get neighboring images for navigation
    image_names = sorted(poses.keys())
    current_idx = image_names.index(image_name) if image_name in image_names else -1

    prev_image = image_names[current_idx - 1] if current_idx > 0 else None
    next_image = image_names[current_idx + 1] if current_idx < len(image_names) - 1 else None

    return {
        "name": image_name,
        "index": current_idx,
        "total": len(image_names),
        "gps": {
            "lat": pose.latitude,
            "lon": pose.longitude,
            "heading": pose.heading,
            "altitude": pose.altitude
        } if pose else None,
        "navigation": {
            "prev": prev_image,
            "next": next_image
        }
    }


@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup."""
    print("Starting SAM3 Web Interface...")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Static directory: {static_dir}")
    # Model will be loaded on first request to avoid blocking startup


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
