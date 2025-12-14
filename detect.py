#!/usr/bin/env python3
"""
SAM3 Object Detection Tool
Detects objects in images using SAM3 with text prompts from a classes file.
Outputs a CSV with detections and low-res images with bounding boxes.
Supports tiling for large panoramic images.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent / "sam3"))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_classes(classes_file: str) -> list[str]:
    """Load class names from a text file (one class per line)."""
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def split_image_into_tiles(image: Image.Image, num_tiles: int = 6) -> list[tuple[Image.Image, int, int]]:
    """
    Split an image into tiles.
    Supported: 6 (2x3), 12 (3x4), 24 (4x6)
    Returns list of (tile_image, offset_x, offset_y).
    """
    width, height = image.size

    # Calculate grid based on num_tiles
    if num_tiles == 6:
        cols, rows = 3, 2
    elif num_tiles == 12:
        cols, rows = 4, 3
    elif num_tiles == 24:
        cols, rows = 6, 4
    else:
        # Auto-calculate for arbitrary num_tiles
        import math
        aspect = width / height
        rows = int(math.sqrt(num_tiles / aspect))
        rows = max(1, rows)
        cols = num_tiles // rows

    tile_width = width // cols
    tile_height = height // rows

    tiles = []
    for row in range(rows):
        for col in range(cols):
            x_offset = col * tile_width
            y_offset = row * tile_height

            # Handle edge tiles (include remaining pixels)
            x_end = width if col == cols - 1 else x_offset + tile_width
            y_end = height if row == rows - 1 else y_offset + tile_height

            tile = image.crop((x_offset, y_offset, x_end, y_end))
            tiles.append((tile, x_offset, y_offset))

    return tiles


def process_tile(
    processor: Sam3Processor,
    tile: Image.Image,
    classes: list[str],
    confidence_threshold: float,
    x_offset: int,
    y_offset: int,
    original_width: int,
    original_height: int
) -> list[dict]:
    """Process a single tile and return detections with adjusted coordinates."""
    tile_width, tile_height = tile.size
    all_detections = []

    for class_name in classes:
        inference_state = processor.set_image(tile)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=class_name)

        if "scores" in inference_state and len(inference_state["scores"]) > 0:
            scores = inference_state["scores"]
            boxes = inference_state["boxes"]

            for i, score in enumerate(scores):
                score_val = score.item() if hasattr(score, "item") else float(score)
                if score_val >= confidence_threshold:
                    box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                    x1, y1, x2, y2 = box

                    # Adjust coordinates to original image space
                    x1_adj = float(x1) + x_offset
                    y1_adj = float(y1) + y_offset
                    x2_adj = float(x2) + x_offset
                    y2_adj = float(y2) + y_offset

                    all_detections.append({
                        "class": class_name,
                        "score": score_val,
                        "bbox": [x1_adj, y1_adj, x2_adj, y2_adj],
                        "bbox_normalized": [
                            x1_adj / original_width,
                            y1_adj / original_height,
                            x2_adj / original_width,
                            y2_adj / original_height
                        ]
                    })

    return all_detections


def get_image_files(input_dir: str) -> list[Path]:
    """Get all image files from a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(image_files)


def draw_detections(image: np.ndarray, detections: list[dict], output_size: int = 640) -> np.ndarray:
    """Draw bounding boxes and labels on an image and resize to low-res."""
    img = image.copy()
    height, width = img.shape[:2]

    # Generate colors for different classes
    np.random.seed(42)
    colors = {det["class"]: tuple(map(int, np.random.randint(0, 255, 3))) for det in detections}

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class"]
        score = det["score"]
        color = colors[class_name]

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw label background
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - label_h - 10), (int(x1) + label_w, int(y1)), color, -1)

        # Draw label text
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Resize to low-res maintaining aspect ratio
    scale = output_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return img_resized


def process_image(
    processor: Sam3Processor,
    image_path: Path,
    classes: list[str],
    confidence_threshold: float = 0.5,
    use_tiles: bool = False,
    num_tiles: int = 6
) -> list[dict]:
    """Process a single image and detect objects for all classes."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    all_detections = []

    if use_tiles:
        # Process image in tiles
        tiles = split_image_into_tiles(image, num_tiles)
        for tile_idx, (tile, x_offset, y_offset) in enumerate(tiles):
            tile_detections = process_tile(
                processor, tile, classes, confidence_threshold,
                x_offset, y_offset, width, height
            )
            all_detections.extend(tile_detections)
    else:
        # Process whole image
        for class_name in classes:
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=class_name)

            if "scores" in inference_state and len(inference_state["scores"]) > 0:
                scores = inference_state["scores"]
                boxes = inference_state["boxes"]

                for i, score in enumerate(scores):
                    score_val = score.item() if hasattr(score, "item") else float(score)
                    if score_val >= confidence_threshold:
                        box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                        x1, y1, x2, y2 = box

                        all_detections.append({
                            "class": class_name,
                            "score": score_val,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "bbox_normalized": [
                                float(x1) / width,
                                float(y1) / height,
                                float(x2) / width,
                                float(y2) / height
                            ]
                        })

    return all_detections


def main():
    parser = argparse.ArgumentParser(description="SAM3 Object Detection Tool")
    parser.add_argument("--input", "-i", required=True, help="Input directory with images")
    parser.add_argument("--classes", "-c", required=True, help="Text file with class names (one per line)")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--confidence", "-t", type=float, default=0.5, help="Confidence threshold (0-1)")
    parser.add_argument("--output-size", "-s", type=int, default=640, help="Output image size (max dimension)")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--tiles", type=int, default=0, help="Split image into N tiles (0=disabled, 6=2x3 grid)")
    args = parser.parse_args()

    use_tiles = args.tiles > 0

    # Validate inputs
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' not found")
        sys.exit(1)

    if not os.path.isfile(args.classes):
        print(f"Error: Classes file '{args.classes}' not found")
        sys.exit(1)

    # Create output directories
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load classes
    classes = load_classes(args.classes)
    print(f"Loaded {len(classes)} classes: {', '.join(classes)}")

    # Get image files
    image_files = get_image_files(args.input)
    if not image_files:
        print(f"Error: No image files found in '{args.input}'")
        sys.exit(1)
    print(f"Found {len(image_files)} images to process")
    if use_tiles:
        print(f"Tiling enabled: splitting each image into {args.tiles} tiles")

    # Initialize model
    print("Loading SAM3 model...")
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    print("Model loaded successfully")

    # Process images
    csv_path = output_dir / "detections.csv"
    csv_fields = ["image", "class", "score", "x1", "y1", "x2", "y2", "x1_norm", "y1_norm", "x2_norm", "y2_norm"]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        total_detections = 0

        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Detect objects
                detections = process_image(processor, image_path, classes, args.confidence, use_tiles, args.tiles)
                total_detections += len(detections)

                # Write to CSV
                for det in detections:
                    writer.writerow({
                        "image": image_path.name,
                        "class": det["class"],
                        "score": f"{det['score']:.4f}",
                        "x1": f"{det['bbox'][0]:.1f}",
                        "y1": f"{det['bbox'][1]:.1f}",
                        "x2": f"{det['bbox'][2]:.1f}",
                        "y2": f"{det['bbox'][3]:.1f}",
                        "x1_norm": f"{det['bbox_normalized'][0]:.4f}",
                        "y1_norm": f"{det['bbox_normalized'][1]:.4f}",
                        "x2_norm": f"{det['bbox_normalized'][2]:.4f}",
                        "y2_norm": f"{det['bbox_normalized'][3]:.4f}"
                    })

                # Generate annotated image
                if detections:
                    img = cv2.imread(str(image_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    annotated = draw_detections(img, detections, args.output_size)
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    output_image_path = images_dir / f"{image_path.stem}_detected.jpg"
                    cv2.imwrite(str(output_image_path), annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            except Exception as e:
                print(f"\nError processing {image_path.name}: {e}")
                continue

    print(f"\nDone! Processed {len(image_files)} images")
    print(f"Total detections: {total_detections}")
    print(f"CSV output: {csv_path}")
    print(f"Annotated images: {images_dir}")


if __name__ == "__main__":
    main()
