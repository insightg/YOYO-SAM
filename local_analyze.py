#!/usr/bin/env python3
"""
Local Object Classification Module
Routes to appropriate model based on class type:
- Traffic signs (GTSRB) -> bazyl/gtsrb-model (43 classes)
- Road cracks (RDD2022) -> YOLOv8-Small-RDD (4 classes: D00, D10, D20, D40)
"""

import torch
from PIL import Image
from pathlib import Path
from typing import Optional

# Import crack analysis module
from crack_analyze import is_crack_class, crack_analyze_detection, classify_cracks_batch

# Global model cache for GTSRB
_processor = None
_model = None
_device = None

# GTSRB class labels (German Traffic Sign Recognition Benchmark)
# Mapping from class index to Italian descriptions
GTSRB_LABELS = {
    0: "limite_velocita_20",
    1: "limite_velocita_30",
    2: "limite_velocita_50",
    3: "limite_velocita_60",
    4: "limite_velocita_70",
    5: "limite_velocita_80",
    6: "fine_limite_velocita_80",
    7: "limite_velocita_100",
    8: "limite_velocita_120",
    9: "divieto_sorpasso",
    10: "divieto_sorpasso_veicoli_pesanti",
    11: "intersezione_precedenza_destra",
    12: "strada_prioritaria",
    13: "dare_precedenza",
    14: "stop",
    15: "divieto_transito",
    16: "divieto_veicoli_pesanti",
    17: "divieto_accesso",
    18: "pericolo_generico",
    19: "curva_pericolosa_sinistra",
    20: "curva_pericolosa_destra",
    21: "doppia_curva",
    22: "strada_dissestata",
    23: "strada_sdrucciolevole",
    24: "carreggiata_ristretta_destra",
    25: "lavori_in_corso",
    26: "semaforo",
    27: "attraversamento_pedonale",
    28: "attraversamento_bambini",
    29: "attraversamento_ciclisti",
    30: "pericolo_neve_ghiaccio",
    31: "attraversamento_animali",
    32: "fine_divieti",
    33: "direzione_obbligatoria_destra",
    34: "direzione_obbligatoria_sinistra",
    35: "direzione_obbligatoria_dritto",
    36: "direzione_obbligatoria_dritto_destra",
    37: "direzione_obbligatoria_dritto_sinistra",
    38: "passaggio_obbligatorio_destra",
    39: "passaggio_obbligatorio_sinistra",
    40: "rotatoria",
    41: "fine_divieto_sorpasso",
    42: "fine_divieto_sorpasso_veicoli_pesanti",
}


def get_model():
    """Load or return cached GTSRB model."""
    global _processor, _model, _device

    if _model is None:
        print("Loading GTSRB model (bazyl/gtsrb-model)...")
        from transformers import ViTImageProcessor, ViTForImageClassification

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = ViTImageProcessor.from_pretrained("bazyl/gtsrb-model")
        _model = ViTForImageClassification.from_pretrained("bazyl/gtsrb-model")
        _model.to(_device)
        _model.eval()
        print(f"GTSRB model loaded on {_device}")

    return _processor, _model, _device


def crop_detection(image: Image.Image, bbox: list, padding_percent: float = 0.1) -> Image.Image:
    """Crop the detection area from the image with optional padding."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Add padding
    pad_x = width * padding_percent
    pad_y = height * padding_percent

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width, x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)

    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def classify_traffic_sign(image: Image.Image) -> tuple[str, float]:
    """
    Classify a traffic sign image using GTSRB model.

    Args:
        image: PIL Image of the traffic sign

    Returns:
        (class_label, confidence)
    """
    processor, model, device = get_model()

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = logits.argmax(-1).item()
        confidence = probs[0, predicted_idx].item()

    # Get label
    label = GTSRB_LABELS.get(predicted_idx, f"classe_{predicted_idx}")

    return label, confidence


def classify_traffic_signs_batch(images: list[Image.Image], batch_size: int = 16) -> list[tuple[str, float]]:
    """
    Classify multiple traffic sign images in batches for GPU efficiency.

    Args:
        images: List of PIL Images of traffic signs
        batch_size: Maximum batch size to avoid OOM

    Returns:
        List of (class_label, confidence) tuples - label only, not prefixed
    """
    if not images:
        return []

    processor, model, device = get_model()
    all_results = []

    # Process in mini-batches to avoid OOM
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]

        # Preprocess all images in batch
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Single batch inference on GPU
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_indices = outputs.logits.argmax(dim=-1)

            # Get confidence for each prediction
            for j, pred_idx in enumerate(predicted_indices):
                idx = pred_idx.item()
                conf = probs[j, idx].item()
                label = GTSRB_LABELS.get(idx, f"classe_{idx}")
                all_results.append((label, conf))

    return all_results


def batch_analyze_detections(
    image: Image.Image,
    detections: list[dict],
    base_class: str,
    module: str = "auto"
) -> list[tuple[str, float]]:
    """
    Analyze multiple detections from the same image in batch.

    Args:
        image: PIL Image (already loaded)
        detections: List of detection dicts with bbox
        base_class: The base class name
        module: Explicit module name ("GTSRB", "RDD") or "auto" for auto-detect

    Returns:
        List of (class_label, confidence) tuples
    """
    if not detections:
        return []

    # Pre-crop all detections
    crops = [crop_detection(image, det["bbox"]) for det in detections]

    # Route to appropriate batch classifier based on explicit module or auto-detect
    if module == "RDD" or (module == "auto" and is_crack_class(base_class)):
        return classify_cracks_batch(crops)
    else:
        # Default to GTSRB for traffic signs
        return classify_traffic_signs_batch(crops)


def local_analyze_detection(
    image_path: Path,
    detection: dict,
    base_class: str,
    module: str = "auto"
) -> tuple[str, float]:
    """
    Perform local analysis on a detection, routing to appropriate model.

    Routes to:
    - RDD2022 (YOLOv8) for cracks, potholes, road damage
    - GTSRB (ViT) for traffic signs (default)

    Args:
        image_path: Path to the full image
        detection: Detection dict with bbox
        base_class: The base class name (without suffix)
        module: Explicit module ("GTSRB", "RDD") or "auto" for auto-detect

    Returns:
        (new_class_name, classification_confidence)
    """
    # Route to RDD model if explicitly requested or auto-detected
    if module == "RDD" or (module == "auto" and is_crack_class(base_class)):
        return crack_analyze_detection(image_path, detection, base_class)

    # Default: use GTSRB for traffic signs
    try:
        # Load and crop image
        image = Image.open(image_path).convert("RGB")
        cropped = crop_detection(image, detection["bbox"])

        # Classify with GTSRB model
        label, confidence = classify_traffic_sign(cropped)

        # Format: base_class.subclass (es. road_sign.limite_velocita_50)
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.{label}", confidence

    except Exception as e:
        print(f"Error in local analysis: {e}")
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.errore", 0.0


def is_local_class(class_name: str) -> tuple[bool, str, str]:
    """
    Check if a class requires local analysis.

    Supported suffixes:
    - "- GTSRB" for traffic signs (German Traffic Sign Recognition)
    - "- RDD" for road damage detection (cracks, potholes)
    - "- local" (legacy, auto-routes based on class name)

    Returns:
        (is_local, base_class_name, module_name)
        module_name is "GTSRB", "RDD", or "auto" for legacy local
    """
    name_lower = class_name.strip().lower()

    # Check for specific module names
    if name_lower.endswith("- gtsrb"):
        base_class = class_name.rsplit("-", 1)[0].strip()
        return True, base_class, "GTSRB"

    if name_lower.endswith("- rdd"):
        base_class = class_name.rsplit("-", 1)[0].strip()
        return True, base_class, "RDD"

    # Legacy support for "- local" (auto-detect based on class name)
    if name_lower.endswith("- local"):
        base_class = class_name.rsplit("-", 1)[0].strip()
        return True, base_class, "auto"

    return False, class_name, ""


if __name__ == "__main__":
    # Test
    print("Testing GTSRB model...")

    # Test class detection
    test_classes = [
        "road sign - local",      # legacy, auto-detect
        "traffic sign - GTSRB",   # explicit GTSRB
        "road crack - RDD",       # explicit RDD
        "pothole - RDD",          # explicit RDD
        "stop sign",              # no suffix
        "speed limit sign - deep" # deep analysis
    ]

    print("\nTest is_local_class():")
    for cls in test_classes:
        is_local, base, module = is_local_class(cls)
        print(f"  {cls} -> local={is_local}, base='{base}', module='{module}'")

    # Test model loading
    print("\nLoading model...")
    processor, model, device = get_model()
    print(f"Model ready on {device}")
    print(f"Number of classes: {len(GTSRB_LABELS)}")
