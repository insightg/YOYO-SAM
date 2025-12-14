#!/usr/bin/env python3
"""
Road Crack Classification using YOLOv8 trained on RDD2022.
Uses YOLOv8-Small for offline classification of road damage.

Classes (RDD2022):
- D00: Longitudinal crack (crepa longitudinale)
- D10: Transverse crack (crepa trasversale)
- D20: Alligator crack (crepa a pelle di coccodrillo)
- D40: Pothole (buca)
"""

import torch
from PIL import Image
from pathlib import Path
from typing import Optional

# Global model cache
_model = None
_device = None

# RDD2022 class labels mapping to Italian descriptions
RDD_LABELS = {
    "D00": "crepa_longitudinale",
    "D10": "crepa_trasversale",
    "D20": "crepa_alligatore",
    "D40": "buca"
}

MODELS_DIR = Path(__file__).parent / "models" / "rdd"


def get_rdd_model():
    """Load or return cached RDD YOLOv8 model."""
    global _model, _device

    if _model is None:
        print("Loading RDD YOLOv8 model...")
        from ultralytics import YOLO

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = MODELS_DIR / "YOLOv8_Small_RDD.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"RDD model not found at {model_path}")

        _model = YOLO(str(model_path))
        print(f"RDD YOLOv8 model loaded on {_device}")

    return _model, _device


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


def classify_crack(image: Image.Image) -> tuple[str, float]:
    """
    Classify a road crack image using YOLOv8-RDD model.

    Args:
        image: PIL Image of the road crack

    Returns:
        (class_label, confidence)
    """
    model, device = get_rdd_model()

    # Run YOLOv8 inference on the cropped image
    results = model.predict(image, conf=0.25, verbose=False, device=device)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return "non_identificabile", 0.0

    # Get the detection with highest confidence
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax()
    class_id = int(boxes.cls[best_idx])
    confidence = float(boxes.conf[best_idx])

    # Get class name from model (D00, D10, D20, D40)
    class_name = results[0].names.get(class_id, f"D{class_id}")

    # Map to Italian label
    label = RDD_LABELS.get(class_name, f"tipo_{class_name}")

    return label, confidence


def classify_cracks_batch(images: list[Image.Image], batch_size: int = 8) -> list[tuple[str, float]]:
    """
    Classify multiple road crack images in batches using YOLOv8-RDD.

    Args:
        images: List of PIL Images of road cracks
        batch_size: Maximum batch size (YOLOv8 uses more VRAM)

    Returns:
        List of (class_label, confidence) tuples
    """
    if not images:
        return []

    model, device = get_rdd_model()
    all_results = []

    # Process in mini-batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]

        # YOLOv8 can process list of images natively
        results = model.predict(batch_images, conf=0.25, verbose=False, device=device)

        for result in results:
            if len(result.boxes) == 0:
                all_results.append(("non_identificabile", 0.0))
            else:
                best_idx = result.boxes.conf.argmax()
                class_id = int(result.boxes.cls[best_idx])
                confidence = float(result.boxes.conf[best_idx])
                class_name = result.names.get(class_id, f"D{class_id}")
                label = RDD_LABELS.get(class_name, f"tipo_{class_name}")
                all_results.append((label, confidence))

    return all_results


def crack_analyze_detection(
    image_path: Path,
    detection: dict,
    base_class: str
) -> tuple[str, float]:
    """
    Perform local analysis on a road crack detection.

    Args:
        image_path: Path to the full image
        detection: Detection dict with bbox
        base_class: The base class name (without "- local" suffix)

    Returns:
        (new_class_name, classification_confidence)
    """
    try:
        # Load and crop image
        image = Image.open(image_path).convert("RGB")
        cropped = crop_detection(image, detection["bbox"])

        # Classify with RDD model
        label, confidence = classify_crack(cropped)

        # Format: base_class.subclass
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.{label}", confidence

    except Exception as e:
        print(f"Error in crack analysis: {e}")
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.errore", 0.0


def is_crack_class(class_name: str) -> bool:
    """
    Check if a class name refers to road crack/damage.

    Returns:
        True if class should use RDD model
    """
    keywords = ["crack", "pothole", "buca", "crepa", "road damage"]
    return any(k in class_name.lower() for k in keywords)


if __name__ == "__main__":
    # Test
    print("Testing RDD YOLOv8 model...")

    # Test model loading
    print("\nLoading model...")
    model, device = get_rdd_model()
    print(f"Model ready on {device}")

    # Show available classes
    print("\nRDD Classes:")
    for code, label in RDD_LABELS.items():
        print(f"  {code}: {label}")

    # Test class detection
    test_classes = [
        "road crack",
        "pothole",
        "asphalt crack",
        "road sign",
        "stop sign"
    ]

    print("\nTest is_crack_class():")
    for cls in test_classes:
        is_crack = is_crack_class(cls)
        print(f"  {cls} -> is_crack={is_crack}")
