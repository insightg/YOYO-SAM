#!/usr/bin/env python3
"""
OCR Text Extraction Module
Extracts alphanumeric text from detected objects using EasyOCR.
"""

import re
import torch
from PIL import Image
from pathlib import Path

# Global model cache
_reader = None


def get_ocr_reader():
    """Load or return cached EasyOCR reader."""
    global _reader
    if _reader is None:
        import easyocr
        # Use CPU for OCR to avoid BFloat16/GPU conflicts with SAM3
        # OCR is fast enough on CPU for cropped detection images
        _reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR loaded (CPU mode)")
    return _reader


def crop_detection(image: Image.Image, bbox: list, padding: float = 0.1) -> Image.Image:
    """Crop detection area with padding."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * padding, h * padding

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width, x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)

    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def extract_text(image: Image.Image) -> tuple[str, float]:
    """
    Extract alphanumeric text from image using OCR.

    Returns:
        (text_string, confidence)
    """
    import numpy as np
    reader = get_ocr_reader()

    img_array = np.array(image)
    results = reader.readtext(img_array)

    if not results:
        return "nd", 0.0  # Non Detected

    # Combine all detected texts
    texts = []
    confidences = []
    for (bbox, text, conf) in results:
        # Clean: keep only letters, numbers
        clean = re.sub(r'[^a-zA-Z0-9]', '', text)
        if clean:
            texts.append(clean)
            confidences.append(conf)

    if not texts:
        return "nd", 0.0

    combined = "_".join(texts).lower()
    avg_conf = sum(confidences) / len(confidences)

    return combined, avg_conf


def extract_text_batch(images: list[Image.Image]) -> list[tuple[str, float]]:
    """Batch extract text from multiple images."""
    return [extract_text(img) for img in images]


def ocr_analyze_detection(
    image_path: Path,
    detection: dict,
    base_class: str
) -> tuple[str, float]:
    """
    Perform OCR on detection to extract text.

    Args:
        image_path: Path to the full image
        detection: Detection dict with bbox
        base_class: The base class name (without suffix)

    Returns:
        (base_class.text, confidence)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        cropped = crop_detection(image, detection["bbox"])

        text, confidence = extract_text(cropped)

        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.{text}", confidence

    except Exception as e:
        print(f"OCR error: {e}")
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.errore", 0.0


if __name__ == "__main__":
    print("Testing OCR module...")
    reader = get_ocr_reader()
    print("OCR reader ready")
