"""
Shared utility functions for SAM3-Tool analysis modules.
"""
import re
from PIL import Image


def crop_detection(image: Image.Image, bbox: list, padding_percent: float = 0.1) -> Image.Image:
    """
    Crop the detection area from the image with optional padding.

    Args:
        image: PIL Image to crop from
        bbox: Bounding box as [x1, y1, x2, y2]
        padding_percent: Padding to add around bbox (default 0.1 = 10%)

    Returns:
        Cropped PIL Image
    """
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


def parse_threshold_prefix(class_name: str) -> tuple[str, float | None]:
    """
    Parse optional threshold prefix from class name.

    Format: "30 light poles" -> ("light poles", 0.30)
            "light poles" -> ("light poles", None)

    Returns:
        (class_name_without_prefix, threshold_or_none)
    """
    name = class_name.strip()
    # Match 2-digit number at start followed by space
    match = re.match(r'^(\d{2})\s+(.+)$', name)
    if match:
        threshold = int(match.group(1)) / 100.0
        return match.group(2), threshold
    return name, None
