#!/usr/bin/env python3
"""
Qwen Vision-Language Analysis Module
Local alternative to deep_analyze using Qwen3-VL-30B-A3B-Instruct
"""

import re
import torch
from PIL import Image
from pathlib import Path
from typing import Optional

# Global model cache
_model = None
_processor = None

# Category-specific guidance (same as deep_analyze.py)
CATEGORY_HINTS = {
    "road sign": "speed_limit_XX, no_parking, yield, stop, one_way, no_entry, generic_danger, dangerous_curve, intersection, pedestrian_crossing, no_overtaking, priority_road, end_restriction",
    "street sign": "street_name, house_number, direction_indicator, locality, km_distance",
    "traffic light": "red, yellow, green, pedestrian_red, pedestrian_green, flashing_yellow, off, green_arrow_right, green_arrow_left",
    "speed limit sign": "limit_20, limit_30, limit_50, limit_60, limit_70, limit_80, limit_90, limit_110, limit_130",
    "road marking": "pedestrian_crossing, solid_line, dashed_line, direction_arrow, stop, yield, parking, no_parking, zigzag",
    "road crack": "longitudinal_crack, transverse_crack, branched_crack, pothole, subsidence, alligator_crack, deteriorated_joint",
    "pothole": "small_pothole, medium_pothole, large_pothole, depression, edge_failure",
    "manhole": "sewer_manhole, water_manhole, gas_manhole, electric_manhole, telecom_manhole, square_cover, round_cover",
    "guardrail": "metal_guardrail, concrete_guardrail, jersey_barrier, double_guardrail, guardrail_terminal",
    "pole": "light_pole, sign_pole, traffic_light_pole, electric_pole, telecom_pole",
    "curb": "standard_curb, lowered_curb, wheelchair_ramp, damaged_curb, sidewalk_edge",
    "sidewalk": "intact_sidewalk, damaged_sidewalk, uneven_pavement, broken_tiles, exposed_roots",
    "crosswalk": "zebra_crossing, raised_crossing, signalized_crossing, bicycle_crossing",
    "fire hydrant": "above_ground_hydrant, underground_hydrant, pillar_hydrant, wall_hydrant",
    "vehicle": "car, motorcycle, bicycle, truck, bus, van, scooter",
    "pedestrian": "adult_pedestrian, child_pedestrian, elderly_pedestrian, wheelchair_person, stroller_person",
    "tree": "deciduous_tree, conifer_tree, palm_tree, small_tree, large_tree, dead_tree, pruned_tree",
}


def get_model():
    """Load or return cached Qwen3-VL model."""
    global _model, _processor

    if _model is None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        print(f"Loading {model_name}...")

        _processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Qwen3-VL loaded successfully")

    return _model, _processor


def build_contextual_prompt(category: str) -> str:
    """Build context-aware prompt for classification (English for better quality)."""
    # Get category-specific hints
    category_lower = category.lower()
    hints = ""
    for key, value in CATEGORY_HINTS.items():
        if key in category_lower:
            hints = f"\nCommon subtypes for {category}: {value}"
            break

    prompt = f"""CONTEXT: This image contains an object detected as "{category}".
Your task is to identify the SPECIFIC SUBTYPE of this {category}.

IMPORTANT:
- Focus ONLY on the "{category}" object in the image
- IGNORE completely any other visible objects (vehicles, people, other elements)
- Classify ONLY the {category}, not other elements present
- If the {category} object is not clearly identifiable, respond "unidentified"
{hints}

RESPONSE FORMAT:
Reply with ONE SINGLE WORD or short phrase (max 3 words), using underscore instead of spaces.
DO NOT add explanations, only the subtype identifier.

What is the specific subtype of this {category}?"""

    return prompt


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


def clean_response(response: str) -> str:
    """Clean and normalize LLM response."""
    # Get first line only
    result = response.strip().split('\n')[0].strip()

    # Remove common prefixes
    result = re.sub(r'^(the |a |an |type:|subtype:|it is |this is )', '', result, flags=re.IGNORECASE)
    result = re.sub(r'^(il |la |lo |l\'|un |una |tipo:|sottotipo:)', '', result, flags=re.IGNORECASE)

    # Normalize: lowercase, replace invalid chars
    result = result.lower()
    result = re.sub(r'[^a-z0-9_]', '_', result)
    result = re.sub(r'_+', '_', result)
    result = result.strip('_')

    # Limit length
    if len(result) > 50:
        result = result[:50]

    return result if result else "unidentified"


def analyze_with_qwen(image: Image.Image, category: str) -> str:
    """Analyze image using local Qwen3-VL model."""
    try:
        model, processor = get_model()

        prompt = build_contextual_prompt(category)

        # Prepare messages in Qwen VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        # Decode response (skip input tokens)
        response = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return clean_response(response)

    except Exception as e:
        print(f"Qwen analysis error: {e}")
        return "error"


def qwen_analyze_detection(
    image_path: Path,
    detection: dict,
    base_class: str
) -> str:
    """
    Analyze detection using local Qwen3-VL model.

    Args:
        image_path: Path to the full image
        detection: Detection dict with bbox
        base_class: The base class name

    Returns:
        Formatted class name: base_class.subclass
    """
    try:
        # Load and crop image
        image = Image.open(image_path).convert("RGB")
        cropped = crop_detection(image, detection["bbox"])

        # Analyze with Qwen
        subclass = analyze_with_qwen(cropped, base_class)

        # Format result
        base_clean = base_class.replace(" ", "_").lower()

        # If not identifiable, return base_class.generic
        if subclass in ("unidentified", "error", "generic"):
            return f"{base_clean}.generic"

        return f"{base_clean}.{subclass}"

    except Exception as e:
        print(f"Error in Qwen analysis: {e}")
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.generic"


def is_qwen_class(class_name: str) -> tuple[bool, str, float | None]:
    """
    Check if a class requires Qwen analysis (ends with '- qwen').

    Returns:
        (is_qwen, base_class_name, threshold_or_none)
    """
    from local_analyze import parse_threshold_prefix

    name, threshold = parse_threshold_prefix(class_name)

    if name.lower().endswith("- qwen"):
        base_class = name.rsplit("-", 1)[0].strip()
        return True, base_class, threshold

    return False, name, threshold


if __name__ == "__main__":
    print("Testing Qwen3-VL module...")

    # Test class detection
    test_classes = [
        "road sign - qwen",
        "40 tree - qwen",
        "traffic light",
        "50 vehicle - qwen",
    ]

    print("\nTest classi:")
    for cls in test_classes:
        is_qwen, base, threshold = is_qwen_class(cls)
        print(f"  {cls} -> qwen={is_qwen}, base='{base}', threshold={threshold}")

    # Show example prompt
    print("\n--- Example prompt for 'road sign' ---")
    print(build_contextual_prompt("road sign"))

    # Test model loading (optional)
    print("\nLoading model (this may take a while on first run)...")
    try:
        model, processor = get_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
