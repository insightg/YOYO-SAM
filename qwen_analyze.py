#!/usr/bin/env python3
"""
Qwen Vision-Language Analysis Module
Local alternative to deep_analyze using Qwen3-VL-30B-A3B-Instruct
"""

import re
import gc
import torch
from PIL import Image
from pathlib import Path
from typing import Optional

from utils import crop_detection, parse_threshold_prefix

# Global model cache (full precision)
_model = None
_processor = None
_qwen_loaded = False

# Global model cache (4-bit quantized)
_model_q = None
_processor_q = None
_qwenq_loaded = False

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


def _unload_sam_for_qwen():
    """Unload SAM model from GPU to make room for Qwen."""
    try:
        # Import here to avoid circular imports
        import app
        if hasattr(app, 'unload_sam_from_gpu'):
            app.unload_sam_from_gpu()
    except Exception as e:
        print(f"Note: Could not unload SAM: {e}")


def _reload_sam_after_qwen():
    """Reload SAM model to GPU after Qwen is done."""
    try:
        import app
        if hasattr(app, 'reload_sam_to_gpu'):
            app.reload_sam_to_gpu()
    except Exception as e:
        print(f"Note: Could not reload SAM: {e}")


def unload_qwen():
    """Unload Qwen model from GPU to free memory."""
    global _model, _processor, _qwen_loaded

    if _model is None:
        return

    print("Unloading Qwen from GPU...")
    try:
        del _model
        del _processor
        _model = None
        _processor = None
        _qwen_loaded = False

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Qwen unloaded, GPU memory freed")
    except Exception as e:
        print(f"Error unloading Qwen: {e}")


def get_model():
    """Load or return cached Qwen3-VL model."""
    global _model, _processor, _qwen_loaded

    if _model is None:
        # First, unload SAM from GPU to make room
        _unload_sam_for_qwen()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        print(f"Loading {model_name}...")

        _processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        _qwen_loaded = True
        print("Qwen3-VL loaded successfully")

    return _model, _processor


def is_qwen_loaded() -> bool:
    """Check if Qwen model is currently loaded."""
    return _qwen_loaded


def get_model_quantized():
    """Load or return cached Qwen3-VL model with 4-bit quantization.

    This version uses ~8-10GB VRAM instead of ~20GB.
    Still requires SAM unload due to memory constraints.
    """
    global _model_q, _processor_q, _qwenq_loaded

    if _model_q is None:
        # Unload SAM from GPU to make room
        _unload_sam_for_qwen()

        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        print(f"Loading {model_name} (4-bit quantized)...")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        _processor_q = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _model_q = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        _qwenq_loaded = True
        print("Qwen3-VL (4-bit) loaded successfully")

    return _model_q, _processor_q


def unload_qwen_quantized():
    """Unload quantized Qwen model from GPU to free memory."""
    global _model_q, _processor_q, _qwenq_loaded

    if _model_q is None:
        return

    print("Unloading Qwen (4-bit) from GPU...")
    try:
        del _model_q
        del _processor_q
        _model_q = None
        _processor_q = None
        _qwenq_loaded = False

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Qwen (4-bit) unloaded, GPU memory freed")
    except Exception as e:
        print(f"Error unloading Qwen (4-bit): {e}")


def is_qwenq_loaded() -> bool:
    """Check if quantized Qwen model is currently loaded."""
    return _qwenq_loaded


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

        # Generate response - disable autocast to avoid dtype conflicts with SAM context
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            # Ensure inputs are in the correct dtype
            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                inputs['pixel_values'] = inputs.pixel_values.to(torch.bfloat16)

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


def analyze_with_qwen_quantized(image: Image.Image, category: str) -> str:
    """Analyze image using 4-bit quantized Qwen3-VL model."""
    try:
        model, processor = get_model_quantized()

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
        print(f"Qwen (4-bit) analysis error: {e}")
        return "error"


def qwenq_analyze_detection(
    image_path: Path,
    detection: dict,
    base_class: str
) -> str:
    """
    Analyze detection using 4-bit quantized Qwen3-VL model.

    This version uses less VRAM and can coexist with SAM.

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

        # Analyze with quantized Qwen
        subclass = analyze_with_qwen_quantized(cropped, base_class)

        # Format result
        base_clean = base_class.replace(" ", "_").lower()

        # If not identifiable, return base_class.generic
        if subclass in ("unidentified", "error", "generic"):
            return f"{base_clean}.generic"

        return f"{base_clean}.{subclass}"

    except Exception as e:
        print(f"Error in Qwen (4-bit) analysis: {e}")
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.generic"


def is_qwen_class(class_name: str) -> tuple[bool, str, float | None]:
    """
    Check if a class requires Qwen analysis (ends with '- qwen').

    Returns:
        (is_qwen, base_class_name, threshold_or_none)
    """
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
