#!/usr/bin/env python3
"""
Tree Species Classification Module using BioCLIP-2
Uses imageomics/bioclip-2 for zero-shot tree species classification
"""

import gc
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import Optional

from utils import crop_detection, parse_threshold_prefix

# Global model cache
_model = None
_preprocess = None
_tokenizer = None
_bioclip_loaded = False

# Italian tree species with scientific names for better BioCLIP accuracy
# Format: (scientific_name, common_name) - BioCLIP trained on taxonomy
TREE_SPECIES_DATA = [
    # Querce (Oaks)
    ("Quercus robur", "farnia"),
    ("Quercus petraea", "rovere"),
    ("Quercus ilex", "leccio"),
    ("Quercus suber", "sughera"),
    ("Quercus pubescens", "roverella"),
    ("Quercus cerris", "cerro"),
    # Conifere
    ("Pinus pinea", "pino_domestico"),
    ("Pinus pinaster", "pino_marittimo"),
    ("Pinus halepensis", "pino_aleppo"),
    ("Pinus nigra", "pino_nero"),
    ("Cupressus sempervirens", "cipresso"),
    ("Cedrus libani", "cedro_libano"),
    ("Cedrus atlantica", "cedro_atlante"),
    ("Abies alba", "abete_bianco"),
    ("Picea abies", "abete_rosso"),
    # Latifoglie comuni
    ("Platanus hispanica", "platano"),
    ("Tilia cordata", "tiglio_selvatico"),
    ("Tilia platyphyllos", "tiglio_nostrano"),
    ("Acer campestre", "acero_campestre"),
    ("Acer pseudoplatanus", "acero_montano"),
    ("Acer platanoides", "acero_riccio"),
    ("Fraxinus excelsior", "frassino_maggiore"),
    ("Fraxinus ornus", "orniello"),
    ("Ulmus minor", "olmo_campestre"),
    ("Populus nigra", "pioppo_nero"),
    ("Populus alba", "pioppo_bianco"),
    ("Salix alba", "salice_bianco"),
    ("Fagus sylvatica", "faggio"),
    ("Castanea sativa", "castagno"),
    ("Carpinus betulus", "carpino_bianco"),
    ("Betula pendula", "betulla"),
    ("Alnus glutinosa", "ontano_nero"),
    ("Robinia pseudoacacia", "robinia"),
    # Mediterranee
    ("Olea europaea", "olivo"),
    ("Ficus carica", "fico"),
    ("Laurus nobilis", "alloro"),
    ("Cercis siliquastrum", "albero_giuda"),
    ("Ceratonia siliqua", "carrubo"),
    ("Nerium oleander", "oleandro"),
    # Palme
    ("Phoenix canariensis", "palma_canarie"),
    ("Phoenix dactylifera", "palma_dattero"),
    ("Trachycarpus fortunei", "palma_cinese"),
    ("Washingtonia filifera", "palma_california"),
    # Fruttiferi ornamentali
    ("Prunus cerasifera", "mirabolano"),
    ("Prunus avium", "ciliegio"),
    ("Malus domestica", "melo"),
    ("Pyrus communis", "pero"),
    ("Citrus limon", "limone"),
    ("Citrus sinensis", "arancio"),
    # Altri ornamentali
    ("Magnolia grandiflora", "magnolia"),
    ("Ginkgo biloba", "ginkgo"),
    ("Aesculus hippocastanum", "ippocastano"),
    ("Juglans regia", "noce"),
    ("Morus alba", "gelso_bianco"),
    ("Catalpa bignonioides", "catalpa"),
    ("Lagerstroemia indica", "lagerstroemia"),
    ("Liquidambar styraciflua", "liquidambar"),
    ("Celtis australis", "bagolaro"),
    ("Ailanthus altissima", "ailanto"),
]

# Scientific names for BioCLIP (primary)
TREE_SPECIES = [species[0] for species in TREE_SPECIES_DATA]

# Mapping scientific -> common name (for output)
SPECIES_TO_COMMON = {species[0]: species[1] for species in TREE_SPECIES_DATA}


def _unload_sam_for_bioclip():
    """Unload SAM model from GPU to make room for BioCLIP."""
    try:
        import app
        if hasattr(app, 'unload_sam_from_gpu'):
            app.unload_sam_from_gpu()
    except Exception as e:
        print(f"Note: Could not unload SAM: {e}")


def _reload_sam_after_bioclip():
    """Reload SAM model to GPU after BioCLIP is done."""
    try:
        import app
        if hasattr(app, 'reload_sam_to_gpu'):
            app.reload_sam_to_gpu()
    except Exception as e:
        print(f"Note: Could not reload SAM: {e}")


def get_model():
    """Load or return cached BioCLIP-2 model."""
    global _model, _preprocess, _tokenizer, _bioclip_loaded

    if _model is None:
        print("Loading BioCLIP-2 model...")

        _model, _, _preprocess = open_clip.create_model_and_transforms(
            'hf-hub:imageomics/bioclip-2'
        )
        _tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

        # Move to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()

        _bioclip_loaded = True
        print(f"BioCLIP-2 loaded on {device}")

    return _model, _preprocess, _tokenizer


def unload_bioclip():
    """Unload BioCLIP model from GPU to free memory."""
    global _model, _preprocess, _tokenizer, _bioclip_loaded

    if _model is None:
        return

    print("Unloading BioCLIP from GPU...")
    try:
        del _model
        del _preprocess
        del _tokenizer
        _model = None
        _preprocess = None
        _tokenizer = None
        _bioclip_loaded = False

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("BioCLIP unloaded, GPU memory freed")
    except Exception as e:
        print(f"Error unloading BioCLIP: {e}")


def is_bioclip_loaded() -> bool:
    """Check if BioCLIP model is currently loaded."""
    return _bioclip_loaded


def classify_tree(image: Image.Image, species_list: list = None) -> tuple[str, float]:
    """
    Classify tree species using BioCLIP-2.

    Args:
        image: PIL Image of the tree
        species_list: Optional list of species to classify against

    Returns:
        (species_name, confidence_score)
    """
    if species_list is None:
        species_list = TREE_SPECIES

    model, preprocess, tokenizer = get_model()
    device = next(model.parameters()).device

    # Preprocess image
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Tokenize species names
    text_tokens = tokenizer(species_list).to(device)

    with torch.no_grad():
        # Get embeddings
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get best match
        confidence, idx = similarity[0].max(dim=0)
        species = species_list[idx.item()]

    return species, confidence.item()


def classify_trees_batch(images: list[Image.Image], species_list: list = None, batch_size: int = 16) -> list[tuple[str, float]]:
    """
    Classify multiple tree images in batch for GPU efficiency.

    Args:
        images: List of PIL Images of trees
        species_list: Optional list of species to classify against
        batch_size: Maximum batch size to avoid OOM

    Returns:
        List of (species_scientific_name, confidence) tuples
    """
    if not images:
        return []

    if species_list is None:
        species_list = TREE_SPECIES

    model, preprocess, tokenizer = get_model()
    device = next(model.parameters()).device

    # Pre-compute text features (same for all images)
    text_tokens = tokenizer(species_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_results = []

    # Process in mini-batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]

        # Preprocess all images in batch
        image_tensors = torch.stack([preprocess(img) for img in batch_images]).to(device)

        with torch.no_grad():
            # Encode all images at once
            image_features = model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity for all images
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Get best match for each image
            confidences, indices = similarity.max(dim=-1)

            for conf, idx in zip(confidences, indices):
                species = species_list[idx.item()]
                all_results.append((species, conf.item()))

    return all_results


def batch_analyze_trees(
    image: Image.Image,
    detections: list[dict],
    base_class: str
) -> list[str]:
    """
    Analyze multiple tree detections from the same image in batch.

    Args:
        image: PIL Image (already loaded)
        detections: List of detection dicts with bbox
        base_class: The base class name

    Returns:
        List of formatted class names (base_class.species)
    """
    if not detections:
        return []

    # Pre-crop all detections
    crops = [crop_detection(image, det["bbox"]) for det in detections]

    # Batch classify
    results = classify_trees_batch(crops)

    # Format results and update detections
    formatted = []
    base_clean = base_class.replace(" ", "_").lower()

    for i, (species_scientific, confidence) in enumerate(results):
        # Convert to common name
        species_common = SPECIES_TO_COMMON.get(species_scientific, species_scientific)
        species_clean = species_common.lower().replace(" ", "_")

        # Update detection with metadata
        detections[i]["tree_confidence"] = round(confidence, 3)
        detections[i]["tree_scientific"] = species_scientific

        formatted.append(f"{base_clean}.{species_clean}")

    return formatted


def tree_analyze_detection(
    image_path: Path,
    detection: dict,
    base_class: str,
    species_list: list = None
) -> str:
    """
    Analyze tree detection using BioCLIP-2.

    Args:
        image_path: Path to the full image
        detection: Detection dict with bbox
        base_class: The base class name (e.g., "tree")
        species_list: Optional list of species to classify against

    Returns:
        Formatted class name: base_class.species
    """
    try:
        # Load and crop image
        image = Image.open(image_path).convert("RGB")
        cropped = crop_detection(image, detection["bbox"])

        # Classify with BioCLIP (returns scientific name)
        species_scientific, confidence = classify_tree(cropped, species_list)

        # Convert to common Italian name if available
        species_common = SPECIES_TO_COMMON.get(species_scientific, species_scientific)
        species_clean = species_common.lower().replace(" ", "_")
        base_clean = base_class.replace(" ", "_").lower()

        # Store both names and confidence in detection
        detection["tree_confidence"] = round(confidence, 3)
        detection["tree_scientific"] = species_scientific

        return f"{base_clean}.{species_clean}"

    except Exception as e:
        print(f"BioCLIP analysis error: {e}")
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.generic"


def is_tree_class(class_name: str) -> tuple[bool, str, float | None]:
    """
    Check if a class requires tree analysis (ends with '- tree' or has 'tree' module).

    Returns:
        (is_tree, base_class_name, threshold_or_none)
    """
    name, threshold = parse_threshold_prefix(class_name)

    if name.lower().endswith("- tree"):
        base_class = name.rsplit("-", 1)[0].strip()
        return True, base_class, threshold

    return False, name, threshold


if __name__ == "__main__":
    print("Testing BioCLIP-2 Tree Classification Module...")

    # Test model loading
    print("\nLoading model...")
    try:
        model, preprocess, tokenizer = get_model()
        print("Model loaded successfully!")

        # Show available species
        print(f"\nConfigured for {len(TREE_SPECIES)} tree species")
        print("Sample species:", TREE_SPECIES[:5])

    except Exception as e:
        print(f"Error loading model: {e}")
