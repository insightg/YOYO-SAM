"""
SAM-3D-Objects Integration Module

Provides 3D reconstruction of detected objects using Facebook's SAM-3D-Objects model.

Requirements:
- SAM-3D-Objects repository cloned to ./sam3d
- Environment setup following sam3d/doc/setup.md
- HuggingFace checkpoints downloaded to sam3d/checkpoints/hf/
- NVIDIA GPU with at least 32GB VRAM

Setup instructions:
1. Clone: Already done at ./sam3d
2. Create conda environment: mamba env create -f sam3d/environments/default.yml
3. Install dependencies: See sam3d/doc/setup.md
4. Download checkpoints from HuggingFace (requires account approval)
"""

import os

# CRITICAL: Set these env vars BEFORE any imports that might load SAM-3D
# This prevents inference_pipeline.py from setting flash_attn based on GPU type
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ["LIDRA_SKIP_INIT"] = "true"

# Add compatibility aliases for utils3d (MoGe version uses different names)
try:
    import utils3d.numpy as u3d_np
    import numpy as _np

    # Simple aliases
    if not hasattr(u3d_np, 'depth_edge'):
        u3d_np.depth_edge = u3d_np.depth_map_edge
    if not hasattr(u3d_np, 'normals_edge'):
        u3d_np.normals_edge = u3d_np.normal_map_edge

    # Wrapper for points_to_normals (old API returned (normals, mask))
    if not hasattr(u3d_np, 'points_to_normals'):
        def _points_to_normals(pointmap, mask=None):
            normals = u3d_np.point_map_to_normal_map(pointmap, mask=mask)
            normals_mask = ~_np.isnan(normals).any(axis=-1) if mask is None else mask
            return normals, normals_mask
        u3d_np.points_to_normals = _points_to_normals

    # Wrapper for image_uv (old API: image_uv(width=w, height=h))
    if not hasattr(u3d_np, 'image_uv'):
        def _image_uv(width=None, height=None):
            return u3d_np.uv_map(height, width)
        u3d_np.image_uv = _image_uv

    # Wrapper for image_mesh (old API: image_mesh(xyz, rgb, uv, mask=, tri=))
    if not hasattr(u3d_np, 'image_mesh'):
        def _image_mesh(xyz, rgb, uv, mask=None, tri=True):
            return u3d_np.build_mesh_from_map(xyz, rgb, uv, mask=mask, tri=tri)
        u3d_np.image_mesh = _image_mesh

except ImportError:
    pass  # Will be caught later when SAM-3D is imported

import hashlib
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Cache configuration
CACHE_DIR = Path(__file__).parent / "cache_3d"
CACHE_MAX_AGE = None  # None = keep forever (no automatic cleanup)

# SAM-3D model (lazy loaded)
_model = None
_model_available = None


def check_sam3d_available() -> tuple[bool, str]:
    """Check if SAM-3D-Objects is properly installed and configured."""
    global _model_available

    if _model_available is not None:
        return _model_available

    sam3d_path = Path(__file__).parent / "sam3d"

    # Check repository exists
    if not sam3d_path.exists():
        return False, "SAM-3D repository not found at ./sam3d"

    # Check checkpoints exist
    config_path = sam3d_path / "checkpoints" / "hf" / "pipeline.yaml"
    if not config_path.exists():
        return False, f"Checkpoints not found. Download from HuggingFace to {config_path.parent}"

    # Set environment variables needed by SAM-3D
    # (normally set by conda environment)
    if "CONDA_PREFIX" not in os.environ:
        # Try to find CUDA installation
        cuda_paths = ["/usr/local/cuda", "/usr/cuda", "/opt/cuda"]
        for cuda_path in cuda_paths:
            if Path(cuda_path).exists():
                os.environ["CUDA_HOME"] = cuda_path
                os.environ["CONDA_PREFIX"] = cuda_path  # Fake for inference.py
                break
        else:
            # Fallback - let PyTorch handle it
            os.environ["CONDA_PREFIX"] = "/usr"
            os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

    # Try to import the inference module
    try:
        import sys
        # Add sam3d root for sam3d_objects package
        sys.path.insert(0, str(sam3d_path))
        # Add notebook folder for inference module
        sys.path.insert(0, str(sam3d_path / "notebook"))
        from inference import Inference
        _model_available = (True, "SAM-3D-Objects ready")
        return _model_available
    except ImportError as e:
        return False, f"Failed to import SAM-3D: {str(e)}. Run environment setup first."
    except Exception as e:
        return False, f"SAM-3D initialization error: {str(e)}"


def get_model():
    """Get or initialize the SAM-3D model."""
    global _model

    if _model is not None:
        return _model

    available, msg = check_sam3d_available()
    if not available:
        raise RuntimeError(msg)

    import sys
    sam3d_path = Path(__file__).parent / "sam3d"
    # Add sam3d root for sam3d_objects package
    if str(sam3d_path) not in sys.path:
        sys.path.insert(0, str(sam3d_path))
    # Add notebook folder for inference module
    if str(sam3d_path / "notebook") not in sys.path:
        sys.path.insert(0, str(sam3d_path / "notebook"))

    from inference import Inference

    config_path = sam3d_path / "checkpoints" / "hf" / "pipeline.yaml"
    _model = Inference(str(config_path), compile=False)

    return _model


def generate_3d_from_detection(
    image_path: str,
    bbox: list[float],
    detection_id: str
) -> Path:
    """
    Generate 3D reconstruction from a detection.

    Args:
        image_path: Path to source image
        bbox: [x1, y1, x2, y2] bounding box coordinates
        detection_id: Unique ID for cache key

    Returns:
        Path to generated PLY file

    Raises:
        RuntimeError: If SAM-3D is not properly configured
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Generate cache key
    cache_key = hashlib.md5(f"{image_path}_{bbox}_{detection_id}".encode()).hexdigest()
    ply_path = CACHE_DIR / f"{cache_key}.ply"

    # Return cached if exists
    if ply_path.exists():
        return ply_path

    # Load image and crop detection
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = [int(c) for c in bbox]

    # Add 10% padding
    w, h = x2 - x1, y2 - y1
    pad = int(max(w, h) * 0.1)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)

    crop = image.crop((x1, y1, x2, y2))

    # Ensure minimum size (SAM-3D needs reasonable input)
    min_size = 64
    if crop.width < min_size or crop.height < min_size:
        # Resize maintaining aspect ratio
        scale = max(min_size / crop.width, min_size / crop.height)
        new_w = int(crop.width * scale)
        new_h = int(crop.height * scale)
        crop = crop.resize((new_w, new_h), Image.LANCZOS)

    # Create full mask (entire crop is the object)
    # SAM-3D expects binary mask (0/1), it will multiply by 255 internally
    mask = np.ones((crop.height, crop.width), dtype=np.uint8)

    # Convert crop to numpy array
    crop_np = np.array(crop)

    print(f"[3D] Generating for crop size: {crop_np.shape}, mask shape: {mask.shape}")

    # Run inference
    model = get_model()
    output = model(crop_np, mask, seed=42)

    # Save PLY
    output["gs"].save_ply(str(ply_path))

    return ply_path


def cleanup_cache():
    """Remove PLY files older than CACHE_MAX_AGE. If CACHE_MAX_AGE is None, keep all files."""
    if not CACHE_DIR.exists():
        return 0

    if CACHE_MAX_AGE is None:
        return 0  # Keep all files

    now = time.time()
    removed = 0
    for ply_file in CACHE_DIR.glob("*.ply"):
        if now - ply_file.stat().st_mtime > CACHE_MAX_AGE:
            try:
                ply_file.unlink()
                removed += 1
            except Exception:
                pass

    return removed


def get_cache_status() -> dict:
    """Get cache statistics."""
    if not CACHE_DIR.exists():
        return {"exists": False, "files": 0, "size_mb": 0}

    files = list(CACHE_DIR.glob("*.ply"))
    total_size = sum(f.stat().st_size for f in files)

    return {
        "exists": True,
        "files": len(files),
        "size_mb": round(total_size / (1024 * 1024), 2)
    }
