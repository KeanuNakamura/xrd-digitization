"""Conservative image loading and preprocessing."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_image(path: Path | str) -> np.ndarray:
    """Load image as RGB uint8 array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix}")

    # Prefer Pillow for broad format support, then convert to RGB numpy
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im)
    logger.info("Loaded image %s shape=%s", path, arr.shape)
    return arr


def save_image(path: Path | str, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image).save(path)
    else:
        Image.fromarray(image.astype(np.uint8)).save(path)


def image_hash(image: np.ndarray) -> str:
    """Stable content hash for cache keys."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(image).tobytes())
    h.update(str(image.shape).encode())
    return h.hexdigest()[:24]


def file_hash(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def conservative_denoise(image: np.ndarray) -> np.ndarray:
    """Very light denoise; preserve axes and thin lines."""
    return cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=25)


def crop_normalized(image: np.ndarray, bbox: list[float], pad: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    if pad:
        dx = (x1 - x0) * pad
        dy = (y1 - y0) * pad
        x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy
    x0 = int(max(0, min(w - 1, round(x0 * w))))
    y0 = int(max(0, min(h - 1, round(y0 * h))))
    x1 = int(max(x0 + 1, min(w, round(x1 * w))))
    y1 = int(max(y0 + 1, min(h, round(y1 * h))))
    return image[y0:y1, x0:x1].copy()


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image
