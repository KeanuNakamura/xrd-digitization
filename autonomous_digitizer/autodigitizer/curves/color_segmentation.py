"""Color-based separation of curve pixels in LAB space."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class ColorCluster:
    label: int
    center_lab: np.ndarray
    center_rgb: tuple[int, int, int]
    mask: np.ndarray
    pixel_count: int


def _rgb_of_lab(lab: np.ndarray) -> tuple[int, int, int]:
    arr = np.uint8([[lab.astype(np.uint8)]])
    rgb = cv2.cvtColor(arr, cv2.COLOR_LAB2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def is_near_gray(lab: np.ndarray, chroma_thresh: float = 12.0) -> bool:
    # OpenCV LAB: L,a,b with a,b centered at 128
    a = float(lab[1]) - 128.0
    b = float(lab[2]) - 128.0
    return (a * a + b * b) ** 0.5 < chroma_thresh


def segment_by_color(
    image: np.ndarray,
    ink_mask: np.ndarray,
    n_clusters: int | None = None,
    max_clusters: int = 8,
    min_pixels: int = 80,
    exclude_near_black: bool = True,
) -> list[ColorCluster]:
    """Cluster non-excluded ink pixels by LAB color."""
    ys, xs = np.where(ink_mask)
    if len(xs) < min_pixels:
        return []

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    samples = lab[ys, xs].astype(np.float32)

    # Drop near-black (axes/text often black) if requested
    if exclude_near_black:
        keep = samples[:, 0] > 25  # L channel
        # also keep chromatic dark colors
        chroma = np.sqrt((samples[:, 1] - 128) ** 2 + (samples[:, 2] - 128) ** 2)
        keep |= chroma > 15
        samples = samples[keep]
        ys, xs = ys[keep], xs[keep]
        if len(xs) < min_pixels:
            return []

    if n_clusters is None:
        n_clusters = _estimate_k(samples, max_clusters=max_clusters)

    n_clusters = int(np.clip(n_clusters, 1, min(max_clusters, len(samples))))
    if n_clusters == 1:
        mask = np.zeros(ink_mask.shape, dtype=bool)
        mask[ys, xs] = True
        center = samples.mean(axis=0)
        return [
            ColorCluster(
                label=0,
                center_lab=center,
                center_rgb=_rgb_of_lab(center),
                mask=mask,
                pixel_count=int(mask.sum()),
            )
        ]

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(samples)
    clusters: list[ColorCluster] = []
    for k in range(n_clusters):
        sel = labels == k
        if sel.sum() < min_pixels:
            continue
        center = km.cluster_centers_[k]
        if is_near_gray(center) and center[0] < 40:
            # Likely axis residue
            continue
        mask = np.zeros(ink_mask.shape, dtype=bool)
        mask[ys[sel], xs[sel]] = True
        clusters.append(
            ColorCluster(
                label=k,
                center_lab=center,
                center_rgb=_rgb_of_lab(center),
                mask=mask,
                pixel_count=int(sel.sum()),
            )
        )
    clusters.sort(key=lambda c: c.pixel_count, reverse=True)
    logger.info("Color segmentation produced %d clusters", len(clusters))
    return clusters


def _estimate_k(samples: np.ndarray, max_clusters: int = 8) -> int:
    """Heuristic K via chroma uniqueness / elbow on small subsample."""
    n = min(len(samples), 4000)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(samples), size=n, replace=False)
    sub = samples[idx]
    # Count rough unique quantized colors
    quant = np.round(sub / 12).astype(int)
    uniq = np.unique(quant, axis=0)
    chromatic = []
    for u in uniq:
        lab = u * 12.0
        if not is_near_gray(lab, chroma_thresh=14):
            chromatic.append(u)
    k = max(1, len(chromatic))
    return int(min(max_clusters, max(1, k)))
