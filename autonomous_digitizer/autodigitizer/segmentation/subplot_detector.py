"""Stage 3: subplot segmentation with CV refinement of LLM boxes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from autodigitizer.models import FigureAnalysis, SubplotSummary
from autodigitizer.processing.preprocessing import save_image

logger = logging.getLogger(__name__)


@dataclass
class SubplotCrop:
    id: str
    bbox_normalized: list[float]
    bbox_pixels: tuple[int, int, int, int]
    image: np.ndarray
    digitizable: bool = True


def _projection_gutters(gray: np.ndarray, axis: int, min_width_frac: float = 0.01) -> list[tuple[int, int]]:
    """Find low-ink gutters along rows (axis=0) or cols (axis=1)."""
    ink = (gray < 240).astype(np.float32)
    profile = ink.mean(axis=1 - axis)
    thr = max(0.02, float(np.percentile(profile, 20)))
    low = profile <= thr
    gutters: list[tuple[int, int]] = []
    n = len(profile)
    min_w = max(2, int(min_width_frac * n))
    i = 0
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        if j - i >= min_w:
            gutters.append((i, j))
        i = j
    return gutters


def _refine_bbox_with_ink(
    image: np.ndarray,
    bbox: list[float],
    max_expand: float = 0.02,
    max_shrink: float = 0.05,
) -> list[float]:
    """Tighten/expand bbox slightly using ink density, conservatively."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    # Expand a bit first to avoid cutting labels
    x0 = max(0.0, x0 - max_expand)
    y0 = max(0.0, y0 - max_expand)
    x1 = min(1.0, x1 + max_expand)
    y1 = min(1.0, y1 + max_expand)

    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = int(x1 * w), int(y1 * h)
    crop = image[py0:py1, px0:px1]
    if crop.size == 0:
        return bbox
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ink = gray < 245
    ys, xs = np.where(ink)
    if len(xs) < 50:
        return [x0, y0, x1, y1]
    # Soft shrink toward ink with margin
    margin_x = int(0.03 * (px1 - px0))
    margin_y = int(0.03 * (py1 - py0))
    nx0 = max(px0, px0 + max(0, xs.min() - margin_x))
    ny0 = max(py0, py0 + max(0, ys.min() - margin_y))
    nx1 = min(px1, px0 + min(crop.shape[1], xs.max() + margin_x + 1))
    ny1 = min(py1, py0 + min(crop.shape[0], ys.max() + margin_y + 1))

    # Limit shrink
    max_dx = max_shrink * w
    max_dy = max_shrink * h
    if nx0 - px0 > max_dx:
        nx0 = px0 + int(max_dx)
    if ny0 - py0 > max_dy:
        ny0 = py0 + int(max_dy)
    if px1 - nx1 > max_dx:
        nx1 = px1 - int(max_dx)
    if py1 - ny1 > max_dy:
        ny1 = py1 - int(max_dy)

    return [nx0 / w, ny0 / h, nx1 / w, ny1 / h]


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-9, area_a + area_b - inter)


def _cv_panel_candidates(image: np.ndarray) -> list[list[float]]:
    """Whitespace / projection based panel proposals as a cross-check."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    row_g = _projection_gutters(gray, axis=0, min_width_frac=0.015)
    col_g = _projection_gutters(gray, axis=1, min_width_frac=0.015)

    # Build bands between gutters
    def bands(gutters: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
        cuts = [0] + [g[0] + (g[1] - g[0]) // 2 for g in gutters] + [n]
        cuts = sorted(set(cuts))
        out = []
        for i in range(len(cuts) - 1):
            a, b = cuts[i], cuts[i + 1]
            if b - a > 0.12 * n:
                out.append((a, b))
        return out or [(0, n)]

    row_bands = bands(row_g, h)
    col_bands = bands(col_g, w)
    boxes = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            boxes.append([c0 / w, r0 / h, c1 / w, r1 / h])
    return boxes


def detect_subplots(
    image: np.ndarray,
    analysis: FigureAnalysis,
) -> list[SubplotCrop]:
    """Combine OpenAI bboxes with CV refinement."""
    h, w = image.shape[:2]
    cv_boxes = _cv_panel_candidates(image) if len(analysis.subplots) > 1 else []
    crops: list[SubplotCrop] = []

    for sp in analysis.subplots:
        bbox = list(sp.bbox_normalized)
        # If a CV box overlaps strongly, blend slightly toward it
        if cv_boxes:
            best = max(cv_boxes, key=lambda b: _iou(bbox, b))
            if _iou(bbox, best) > 0.35:
                bbox = [
                    0.7 * bbox[i] + 0.3 * best[i] for i in range(4)
                ]
        bbox = _refine_bbox_with_ink(image, bbox)
        x0, y0, x1, y1 = [
            int(round(bbox[0] * w)),
            int(round(bbox[1] * h)),
            int(round(bbox[2] * w)),
            int(round(bbox[3] * h)),
        ]
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0 + 1, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0 + 1, y1))
        crop_img = image[y0:y1, x0:x1].copy()
        crops.append(
            SubplotCrop(
                id=sp.id,
                bbox_normalized=[x0 / w, y0 / h, x1 / w, y1 / h],
                bbox_pixels=(x0, y0, x1, y1),
                image=crop_img,
                digitizable=sp.digitizable,
            )
        )
        logger.info("Subplot %s crop pixels=%s", sp.id, (x0, y0, x1, y1))
    return crops


def render_detection_overlay(image: np.ndarray, crops: list[SubplotCrop], path) -> None:
    vis = image.copy()
    for c in crops:
        x0, y0, x1, y1 = c.bbox_pixels
        color = (0, 180, 0) if c.digitizable else (200, 0, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            vis,
            c.id,
            (x0 + 6, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    save_image(path, vis)


def split_equal_grid(image: np.ndarray, rows: int, cols: int) -> list[SubplotCrop]:
    """Utility for synthetic tests: equal grid split."""
    h, w = image.shape[:2]
    crops = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            sid = chr(ord("A") + idx)
            crops.append(
                SubplotCrop(
                    id=sid,
                    bbox_normalized=[x0 / w, y0 / h, x1 / w, y1 / h],
                    bbox_pixels=(x0, y0, x1, y1),
                    image=image[y0:y1, x0:x1].copy(),
                )
            )
            idx += 1
    return crops
