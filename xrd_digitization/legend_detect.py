"""Detect legend regions so swatches and text are excluded from curve tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from xrd_digitization.types import AxisCalibrationResult


@dataclass
class LegendRegion:
    bbox: tuple[int, int, int, int]  # in crop / working-image coordinates
    confidence: float
    method: str


def detect_legend_bbox(
    image_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    curve_labels: Sequence[str] | None = None,
) -> LegendRegion | None:
    """
    Heuristic legend box in the upper-right of the plot interior.

    Only returns a region when colored swatches are present. Curve labels alone
    (common on gray/black XRD) must not mask the plot — those sit on the data.
    """
    height, width = image_bgr.shape[:2]
    left = int(calibration.plot_left)
    right = int(calibration.plot_right)
    top = int(calibration.plot_top)
    bottom = int(calibration.plot_bottom)
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)

    # Search window: top-right quadrant of the plot.
    x0 = left + int(0.58 * plot_w)
    x1 = right - max(2, int(0.01 * plot_w))
    y0 = top + max(2, int(0.02 * plot_h))
    y1 = top + int(0.42 * plot_h)
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    colored = (sat > 45) & (gray < 245)
    dark_text = (gray < 100) & (sat < 40)

    # Row occupancy of colored swatches — legends are short horizontal color bars.
    row_col = colored.sum(axis=1).astype(float)
    row_txt = dark_text.sum(axis=1).astype(float)
    # Require real color swatches; ignore text-only / gray-black label blocks.
    if row_col.max() < max(4.0, 0.04 * (x1 - x0)):
        return None

    active = (row_col > max(3.0, 0.03 * (x1 - x0))) | (
        (row_txt > max(4.0, 0.05 * (x1 - x0))) & (row_col > max(2.0, 0.02 * (x1 - x0)))
    )
    if not active.any():
        return None

    ys = np.flatnonzero(active)
    # Take contiguous block from first active row with limited height.
    start = int(ys[0])
    end = start
    for y in ys:
        if y <= end + 6:
            end = int(y)
        else:
            break
    end = min(roi.shape[0] - 1, end + 8)
    # Expand horizontally to cover text next to swatches.
    col_score = colored[start : end + 1, :].sum(axis=0) + dark_text[start : end + 1, :].sum(axis=0)
    xs = np.flatnonzero(col_score > 0)
    if xs.size == 0:
        lx0, lx1 = 0, roi.shape[1]
    else:
        lx0 = max(0, int(xs[0]) - 4)
        lx1 = min(roi.shape[1], int(xs[-1]) + 8)

    bbox = (x0 + lx0, y0 + start, x0 + lx1, y0 + end + 1)
    conf = 0.7 if row_col.max() > 0 else 0.45
    _ = curve_labels  # reserved for future text-anchored swatch association
    return LegendRegion(bbox=bbox, confidence=conf, method="swatch_cluster")


def apply_legend_mask(
    allow: np.ndarray,
    legend: LegendRegion | None,
    *,
    dilate_px: int = 4,
) -> np.ndarray:
    """Clear ``allow`` inside the legend bbox (including colored swatches)."""
    if legend is None:
        return allow
    out = allow.copy()
    x0, y0, x1, y1 = legend.bbox
    if dilate_px > 0:
        x0 = max(0, x0 - dilate_px)
        y0 = max(0, y0 - dilate_px)
        x1 = min(out.shape[1], x1 + dilate_px)
        y1 = min(out.shape[0], y1 + dilate_px)
    out[y0:y1, x0:x1] = False
    return out
