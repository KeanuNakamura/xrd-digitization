"""Detect the true plotting rectangle inside a subplot crop."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from autodigitizer.models import PlotArea
from autodigitizer.processing.preprocessing import save_image

logger = logging.getLogger(__name__)


def detect_plot_area(image: np.ndarray) -> PlotArea:
    """Find the plotting rectangle, including L-shaped axes (no top/right spines)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark = (gray < 90).astype(np.uint8)

    # Bottom spine: strongest horizontal dark line in lower half
    row_dark = dark.mean(axis=1)
    lower = row_dark.copy()
    lower[: int(0.5 * h)] = 0
    bottom = int(np.argmax(lower)) if lower.max() > 0.01 else int(0.88 * h)

    # Left spine: strongest vertical dark line in left 35%
    col_dark = dark.mean(axis=0)
    left_region = col_dark.copy()
    left_region[int(0.35 * w) :] = 0
    left = int(np.argmax(left_region)) if left_region.max() > 0.01 else int(0.1 * w)

    # Right edge: prefer extent of the bottom axis line (works for L-frames)
    axis_band = dark[max(0, bottom - 2) : min(h, bottom + 3), :]
    axis_cols = np.where(axis_band.any(axis=0))[0]
    if len(axis_cols) > 10:
        # Continuous run containing `left`
        runs = _runs(axis_cols)
        best = max(runs, key=lambda r: (r[0] <= left <= r[1], r[1] - r[0]))
        right_from_axis = int(best[1])
    else:
        right_from_axis = int(0.95 * w)

    # Also consider non-background content (curve ink), excluding near-black axes/text
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    chroma = np.sqrt((lab[:, :, 1].astype(float) - 128) ** 2 + (lab[:, :, 2].astype(float) - 128) ** 2)
    colorful = (chroma > 12) & (gray < 240)
    # Drop bottom label strip and left label strip roughly
    colorful[: max(0, int(0.02 * h)), :] = False
    colorful[min(h, bottom + 1) :, :] = False
    colorful[:, : max(0, left - 2)] = False

    ys, xs = np.where(colorful)
    if len(xs) > 30:
        right_from_curve = int(min(w - 1, xs.max() + max(4, int(0.01 * w))))
        top_from_curve = int(max(0, ys.min() - max(4, int(0.02 * h))))
    else:
        # Fallback: any non-white ink inside candidate frame
        ink = gray < 245
        ink[min(h, bottom + 1) :, :] = False
        ink[:, : max(0, left)] = False
        ys2, xs2 = np.where(ink)
        if len(xs2) > 30:
            right_from_curve = int(min(w - 1, xs2.max() + 4))
            top_from_curve = int(max(0, ys2.min() - 4))
        else:
            right_from_curve = right_from_axis
            top_from_curve = int(0.08 * h)

    # Do NOT trust a spurious "right spine" peak from a tall peak line.
    # Use the farther of axis extent and curve extent, capped by image.
    right = max(right_from_axis, right_from_curve)
    # If axis run is much shorter than curve, prefer curve (axis detection clipped)
    if right_from_curve - left > 0.7 * (right_from_axis - left) and right_from_curve > right_from_axis:
        right = right_from_curve
    # If axis extends further (padding past last peak), prefer axis
    if right_from_axis > right_from_curve:
        right = right_from_axis

    top = top_from_curve
    # If a real top spine exists near the top, snap to it
    upper = row_dark.copy()
    upper[int(0.4 * h) :] = 0
    if upper.max() > 0.04:
        cand = int(np.argmax(upper))
        if cand < top + 0.15 * h:
            top = cand

    left = int(np.clip(left, 0, w - 2))
    right = int(np.clip(right, left + 2, w - 1))
    top = int(np.clip(top, 0, h - 2))
    bottom = int(np.clip(bottom, top + 2, h - 1))

    # Guard: plot should cover a substantial fraction of width
    if right - left < 0.5 * w:
        right = max(right, int(0.92 * w))

    conf = 0.75
    if right - left < 0.35 * w or bottom - top < 0.3 * h:
        conf = 0.4

    area = PlotArea(left=left, right=right, top=top, bottom=bottom, confidence=conf)
    logger.debug("Plot area: L=%d R=%d T=%d B=%d conf=%.2f", left, right, top, bottom, conf)
    return area


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return [(0, 0)]
    runs = []
    start = int(indices[0])
    prev = start
    for x in indices[1:]:
        x = int(x)
        if x - prev > 3:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    return runs


def render_plot_area(image: np.ndarray, area: PlotArea, path) -> None:
    vis = image.copy()
    cv2.rectangle(vis, (area.left, area.top), (area.right, area.bottom), (255, 0, 0), 2)
    save_image(path, vis)
