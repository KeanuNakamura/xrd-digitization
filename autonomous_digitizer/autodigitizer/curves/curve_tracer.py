"""Trace ordered curves from binary masks and resample."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from autodigitizer.models import PlotArea

logger = logging.getLogger(__name__)


def mask_to_column_trace(
    mask: np.ndarray,
    plot_area: PlotArea,
    gap_fill: int = 4,
) -> list[tuple[float, float]]:
    """Extract a single-valued y(x) trace via column-wise median of mask pixels.

    Handles thick lines by taking the median y in each column. Small gaps are
    linearly interpolated.
    """
    roi = mask[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right]
    if roi.size == 0:
        return []
    h, w = roi.shape
    xs: list[int] = []
    ys: list[float] = []
    for x in range(w):
        cols = np.where(roi[:, x])[0]
        if len(cols) == 0:
            continue
        xs.append(x)
        ys.append(float(np.median(cols)))

    if len(xs) < 2:
        return []

    # Fill small gaps
    filled_x: list[float] = []
    filled_y: list[float] = []
    for i in range(len(xs)):
        filled_x.append(float(xs[i]))
        filled_y.append(ys[i])
        if i + 1 < len(xs):
            gap = xs[i + 1] - xs[i]
            if 1 < gap <= gap_fill:
                for g in range(1, gap):
                    alpha = g / gap
                    filled_x.append(xs[i] + g)
                    filled_y.append(ys[i] * (1 - alpha) + ys[i + 1] * alpha)

    points = [
        (plot_area.left + fx, plot_area.top + fy)
        for fx, fy in zip(filled_x, filled_y)
    ]
    return points


def continuity_multi_trace(
    mask: np.ndarray,
    plot_area: PlotArea,
    n_curves: int,
    max_jump: float = 12.0,
) -> list[list[tuple[float, float]]]:
    """Track multiple same-color curves with greedy continuity across columns."""
    roi = mask[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right]
    h, w = roi.shape
    # Per-column candidate y positions via connected runs
    candidates: list[list[float]] = []
    for x in range(w):
        col = roi[:, x]
        ys = []
        i = 0
        while i < h:
            if not col[i]:
                i += 1
                continue
            j = i
            while j < h and col[j]:
                j += 1
            ys.append(0.5 * (i + j - 1))
            i = j
        candidates.append(ys)

    # Initialize tracks from the column with most candidates near n_curves
    start = max(range(w), key=lambda x: -abs(len(candidates[x]) - n_curves) * 1000 + len(candidates[x]))
    init = sorted(candidates[start])[:n_curves]
    if not init:
        # fallback single
        single = mask_to_column_trace(mask, plot_area)
        return [single] if single else []

    tracks: list[list[tuple[float, float] | None]] = [[None] * w for _ in range(len(init))]
    for ti, y in enumerate(init):
        tracks[ti][start] = (float(plot_area.left + start), float(plot_area.top + y))

    # Forward
    last = list(init)
    for x in range(start + 1, w):
        cands = list(candidates[x])
        used = set()
        new_last = list(last)
        order = sorted(range(len(last)), key=lambda i: last[i])
        for ti in order:
            if not cands:
                new_last[ti] = last[ti]
                continue
            # nearest unused
            dists = [(abs(c - last[ti]), ci, c) for ci, c in enumerate(cands) if ci not in used]
            if not dists:
                continue
            dists.sort()
            d, ci, c = dists[0]
            if d <= max_jump:
                used.add(ci)
                new_last[ti] = c
                tracks[ti][x] = (float(plot_area.left + x), float(plot_area.top + c))
            else:
                new_last[ti] = last[ti]
        last = new_last

    # Backward
    last = list(init)
    for x in range(start - 1, -1, -1):
        cands = list(candidates[x])
        used = set()
        new_last = list(last)
        order = sorted(range(len(last)), key=lambda i: last[i])
        for ti in order:
            if not cands:
                new_last[ti] = last[ti]
                continue
            dists = [(abs(c - last[ti]), ci, c) for ci, c in enumerate(cands) if ci not in used]
            if not dists:
                continue
            dists.sort()
            d, ci, c = dists[0]
            if d <= max_jump:
                used.add(ci)
                new_last[ti] = c
                tracks[ti][x] = (float(plot_area.left + x), float(plot_area.top + c))
            else:
                new_last[ti] = last[ti]
        last = new_last

    curves = []
    for tr in tracks:
        pts = [p for p in tr if p is not None]
        if len(pts) >= 5:
            curves.append(pts)
    return curves


def skeleton_trace(mask: np.ndarray, plot_area: PlotArea) -> list[tuple[float, float]]:
    """Skeletonize and order points left-to-right as a fallback."""
    roi = mask[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right]
    if roi.sum() < 10:
        return []
    sk = skeletonize(roi.astype(bool))
    ys, xs = np.where(sk)
    if len(xs) == 0:
        return mask_to_column_trace(mask, plot_area)
    order = np.argsort(xs)
    points = [
        (float(plot_area.left + xs[i]), float(plot_area.top + ys[i]))
        for i in order
    ]
    return points


def resample_points(
    points: list[tuple[float, float]],
    n_samples: int,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    pts = np.asarray(points, dtype=float)
    # Sort by x and average duplicate x
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    # Unique x via averaging
    uniq_x, inv = np.unique(np.round(pts[:, 0], 3), return_inverse=True)
    uniq_y = np.zeros_like(uniq_x)
    counts = np.zeros_like(uniq_x)
    for i, y in enumerate(pts[:, 1]):
        uniq_y[inv[i]] += y
        counts[inv[i]] += 1
    uniq_y /= np.maximum(counts, 1)
    if len(uniq_x) < 2:
        return points
    x_new = np.linspace(uniq_x.min(), uniq_x.max(), n_samples)
    y_new = np.interp(x_new, uniq_x, uniq_y)
    return list(zip(x_new.tolist(), y_new.tolist()))


def calibrate_points(
    points: list[tuple[float, float]],
    x_cal,
    y_cal,
) -> list[tuple[float, float]]:
    out = []
    for x, y in points:
        try:
            out.append((x_cal.pixel_to_data(x), y_cal.pixel_to_data(y)))
        except Exception:
            continue
    return out
