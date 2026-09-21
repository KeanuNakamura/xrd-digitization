"""Semantic and heuristic masks for excluding non-curve pixels."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from autodigitizer.models import PlotArea

logger = logging.getLogger(__name__)


def background_mask(image: np.ndarray, luma_threshold: int = 245) -> np.ndarray:
    """True where pixels look like paper/background."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return gray >= luma_threshold


def near_black_mask(image: np.ndarray, threshold: int = 40) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return gray <= threshold


def axis_spine_mask(image: np.ndarray, plot_area: PlotArea, thickness: int = 3) -> np.ndarray:
    """Mask likely axis spines near plot borders."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    l, r, t, b = plot_area.left, plot_area.right, plot_area.top, plot_area.bottom
    mask[max(0, t - thickness) : min(h, t + thickness + 1), l:r] = True
    mask[max(0, b - thickness) : min(h, b + thickness + 1), l:r] = True
    mask[t:b, max(0, l - thickness) : min(w, l + thickness + 1)] = True
    mask[t:b, max(0, r - thickness) : min(w, r + thickness + 1)] = True
    return mask


def grid_line_mask(image: np.ndarray, plot_area: PlotArea) -> np.ndarray:
    """Detect faint horizontal/vertical grid lines inside the plot area."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    roi = gray[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right]
    if roi.size == 0:
        return mask

    edges = cv2.Canny(roi, 40, 120)
    row_sum = edges.mean(axis=1)
    col_sum = edges.mean(axis=0)
    row_thr = max(0.15, float(np.percentile(row_sum, 92)))
    col_thr = max(0.15, float(np.percentile(col_sum, 92)))
    for i, v in enumerate(row_sum):
        if v >= row_thr:
            yy = plot_area.top + i
            mask[yy, plot_area.left : plot_area.right] = True
    for j, v in enumerate(col_sum):
        if v >= col_thr:
            xx = plot_area.left + j
            mask[plot_area.top : plot_area.bottom, xx] = True
    return mask


def vertical_guide_mask(
    plot_area: PlotArea,
    xs_normalized: list[float],
    shape: tuple[int, ...],
    half_width_px: int = 2,
) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for xn in xs_normalized or []:
        x = int(round(plot_area.left + float(xn) * plot_area.width))
        x0 = max(plot_area.left, x - half_width_px)
        x1 = min(plot_area.right, x + half_width_px + 1)
        mask[plot_area.top : plot_area.bottom, x0:x1] = True
    return mask


def normalized_regions_to_mask(
    image_shape: tuple[int, ...],
    regions: list[list[float]],
    relative_to_plot: PlotArea | None = None,
    *,
    shrink: float = 0.0,
) -> np.ndarray:
    """Convert normalized [x0,y0,x1,y1] regions to a boolean mask.

    If relative_to_plot is set, coordinates are relative to that plot rectangle.
    Otherwise they are relative to the full image/crop.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for reg in regions or []:
        if len(reg) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in reg]
        if shrink:
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            x0 = cx + (x0 - cx) * (1 - shrink)
            x1 = cx + (x1 - cx) * (1 - shrink)
            y0 = cy + (y0 - cy) * (1 - shrink)
            y1 = cy + (y1 - cy) * (1 - shrink)
        if relative_to_plot is not None:
            px0 = relative_to_plot.left + x0 * relative_to_plot.width
            px1 = relative_to_plot.left + x1 * relative_to_plot.width
            py0 = relative_to_plot.top + y0 * relative_to_plot.height
            py1 = relative_to_plot.top + y1 * relative_to_plot.height
        else:
            px0, px1 = x0 * w, x1 * w
            py0, py1 = y0 * h, y1 * h
        ix0, iy0 = int(max(0, px0)), int(max(0, py0))
        ix1, iy1 = int(min(w, px1)), int(min(h, py1))
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = True
    return mask


def textlike_component_mask(image: np.ndarray, plot_area: PlotArea) -> np.ndarray:
    """Heuristic: remove small/tall connected components that look like glyphs, not curves."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    dark = ((gray < 100) & ~background_mask(image)).astype(np.uint8)
    roi = dark[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right]
    if roi.sum() < 20:
        return np.zeros((h, w), dtype=bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    text = np.zeros_like(roi, dtype=bool)
    plot_w = max(1, roi.shape[1])
    plot_h = max(1, roi.shape[0])
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 8:
            text[labels == i] = True
            continue
        # Tall thin blobs (rotated Miller indices) or compact letter-sized blobs
        aspect = bh / max(1, bw)
        width_frac = bw / plot_w
        height_frac = bh / plot_h
        if width_frac < 0.04 and aspect > 1.3 and height_frac < 0.25:
            text[labels == i] = True
        elif width_frac < 0.08 and height_frac < 0.08 and area < 0.002 * roi.size:
            text[labels == i] = True
    out = np.zeros((h, w), dtype=bool)
    out[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right] = text
    return out


def build_exclusion_mask(
    image: np.ndarray,
    plot_area: PlotArea,
    annotation_regions: list[list[float]] | None = None,
    legend_bbox: list[float] | None = None,
    *,
    text_regions_plot: list[list[float]] | None = None,
    vertical_guides_x: list[float] | None = None,
    include_grid: bool = False,
    include_textlike_cc: bool = True,
    annotation_is_plot_normalized: bool = False,
) -> np.ndarray:
    """Boolean mask of pixels that should not be traced as curves.

    Prefer tight `text_regions_plot` (plot-normalized) over coarse annotation boxes.
    Coarse annotation regions are shrunk and only used if no text_regions_plot given.
    """
    h, w = image.shape[:2]
    excl = np.zeros((h, w), dtype=bool)
    excl |= axis_spine_mask(image, plot_area)
    if include_grid:
        excl |= grid_line_mask(image, plot_area)
    if vertical_guides_x:
        excl |= vertical_guide_mask(plot_area, vertical_guides_x, image.shape)

    if text_regions_plot:
        # Labels sit above peaks: only mask the upper portion of each box so tips survive
        trimmed = []
        for reg in text_regions_plot:
            if len(reg) != 4:
                continue
            x0, y0, x1, y1 = reg
            y_cut = y0 + 0.65 * (y1 - y0)  # keep bottom ~35% of box (near peak)
            trimmed.append([x0, y0, x1, y_cut])
        excl |= normalized_regions_to_mask(
            image.shape, trimmed, relative_to_plot=plot_area, shrink=0.15
        )
    elif annotation_regions:
        # Conservative: shrink coarse boxes so peak tips survive
        excl |= normalized_regions_to_mask(
            image.shape,
            annotation_regions,
            relative_to_plot=plot_area if annotation_is_plot_normalized else None,
            shrink=0.35,
        )

    if legend_bbox:
        # Legend usually crop-normalized; shrink slightly
        excl |= normalized_regions_to_mask(image.shape, [legend_bbox], shrink=0.05)

    if include_textlike_cc:
        excl |= textlike_component_mask(image, plot_area)

    outside = np.ones((h, w), dtype=bool)
    outside[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right] = False
    excl |= outside
    return excl


def ink_mask(image: np.ndarray, excl: np.ndarray, luma_threshold: int = 245) -> np.ndarray:
    """Candidate curve/ink pixels inside plot, excluding masked regions."""
    bg = background_mask(image, luma_threshold)
    ink = ~bg & ~excl
    return ink
