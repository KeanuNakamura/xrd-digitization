"""Text-region soft penalties for curve extraction.

Text masks are metadata aligned with the canonical figure rendering. They must
not hard-delete curve pixels; intersections only down-rank text-like
components while curve-like components continue through labels.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def score_curve_like_component(
    *,
    width: int,
    height: int,
    area: int,
    horizontal_coverage: int,
    plot_width: int,
    plot_height: int,
    skeleton_length: int | None = None,
    centroid_xy: tuple[float, float] | None = None,
    plot_rect: tuple[int, int, int, int] | None = None,
    text_overlap_fraction: float = 0.0,
) -> dict[str, Any]:
    """
    Score whether a connected component looks like a curve vs text.

    Uses horizontal coverage, width/height ratio, density, skeleton length,
    location inside the plot rectangle, and soft text-mask overlap.
    """
    aspect = width / max(height, 1)
    coverage_frac = horizontal_coverage / max(plot_width, 1)
    density = area / max(width * height, 1)

    inside_plot = True
    if plot_rect is not None and centroid_xy is not None:
        left, top, right, bottom = plot_rect
        cx, cy = centroid_xy
        inside_plot = left <= cx <= right and top <= cy <= bottom

    skeleton_frac = None
    if skeleton_length is not None:
        skeleton_frac = skeleton_length / max(plot_width, 1)

    curve_score = 0.0
    curve_score += min(1.0, coverage_frac / 0.35) * 0.35
    if aspect >= 3.0:
        curve_score += 0.2
    elif aspect <= 1.2 and height <= max(18, plot_height * 0.08):
        curve_score -= 0.25
    if density < 0.25:
        curve_score += 0.15
    elif density > 0.55:
        curve_score -= 0.2
    if skeleton_frac is not None:
        curve_score += min(0.2, skeleton_frac * 0.2)
    if inside_plot:
        curve_score += 0.1
    else:
        curve_score -= 0.15

    # Soft text penalty — reduces score but does not force rejection alone.
    curve_score -= 0.35 * text_overlap_fraction

    return {
        "curve_score": float(curve_score),
        "aspect_ratio": float(aspect),
        "horizontal_coverage_fraction": float(coverage_frac),
        "density": float(density),
        "skeleton_length_fraction": skeleton_frac,
        "inside_plot": inside_plot,
        "text_overlap_fraction": float(text_overlap_fraction),
        "likely_curve": curve_score >= 0.15,
        "likely_text": curve_score < 0.0 and text_overlap_fraction > 0.25,
    }


def apply_text_mask_soft_penalty(
    curve_mask: np.ndarray,
    text_mask: np.ndarray | None,
    *,
    plot_left: int | None = None,
    plot_top: int | None = None,
    plot_right: int | None = None,
    plot_bottom: int | None = None,
) -> np.ndarray:
    """
    Soft-filter a curve mask using a text-region mask.

    Curve-like components are kept even where they cross labels; text-like
    components that mostly sit in the text mask are removed.
    """
    if text_mask is None or text_mask.size == 0 or curve_mask.size == 0:
        return curve_mask

    if text_mask.shape[:2] != curve_mask.shape[:2]:
        LOGGER.warning(
            "Text mask shape %s does not match curve mask %s; skipping penalty",
            text_mask.shape,
            curve_mask.shape,
        )
        return curve_mask

    text_bin = (text_mask > 0).astype(np.uint8)
    work = curve_mask.copy()
    height, width = work.shape
    plot_width = (plot_right - plot_left) if None not in (plot_left, plot_right) else width
    plot_height = (
        (plot_bottom - plot_top) if None not in (plot_top, plot_bottom) else height
    )
    plot_rect = (
        (plot_left, plot_top, plot_right, plot_bottom)
        if None not in (plot_left, plot_top, plot_right, plot_bottom)
        else None
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        work, connectivity=8
    )
    kept = np.zeros_like(work)
    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        ys, xs = np.where(component)
        horizontal_coverage = int(np.unique(xs).size) if len(xs) else 0
        overlap = float(np.count_nonzero(component & (text_bin > 0))) / max(area, 1)

        skeleton_length = None
        try:
            component_u8 = component.astype(np.uint8) * 255
            skeleton = cv2.ximgproc.thinning(component_u8)  # type: ignore[attr-defined]
            skeleton_length = int(np.count_nonzero(skeleton))
        except Exception:
            skeleton_length = int(max(w, h) * (0.5 + 0.5 * (1.0 - overlap)))

        score = score_curve_like_component(
            width=w,
            height=h,
            area=area,
            horizontal_coverage=horizontal_coverage,
            plot_width=plot_width,
            plot_height=plot_height,
            skeleton_length=skeleton_length,
            centroid_xy=(float(centroids[label][0]), float(centroids[label][1])),
            plot_rect=plot_rect,
            text_overlap_fraction=overlap,
        )

        if score["likely_text"] and not score["likely_curve"]:
            continue

        kept[component] = 255

    return kept
