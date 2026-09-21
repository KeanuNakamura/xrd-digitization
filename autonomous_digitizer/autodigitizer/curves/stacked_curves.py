"""Stacked / vertically-offset curve helpers."""

from __future__ import annotations

import logging

import numpy as np

from autodigitizer.models import ExtractedCurve, PlotArea
from autodigitizer.curves.curve_tracer import continuity_multi_trace, mask_to_column_trace

logger = logging.getLogger(__name__)


def detect_vertical_offsets(curves_pixels: list[list[tuple[float, float]]]) -> bool:
    """Heuristic: multiple curves with systematically different mean y."""
    if len(curves_pixels) < 2:
        return False
    means = []
    for pts in curves_pixels:
        if not pts:
            continue
        means.append(float(np.mean([p[1] for p in pts])))
    if len(means) < 2:
        return False
    means = sorted(means)
    gaps = np.diff(means)
    # Large, relatively uniform gaps suggest stacking
    if np.median(gaps) > 15 and (np.max(gaps) / max(np.min(gaps), 1e-6)) < 4:
        return True
    return False


def extract_stacked_from_mask(
    mask: np.ndarray,
    plot_area: PlotArea,
    n_curves: int,
) -> list[list[tuple[float, float]]]:
    """Trace stacked same-style curves by continuity with vertical ordering."""
    n_curves = max(1, n_curves)
    curves = continuity_multi_trace(mask, plot_area, n_curves=n_curves, max_jump=18.0)
    # Sort top-to-bottom (smaller y first in image coords = higher on plot)
    curves.sort(key=lambda pts: np.mean([p[1] for p in pts]) if pts else 0)
    return curves


def annotate_offset_metadata(curves: list[ExtractedCurve], offset_detected: bool) -> None:
    for c in curves:
        c.vertical_offset_detected = offset_detected
        c.displayed_values = True
        c.offset_corrected = False
