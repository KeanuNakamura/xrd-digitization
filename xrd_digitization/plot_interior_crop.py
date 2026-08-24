"""Detect an inset crop of the plotting interior for text-removal APIs.

The goal is to exclude axis spines, tick marks, and external tick/title labels
from the crop sent to ClipDrop (or similar), while still covering in-plot
annotations. Detection is independent from the ClipDrop client so it can be
improved without touching API code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xrd_digitization.crop_plot_area import (
    MIN_PLOT_HEIGHT,
    MIN_PLOT_WIDTH,
    _detect_axis_lines,
    _fallback_plot_bbox,
)

# Default inset keeps axis strokes / inward ticks out of the API crop.
DEFAULT_INSET_PX = 4
DEFAULT_INSET_FRAC = 0.004


@dataclass(frozen=True)
class PlotInteriorCrop:
    """Inset crop of the plotting interior in full-image coordinates."""

    cropped_bgr: Any
    bbox: tuple[int, int, int, int]
    frame_bbox: tuple[int, int, int, int]
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    method: str = "axis_lines_inset"


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    x1 = max(x0 + 1, min(int(x1), width))
    y1 = max(y0 + 1, min(int(y1), height))
    return x0, y0, x1, y1


def _resolve_inset_px(
    frame_bbox: tuple[int, int, int, int],
    *,
    inset_px: int | None,
    inset_frac: float,
) -> int:
    x0, y0, x1, y1 = frame_bbox
    frame_w = max(1, x1 - x0)
    frame_h = max(1, y1 - y0)
    if inset_px is not None:
        return max(0, int(inset_px))
    return max(DEFAULT_INSET_PX, int(round(min(frame_w, frame_h) * inset_frac)))


def inset_bbox(
    frame_bbox: tuple[int, int, int, int],
    *,
    inset_px: int | None = None,
    inset_frac: float = DEFAULT_INSET_FRAC,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Shrink ``frame_bbox`` inward, clamping to the image."""
    x0, y0, x1, y1 = frame_bbox
    inset = _resolve_inset_px(frame_bbox, inset_px=inset_px, inset_frac=inset_frac)
    cropped = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    return _clamp_bbox(cropped, image_width, image_height)


def detect_axes_frame_bbox(
    image_bgr: np.ndarray,
) -> tuple[tuple[int, int, int, int], float, list[str], str]:
    """
    Detect the rectangular plotting frame (axis spines) in full-image coords.

    Does not apply outward padding or curve-extension — those would pull tick
    labels / legends into a ClipDrop crop.
    """
    warnings: list[str] = []
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    left, right, top, bottom = _detect_axis_lines(gray)

    if None in (left, right, top, bottom):
        warnings.append("axis_lines_not_fully_detected")
        frame = _fallback_plot_bbox(gray)
        confidence = 0.45
        method = "fallback_margins"
    else:
        assert left is not None and right is not None and top is not None and bottom is not None
        frame = (int(left), int(top), int(right), int(bottom))
        confidence = 0.85
        method = "axis_lines"

    frame = _clamp_bbox(frame, width, height)
    if frame[2] - frame[0] < MIN_PLOT_WIDTH or frame[3] - frame[1] < MIN_PLOT_HEIGHT:
        warnings.append("axes_frame_too_small_using_fallback")
        frame = _clamp_bbox(_fallback_plot_bbox(gray), width, height)
        confidence = min(confidence, 0.4)
        method = "fallback_margins"

    return frame, confidence, warnings, method


def resolve_plot_interior_bbox(
    image_bgr: np.ndarray,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    inset_px: int | None = None,
    inset_frac: float = DEFAULT_INSET_FRAC,
    apply_inset_to_manual: bool = True,
) -> PlotInteriorCrop:
    """
    Return an inset interior crop for ClipDrop (or equivalent).

    Parameters
    ----------
    bbox:
        Optional manual full-image crop ``(x0, y0, x1, y1)``. When set, axis
        detection is skipped. By default a small inset is still applied so the
        manual box can be the outer axes rectangle.
    inset_px / inset_frac:
        Inward shrink from the detected (or manual) frame. ``inset_px=0``
        disables inset. When ``inset_px`` is None, uses
        ``max(DEFAULT_INSET_PX, frac * min(frame_w, frame_h))``.
    apply_inset_to_manual:
        If False, a provided ``bbox`` is used as-is (still clamped).
    """
    height, width = image_bgr.shape[:2]
    warnings: list[str] = []

    if bbox is not None:
        frame = _clamp_bbox(bbox, width, height)
        confidence = 1.0
        method = "manual"
        if apply_inset_to_manual:
            interior = inset_bbox(
                frame,
                inset_px=inset_px,
                inset_frac=inset_frac,
                image_width=width,
                image_height=height,
            )
            method = "manual_inset"
        else:
            interior = frame
    else:
        frame, confidence, det_warnings, method = detect_axes_frame_bbox(image_bgr)
        warnings.extend(det_warnings)
        interior = inset_bbox(
            frame,
            inset_px=inset_px,
            inset_frac=inset_frac,
            image_width=width,
            image_height=height,
        )
        method = f"{method}_inset"

    if interior[2] - interior[0] < MIN_PLOT_WIDTH or interior[3] - interior[1] < MIN_PLOT_HEIGHT:
        warnings.append("interior_crop_too_small")
        raise ValueError(
            f"Plot interior crop too small after inset: {interior} "
            f"(frame={frame}, image={(width, height)})"
        )

    x0, y0, x1, y1 = interior
    cropped = image_bgr[y0:y1, x0:x1].copy()
    return PlotInteriorCrop(
        cropped_bgr=cropped,
        bbox=interior,
        frame_bbox=frame,
        confidence=confidence,
        warnings=warnings,
        method=method,
    )


def load_image_bgr(image_path: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image_path, np.ndarray):
        return image_path.copy()
    path = Path(image_path)
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {path}")
    return image_bgr
