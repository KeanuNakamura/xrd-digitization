"""
Tight raster text detection for scientific figure images.

The active masking pipeline lives in ``raster_text_masking`` and is
geometry-first: text-like regions inside the plot are masked for digitization
even when OCR transcription is wrong or empty. Axis-band text is preserved by
spatial role, not OCR confidence.

This module keeps shared helpers (caption split, panels, geometry gates) and
re-exports the masking entrypoint for compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import cv2
import numpy as np

# Re-export the geometry-first masking API as the canonical detector.
from raster_text_masking import (  # noqa: E402
    MAX_REMOVABLE_PLOT_COVERAGE,
    RasterTextDetectionResult,
    build_box_mask_array,
    build_glyph_mask,
    detect_inner_plot_frame,
    detect_raster_text_regions,
    mask_coverage_fraction,
    render_detection_debug_image,
    validate_mask_for_combine,
)
from raster_text_masking import (  # noqa: E402
    _nms_xywh_proposals,
    _select_rejected_for_debug,
)

LOGGER = logging.getLogger(__name__)

# Back-compat aliases used by pdf_figure_structure / tests.
MAX_RASTER_MASK_COVERAGE = MAX_REMOVABLE_PLOT_COVERAGE
MAX_SHORT_WORD_WIDTH_FRAC = 0.25
MAX_HORIZONTAL_TEXT_HEIGHT_FRAC = 0.10
MAX_BOX_AREA_FRAC = 0.02
MAX_PANEL_COVERAGE_FRAC = 0.12
MAX_WIDTH_VS_EXPECTED = 3.5
MIN_OCR_CONFIDENCE = 35.0
SHORT_WORD_MAX_CHARS = 4
CHAR_WIDTH_FACTOR = 0.72
REJECT_DEBUG_MAX = 50
PROPOSAL_NMS_IOU = 0.55


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_width(box: Sequence[float]) -> float:
    return max(0.0, float(box[2] - box[0]))


def _box_height(box: Sequence[float]) -> float:
    return max(0.0, float(box[3] - box[1]))


def _clamp_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> list[float]:
    x0 = float(max(0, min(width, box[0])))
    y0 = float(max(0, min(height, box[1])))
    x1 = float(max(0, min(width, box[2])))
    y1 = float(max(0, min(height, box[3])))
    return [x0, y0, x1, y1]


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _expected_text_width(text: str, height: float) -> float:
    n = max(1, len(text.strip()))
    return n * max(1.0, height) * CHAR_WIDTH_FACTOR


def split_figure_caption_region(
    image_bgr: np.ndarray,
    *,
    pdf_text_boxes_px: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    """
    Split a full figure crop into plot-only and caption regions.

    Returns ``(plot_bbox, caption_bbox)`` in pixel coordinates relative to
    ``image_bgr``. Caption may be ``None`` when none is detected.
    """
    height, width = image_bgr.shape[:2]
    caption_y0: int | None = None
    min_caption_y = int(height * 0.96)

    if pdf_text_boxes_px:
        caption_like: list[Sequence[float]] = []
        for box in pdf_text_boxes_px:
            if len(box) < 4:
                continue
            y0 = float(box[1])
            h = float(box[3] - box[1])
            if y0 >= min_caption_y and h <= max(28.0, height * 0.04):
                caption_like.append(box)
        if caption_like:
            caption_y0 = int(max(min_caption_y, min(float(b[1]) for b in caption_like) - 4))

    if caption_y0 is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        search_start = int(height * 0.94)
        row_mean = gray.mean(axis=1)
        band = row_mean[search_start:]
        if band.size:
            white = band >= 245
            best: tuple[int, int] | None = None
            start: int | None = None
            for i, is_white in enumerate(white):
                if is_white and start is None:
                    start = i
                elif not is_white and start is not None:
                    if i - start >= max(3, height // 250):
                        best = (start, i)
                        break
                    start = None
            if best is not None:
                after = search_start + best[1]
                if after >= min_caption_y and after < height - 8:
                    if row_mean[after:].min() < 200:
                        caption_y0 = after

    if caption_y0 is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bottom = gray[min_caption_y:, :]
        if bottom.size and (bottom < 180).mean() > 0.015:
            caption_y0 = min_caption_y
        else:
            return (0, 0, width, height), None

    caption_y0 = max(min_caption_y, min(caption_y0, height - 8))
    plot_bbox = (0, 0, width, max(1, caption_y0))
    caption_bbox = (0, caption_y0, width, height)
    return plot_bbox, caption_bbox


def detect_chart_panels(
    image_bgr: np.ndarray,
    *,
    plot_bbox: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Detect chart panels inside the plot region."""
    height, width = image_bgr.shape[:2]
    if plot_bbox is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = plot_bbox
    region = image_bgr[y0:y1, x0:x1]
    if region.size == 0:
        return [(0, 0, width, height)]

    panels: list[tuple[int, int, int, int]] = []
    try:
        from xrd_digitization.detect_panels import detect_plot_panels

        for panel in detect_plot_panels(image_bgr, crop_bbox=(x0, y0, x1, y1)):
            px0, py0, px1, py1 = panel.bbox
            panels.append((int(px0), int(py0), int(px1), int(py1)))
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("detect_plot_panels unavailable: %s", exc)

    if len(panels) < 2:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        rh, rw = gray.shape
        col_mean = gray.mean(axis=0)
        search_lo = int(rw * 0.25)
        search_hi = int(rw * 0.75)
        if search_hi > search_lo + 10:
            segment = col_mean[search_lo:search_hi]
            threshold = max(240.0, float(np.percentile(segment, 90)))
            white = segment >= threshold
            gaps: list[tuple[int, int]] = []
            start: int | None = None
            for i, is_white in enumerate(white):
                if is_white and start is None:
                    start = i
                elif not is_white and start is not None:
                    if i - start >= max(6, rw // 40):
                        gaps.append((start, i))
                    start = None
            if start is not None and len(white) - start >= max(6, rw // 40):
                gaps.append((start, len(white)))
            if gaps:
                gap = max(gaps, key=lambda g: g[1] - g[0])
                split = search_lo + (gap[0] + gap[1]) // 2
                left_w = split
                right_w = rw - split
                if left_w >= rw * 0.28 and right_w >= rw * 0.28:
                    panels = [
                        (x0, y0, x0 + split, y1),
                        (x0 + split, y0, x1, y1),
                    ]

    if not panels:
        panels = [(x0, y0, x1, y1)]
    return panels


def build_graphics_exclusion_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Mask likely non-text graphical primitives (axes, borders, thick lines)."""
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    exclusion = np.zeros((height, width), dtype=np.uint8)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    h_len = max(25, width // 12)
    v_len = max(25, height // 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    exclusion = cv2.bitwise_or(
        exclusion, cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    )
    exclusion = cv2.bitwise_or(
        exclusion, cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
    )
    binary = (gray < 90).astype(np.uint8) * 255
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = float(height * width)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 40:
            continue
        aspect = w / max(h, 1)
        area_frac = area / image_area
        long_thin = (aspect >= 8 and h <= max(8, height * 0.03)) or (
            aspect <= 0.125 and w <= max(8, width * 0.03)
        )
        huge_block = area_frac > 0.03 and not (
            0.2 <= aspect <= 5.0 and h < height * 0.08
        )
        if long_thin or huge_block:
            exclusion[y : y + h, x : x + w] = 255
    exclusion = cv2.dilate(
        exclusion,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return exclusion


def validate_text_geometry(
    box: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    text: str | None = None,
    panel_box: Sequence[float] | None = None,
) -> str | None:
    """Return a reject reason if a box is implausibly large for text."""
    x0, y0, x1, y1 = box
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    area = w * h
    img_area = float(max(1, image_width * image_height))
    if area / img_area > MAX_BOX_AREA_FRAC:
        return "area_too_large"
    if h / image_height > MAX_HORIZONTAL_TEXT_HEIGHT_FRAC and w > h * 1.5:
        return "height_too_large"
    if text and len(text) <= SHORT_WORD_MAX_CHARS:
        if w / image_width > MAX_SHORT_WORD_WIDTH_FRAC:
            return "short_word_too_wide"
        expected = _expected_text_width(text, h)
        if expected > 0 and w > expected * MAX_WIDTH_VS_EXPECTED:
            return "wider_than_expected_text"
    if panel_box is not None:
        pw = max(1.0, float(panel_box[2] - panel_box[0]))
        ph = max(1.0, float(panel_box[3] - panel_box[1]))
        if area / (pw * ph) > MAX_PANEL_COVERAGE_FRAC:
            return "panel_coverage_too_high"
    return None


def pdf_span_boxes_to_pixels(
    text_spans: Sequence[dict[str, Any]],
    figure_rect: Sequence[float] | Any,
    *,
    dpi: int,
) -> list[list[float]]:
    """Convert PDF text-span bboxes into pixel coordinates for the figure crop."""
    zoom = dpi / 72.0
    if hasattr(figure_rect, "x0"):
        fx0, fy0 = float(figure_rect.x0), float(figure_rect.y0)
    else:
        fx0, fy0 = float(figure_rect[0]), float(figure_rect[1])
    boxes: list[list[float]] = []
    for span in text_spans:
        bbox = span.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        boxes.append(
            [
                (x0 - fx0) * zoom,
                (y0 - fy0) * zoom,
                (x1 - fx0) * zoom,
                (y1 - fy0) * zoom,
            ]
        )
    return boxes


__all__ = [
    "MAX_RASTER_MASK_COVERAGE",
    "RasterTextDetectionResult",
    "build_box_mask_array",
    "build_glyph_mask",
    "build_graphics_exclusion_mask",
    "detect_chart_panels",
    "detect_inner_plot_frame",
    "detect_raster_text_regions",
    "mask_coverage_fraction",
    "pdf_span_boxes_to_pixels",
    "render_detection_debug_image",
    "split_figure_caption_region",
    "validate_mask_for_combine",
    "validate_text_geometry",
    "_nms_xywh_proposals",
    "_select_rejected_for_debug",
]
