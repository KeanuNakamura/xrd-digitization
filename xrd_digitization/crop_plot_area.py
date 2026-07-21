from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from xrd_digitization.types import PlotCropResult

MIN_PLOT_WIDTH = 40
MIN_PLOT_HEIGHT = 40


def _detect_axis_lines(gray: np.ndarray) -> tuple[int | None, int | None, int | None, int | None]:
    """Return (left, right, top, bottom) plot bounds from axis-like lines."""
    height, width = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 20)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 20), 1))

    vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)

    left_scores = vertical[:, : int(width * 0.35)].sum(axis=0)
    right_scores = vertical[:, int(width * 0.65) :].sum(axis=0)
    top_scores = horizontal[: int(height * 0.35), :].sum(axis=1)
    bottom_scores = horizontal[int(height * 0.65) :, :].sum(axis=1)

    left = int(np.argmax(left_scores)) if left_scores.size and left_scores.max() > 0 else None
    right = (
        int(width * 0.65 + np.argmax(right_scores))
        if right_scores.size and right_scores.max() > 0
        else None
    )
    top = int(np.argmax(top_scores)) if top_scores.size and top_scores.max() > 0 else None
    bottom = (
        int(height * 0.65 + np.argmax(bottom_scores))
        if bottom_scores.size and bottom_scores.max() > 0
        else None
    )

    return left, right, top, bottom


def _fallback_plot_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    height, width = gray.shape
    margin_x = int(width * 0.12)
    margin_top = int(height * 0.08)
    margin_bottom = int(height * 0.18)
    return margin_x, margin_top, width - int(width * 0.04), height - margin_bottom


def _extend_crop_bbox_to_curve(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Widen the crop when the plotted curve extends past the detected right frame."""
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    if height <= 0:
        return bbox

    gray = cv2.cvtColor(image_bgr[y0:y1, x0:], cv2.COLOR_BGR2GRAY)
    roi = gray[int(height * 0.05) : int(height * 0.88), :]
    if roi.size == 0:
        return bbox

    last_curve_col = None
    for col in range(roi.shape[1] - 1, -1, -1):
        if float(roi[:, col].min()) <= 200:
            last_curve_col = col
            break

    if last_curve_col is None:
        return bbox

    margin = max(8, int(0.01 * image_bgr.shape[1]))
    new_x1 = min(image_bgr.shape[1], x0 + last_curve_col + margin)
    if new_x1 <= x1:
        return bbox
    return x0, y0, new_x1, y1


def crop_plot_area(
    image_path: str | Path | np.ndarray,
    *,
    padding: int = 2,
) -> PlotCropResult:
    """
    Detect and crop the plot region from a figure image.

    Attempts to locate axis lines; falls back to conservative margins.
    """
    warnings: list[str] = []

    if isinstance(image_path, np.ndarray):
        image_bgr = image_path.copy()
        source_shape = image_bgr.shape[:2]
    else:
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        source_shape = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    left, right, top, bottom = _detect_axis_lines(gray)

    if None in (left, right, top, bottom):
        warnings.append("axis_lines_not_fully_detected")
        x0, y0, x1, y1 = _fallback_plot_bbox(gray)
    else:
        x0, y0, x1, y1 = left, top, right, bottom

    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image_bgr.shape[1], x1 + padding)
    y1 = min(image_bgr.shape[0], y1 + padding)

    if x1 - x0 < MIN_PLOT_WIDTH or y1 - y0 < MIN_PLOT_HEIGHT:
        warnings.append("plot_crop_too_small_using_fallback")
        x0, y0, x1, y1 = _fallback_plot_bbox(gray)

    x0, y0, x1, y1 = _extend_crop_bbox_to_curve(image_bgr, (x0, y0, x1, y1))

    cropped = image_bgr[y0:y1, x0:x1].copy()

    # Multi-panel heuristic: strong vertical whitespace inside crop.
    crop_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    col_mean = crop_gray.mean(axis=0)
    valleys = np.where(col_mean > np.percentile(col_mean, 92))[0]
    if len(valleys) > 0:
        gaps = np.split(valleys, np.where(np.diff(valleys) > 3)[0] + 1)
        wide_gaps = [g for g in gaps if len(g) > cropped.shape[1] * 0.03]
        if wide_gaps:
            warnings.append("possible_multi_panel_in_plot_area")

    confidence = 0.85 if "axis_lines_not_fully_detected" not in warnings else 0.55

    return PlotCropResult(
        cropped_bgr=cropped,
        bbox=(x0, y0, x1, y1),
        confidence=confidence,
        warnings=warnings,
    )
