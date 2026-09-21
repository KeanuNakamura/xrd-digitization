from __future__ import annotations

import logging
import re
from typing import Sequence

import cv2
import numpy as np

from xrd_digitization.types import PlotPanel

LOGGER = logging.getLogger(__name__)

_PANEL_LABEL_RE = re.compile(r"^[\(\[]?([A-Za-z]|[0-9]{1,2})[\)\].:]?$")


def _curve_pixel_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    colored = (sat > 25) & (gray < 245)
    dark = gray < 90
    return colored | dark


def _horizontal_whitespace_runs(
    gray: np.ndarray,
    *,
    min_width_fraction: float = 0.55,
    white_threshold: float = 235.0,
) -> list[tuple[int, int]]:
    """Return (start_row, end_row) for near-white horizontal bands."""
    height, width = gray.shape
    row_mean = gray.mean(axis=1)
    white = row_mean >= white_threshold
    min_width = int(width * min_width_fraction)

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for row, is_white in enumerate(white):
        if is_white and start is None:
            start = row
        elif not is_white and start is not None:
            if row - start >= 3:
                band = gray[start:row, :]
                if band.shape[1] >= min_width and band.mean() >= white_threshold - 5:
                    runs.append((start, row))
            start = None
    if start is not None and height - start >= 3:
        runs.append((start, height))
    return runs


def _vertical_whitespace_runs(
    gray: np.ndarray,
    *,
    min_height_fraction: float = 0.55,
    white_threshold: float = 235.0,
) -> list[tuple[int, int]]:
    """Return (start_col, end_col) for near-white vertical bands."""
    height, width = gray.shape
    col_mean = gray.mean(axis=0)
    white = col_mean >= white_threshold
    min_height = int(height * min_height_fraction)

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for col, is_white in enumerate(white):
        if is_white and start is None:
            start = col
        elif not is_white and start is not None:
            if col - start >= 3:
                band = gray[:, start:col]
                if band.shape[0] >= min_height and band.mean() >= white_threshold - 5:
                    runs.append((start, col))
            start = None
    if start is not None and width - start >= 3:
        runs.append((start, width))
    return runs


def _panel_separator_score_horizontal(
    gray: np.ndarray,
    curve_mask: np.ndarray,
    gap_start: int,
    gap_end: int,
) -> float:
    """Score how likely a horizontal band is a subplot separator."""
    height, width = gray.shape
    gap_len = gap_end - gap_start
    if gap_len < max(8, int(height * 0.03)):
        return 0.0

    gap_curve = curve_mask[gap_start:gap_end, :].mean()
    if gap_curve > 0.05:
        return 0.0

    gap_center = (gap_start + gap_end) / 2.0
    if gap_center < height * 0.18 or gap_center > height * 0.82:
        return 0.0

    above = curve_mask[:gap_start, :]
    below = curve_mask[gap_end:, :]
    if above.size == 0 or below.size == 0:
        return 0.0

    above_density = above.mean()
    below_density = below.mean()
    if above_density < 0.002 or below_density < 0.002:
        return 0.0

    above_h = gap_start
    below_h = height - gap_end
    size_ratio = min(above_h, below_h) / max(above_h, below_h)
    if size_ratio < 0.18:
        return 0.0

    below_gray = gray[gap_end:, :]
    axis_window = below_gray[: max(20, int(below_h * 0.2)), :]
    dark_rows = (axis_window < 120).sum(axis=1)
    has_frame_below = bool(len(dark_rows) and dark_rows.max() >= width * 0.25)

    gap_score = min(1.0, gap_len / max(12.0, height * 0.06))
    cleanliness = 1.0 - min(1.0, gap_curve * 20.0)
    centrality = 1.0 - abs(gap_center / height - 0.5) * 1.2
    frame_bonus = 1.25 if has_frame_below else 0.55
    return gap_score * cleanliness * centrality * size_ratio * frame_bonus


def _panel_separator_score_vertical(
    gray: np.ndarray,
    curve_mask: np.ndarray,
    gap_start: int,
    gap_end: int,
) -> float:
    """Score how likely a vertical band is a side-by-side subplot separator."""
    height, width = gray.shape
    gap_len = gap_end - gap_start
    if gap_len < max(6, int(width * 0.01)):
        return 0.0

    gap_curve = curve_mask[:, gap_start:gap_end].mean()
    if gap_curve > 0.05:
        return 0.0

    gap_center = (gap_start + gap_end) / 2.0
    if gap_center < width * 0.18 or gap_center > width * 0.82:
        return 0.0

    left = curve_mask[:, :gap_start]
    right = curve_mask[:, gap_end:]
    if left.size == 0 or right.size == 0:
        return 0.0

    left_density = left.mean()
    right_density = right.mean()
    if left_density < 0.002 or right_density < 0.002:
        return 0.0

    left_w = gap_start
    right_w = width - gap_end
    size_ratio = min(left_w, right_w) / max(left_w, right_w)
    if size_ratio < 0.25:
        return 0.0

    # Prefer a dark vertical frame edge near either side of the gap.
    left_edge = gray[:, max(0, gap_start - 8) : gap_start]
    right_edge = gray[:, gap_end : min(width, gap_end + 8)]
    has_frame = False
    for edge in (left_edge, right_edge):
        if edge.size == 0:
            continue
        dark_cols = (edge < 120).sum(axis=0)
        if len(dark_cols) and dark_cols.max() >= height * 0.25:
            has_frame = True
            break

    gap_score = min(1.0, gap_len / max(8.0, width * 0.03))
    cleanliness = 1.0 - min(1.0, gap_curve * 20.0)
    centrality = 1.0 - abs(gap_center / width - 0.5) * 1.2
    frame_bonus = 1.3 if has_frame else 0.7
    return gap_score * cleanliness * centrality * size_ratio * frame_bonus


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter)


def _nms_boxes(
    boxes: list[tuple[int, int, int, int]],
    *,
    iou_threshold: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for idx in order:
        candidate = boxes[idx]
        if any(_iou(candidate, existing) >= iou_threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def detect_axes_frame_boxes(
    image_bgr: np.ndarray,
    *,
    min_area_fraction: float = 0.08,
    max_area_fraction: float = 0.85,
) -> list[tuple[int, int, int, int]]:
    """
    Detect rectangular axes/plot frames via edge contours.

    Returns absolute (x0, y0, x1, y1) boxes sorted in reading order.
    """
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(max(1, width * height))
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h < 40:
            continue
        area_frac = (w * h) / image_area
        if area_frac < min_area_fraction or area_frac > max_area_fraction:
            continue
        aspect = w / float(h)
        if aspect < 0.35 or aspect > 4.0:
            continue
        # Prefer rectangles with a strong border (axes frame).
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) < 4:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))

    boxes = _nms_boxes(boxes)
    boxes.sort(key=lambda b: (b[1] // max(1, height // 8), b[0]))
    return boxes


def _split_by_best_gap(
    region: np.ndarray,
    *,
    offset_x: int,
    offset_y: int,
    orientation: str,
) -> list[tuple[int, int, int, int]] | None:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    curve_mask = _curve_pixel_mask(region)
    height, width = gray.shape

    if orientation == "horizontal":
        runs = _horizontal_whitespace_runs(gray)
        best: tuple[float, int, int] | None = None
        for gap_start, gap_end in runs:
            score = _panel_separator_score_horizontal(gray, curve_mask, gap_start, gap_end)
            if score <= 0.45:
                continue
            if best is None or score > best[0]:
                best = (score, gap_start, gap_end)
        if best is None:
            return None
        _, gap_start, gap_end = best
        boxes = [
            (offset_x, offset_y, offset_x + width, offset_y + gap_start),
            (offset_x, offset_y + gap_end, offset_x + width, offset_y + height),
        ]
        min_dim = max(40, int(height * 0.12))
        valid = [b for b in boxes if (b[3] - b[1]) >= min_dim]
        return valid if len(valid) >= 2 else None

    runs = _vertical_whitespace_runs(gray)
    best = None
    for gap_start, gap_end in runs:
        score = _panel_separator_score_vertical(gray, curve_mask, gap_start, gap_end)
        if score <= 0.45:
            continue
        if best is None or score > best[0]:
            best = (score, gap_start, gap_end)
    if best is None:
        return None
    _, gap_start, gap_end = best
    boxes = [
        (offset_x, offset_y, offset_x + gap_start, offset_y + height),
        (offset_x + gap_end, offset_y, offset_x + width, offset_y + height),
    ]
    min_dim = max(40, int(width * 0.12))
    valid = [b for b in boxes if (b[2] - b[0]) >= min_dim]
    return valid if len(valid) >= 2 else None


def _ocr_panel_labels(
    image_bgr: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[str | None, float]]:
    """Try to read A/B/C-style panel labels near the top-left of each box."""
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return [(None, 0.0) for _ in boxes]

    results: list[tuple[str | None, float]] = []
    for x0, y0, x1, y1 in boxes:
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        # Panel letters usually sit in the upper-left interior of the axes frame.
        roi = image_bgr[
            y0 : y0 + max(24, int(height * 0.22)),
            x0 : x0 + max(24, int(width * 0.18)),
        ]
        if roi.size == 0:
            results.append((None, 0.0))
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            data = pytesseract.image_to_data(
                binary,
                config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIabcdefghi0123456789()",
                output_type=Output.DICT,
            )
        except Exception:
            results.append((None, 0.0))
            continue

        best_label: str | None = None
        best_conf = 0.0
        n = len(data.get("text") or [])
        for i in range(n):
            text = str(data["text"][i] or "").strip()
            if not text:
                continue
            match = _PANEL_LABEL_RE.match(text)
            if not match:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 55:
                continue
            label = match.group(1).upper()
            if conf > best_conf:
                best_conf = conf
                best_label = label
        results.append((best_label, best_conf / 100.0 if best_label else 0.0))
    return results


def _assign_panel_labels(
    image_bgr: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    min_label_confidence: float = 0.7,
) -> list[tuple[str, float, str]]:
    """
    Return (label, confidence, source) for each box.

    Prefer OCR-detected letters when confidence is high and unique; otherwise
    fall back to reading-order A, B, C, ...
    """
    ocr = _ocr_panel_labels(image_bgr, boxes)
    used: set[str] = set()
    assigned: list[tuple[str | None, float, str]] = []
    for label, conf in ocr:
        if (
            label
            and conf >= min_label_confidence
            and label not in used
            and label.isalpha()
        ):
            used.add(label)
            assigned.append((label, conf, "ocr"))
        else:
            assigned.append((None, 0.0, "pending"))

    reading_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    next_idx = 0
    out: list[tuple[str, float, str]] = []
    for label, conf, source in assigned:
        if label is not None:
            out.append((label, conf, source))
            continue
        while next_idx < len(reading_order) and reading_order[next_idx] in used:
            next_idx += 1
        fallback = reading_order[next_idx] if next_idx < len(reading_order) else str(len(out) + 1)
        used.add(fallback)
        next_idx += 1
        out.append((fallback, 0.0, "reading_order"))
    return out


def _single_panel(
    *,
    width: int,
    height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[PlotPanel]:
    return [
        PlotPanel(
            index=1,
            bbox=(offset_x, offset_y, offset_x + width, offset_y + height),
            label=None,
            detection_method="single",
        )
    ]


def detect_plot_panels(
    image_bgr: np.ndarray,
    *,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> list[PlotPanel]:
    """
    Split a figure into subplot panels using axes-frame geometry first, then
    whitespace-gap heuristics (vertical stacks and side-by-side layouts).
    """
    if crop_bbox is not None:
        x0, y0, x1, y1 = crop_bbox
        region = image_bgr[y0:y1, x0:x1]
        offset_x, offset_y = x0, y0
    else:
        region = image_bgr
        offset_x, offset_y = 0, 0

    height, width = region.shape[:2]
    if height < 80 or width < 80:
        return _single_panel(width=width, height=height, offset_x=offset_x, offset_y=offset_y)

    # 1) Axes / plot-frame rectangles.
    frame_boxes = detect_axes_frame_boxes(region)
    absolute_frames = [
        (fx0 + offset_x, fy0 + offset_y, fx1 + offset_x, fy1 + offset_y)
        for fx0, fy0, fx1, fy1 in frame_boxes
    ]
    if len(absolute_frames) >= 2:
        boxes = absolute_frames
        method = "axes_frames"
    else:
        # 2) Whitespace gaps — prefer side-by-side for wide figures.
        aspect = width / float(height)
        orientations = ("vertical", "horizontal") if aspect >= 1.15 else ("horizontal", "vertical")
        boxes = None
        method = "whitespace_gap"
        for orientation in orientations:
            split = _split_by_best_gap(
                region,
                offset_x=offset_x,
                offset_y=offset_y,
                orientation=orientation,
            )
            if split is not None and len(split) >= 2:
                boxes = split
                method = f"whitespace_gap_{orientation}"
                break
        if boxes is None:
            return _single_panel(width=width, height=height, offset_x=offset_x, offset_y=offset_y)

    labels = _assign_panel_labels(image_bgr, boxes)
    panels: list[PlotPanel] = []
    for index, (bbox, (label, conf, _src)) in enumerate(zip(boxes, labels), start=1):
        panels.append(
            PlotPanel(
                index=index,
                bbox=bbox,
                label=label,
                label_confidence=float(conf),
                detection_method=method,
                digitizable_after_split=True,
                axes_bbox=bbox,
                panel_source_bbox=None,
            )
        )
    return panels


def _row_occupancy_profile(
    cropped_bgr: np.ndarray,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
) -> np.ndarray:
    sub = cropped_bgr[plot_top:plot_bottom, plot_left:plot_right]
    if sub.size == 0:
        return np.array([])

    margin = int(sub.shape[1] * 0.1)
    plot_area = sub[:, margin:]
    gray = cv2.cvtColor(plot_area, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plot_area, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # Colored XRD traces (sat) OR dark ink (black/gray curves). Thresholds are
    # intentionally permissive so thin anti-aliased baselines still register.
    colored = (sat > 25) & (gray < 250)
    dark = gray < 140
    curve_pixels = colored | dark
    return curve_pixels.sum(axis=1).astype(float)


def detect_stacked_curve_bands(
    cropped_bgr: np.ndarray,
    plot_top: int,
    plot_bottom: int,
    plot_left: int,
    plot_right: int,
) -> list[tuple[int, int]]:
    """
    Split a stacked multi-curve plot into horizontal bands, one per curve.

    Uses row occupancy of saturated (non-background) pixels to find valleys
    between vertically offset traces.
    """
    sub_height = plot_bottom - plot_top
    if sub_height <= 0:
        return [(plot_top, plot_bottom)]

    margin_top = int(sub_height * 0.06)
    margin_bottom = int(sub_height * 0.1)
    scan_top = plot_top + margin_top
    scan_bottom = plot_bottom - margin_bottom
    if scan_bottom - scan_top < max(40, sub_height // 4):
        return [(plot_top, plot_bottom)]

    row_counts = _row_occupancy_profile(
        cropped_bgr,
        plot_left,
        plot_right,
        scan_top,
        scan_bottom,
    )
    if row_counts.size == 0 or row_counts.max() <= 0:
        return [(plot_top, plot_bottom)]

    kernel = max(3, sub_height // 50)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = np.convolve(row_counts, np.ones(kernel) / kernel, mode="same")
    threshold = max(1.5, smoothed.max() * 0.08)
    empty = smoothed <= threshold

    bands: list[tuple[int, int]] = []
    start: int | None = None
    # Thin colored baselines often span only ~10-15px in the occupancy profile.
    min_band = max(5, sub_height // 35)
    for row, is_empty in enumerate(empty):
        if not is_empty and start is None:
            start = row
        elif is_empty and start is not None:
            if row - start >= min_band:
                bands.append((scan_top + start, scan_top + row))
            start = None
    if start is not None and len(row_counts) - start >= min_band:
        bands.append((scan_top + start, scan_bottom))

    if len(bands) <= 1:
        return [(plot_top, plot_bottom)]

    merged: list[tuple[int, int]] = []
    min_band_height = max(8, sub_height // 16)
    for band in bands:
        if band[1] - band[0] < min_band_height and merged:
            prev = merged.pop()
            merged.append((prev[0], band[1]))
        else:
            merged.append(band)

    if len(merged) < 2:
        return [(plot_top, plot_bottom)]

    # True stacked XRD traces have multiple comparable-height bands. A dominant
    # band plus a thin fringe (axis ticks / quiet mid-curve gap) is a single curve.
    if not _bands_look_like_stacked_curves(merged, sub_height):
        return [(plot_top, plot_bottom)]
    return merged


def _bands_look_like_stacked_curves(
    bands: list[tuple[int, int]],
    plot_height: int,
) -> bool:
    """Return True only when at least two bands look like separate stacked traces."""
    if len(bands) < 2 or plot_height <= 0:
        return False

    heights = [bottom - top for top, bottom in bands]
    largest = max(heights)
    min_comparable = max(largest * 0.4, plot_height * 0.12)
    comparable = [height for height in heights if height >= min_comparable]
    return len(comparable) >= 2
