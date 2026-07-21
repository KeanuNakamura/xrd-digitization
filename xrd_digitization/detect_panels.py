from __future__ import annotations

import cv2
import numpy as np

from xrd_digitization.types import PlotPanel


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


def _panel_separator_score(
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
    if gap_curve > 0.5:
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

    # Subplot separators usually start a new framed panel below the gap.
    below_gray = gray[gap_end:, :]
    axis_window = below_gray[: max(20, int(below_h * 0.2)), :]
    dark_rows = (axis_window < 120).sum(axis=1)
    has_frame_below = bool(len(dark_rows) and dark_rows.max() >= width * 0.25)

    gap_score = min(1.0, gap_len / max(12.0, height * 0.06))
    cleanliness = 1.0 - min(1.0, gap_curve * 20.0)
    centrality = 1.0 - abs(gap_center / height - 0.5) * 1.2
    frame_bonus = 1.25 if has_frame_below else 0.55
    return gap_score * cleanliness * centrality * size_ratio * frame_bonus


def detect_plot_panels(
    image_bgr: np.ndarray,
    *,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> list[PlotPanel]:
    """
    Split a figure into vertically stacked plot panels when clear whitespace
    gaps separate subplots (e.g. figure_7 with top and bottom XRD patterns).
    """
    if crop_bbox is not None:
        x0, y0, x1, y1 = crop_bbox
        region = image_bgr[y0:y1, x0:x1]
        offset_y = y0
    else:
        region = image_bgr
        offset_y = 0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    if height < 80:
        return [
            PlotPanel(
                index=1,
                bbox=(0, offset_y, region.shape[1], offset_y + height),
                label=None,
            )
        ]

    curve_mask = _curve_pixel_mask(region)
    runs = _horizontal_whitespace_runs(gray)

    best: tuple[float, int, int] | None = None
    for gap_start, gap_end in runs:
        score = _panel_separator_score(gray, curve_mask, gap_start, gap_end)
        if score <= 0.45:
            continue
        if best is None or score > best[0]:
            best = (score, gap_start, gap_end)

    if best is None:
        return [
            PlotPanel(
                index=1,
                bbox=(0, offset_y, region.shape[1], offset_y + height),
                label=None,
            )
        ]

    _, gap_start, gap_end = best
    panels = [
        PlotPanel(index=1, bbox=(0, offset_y, region.shape[1], offset_y + gap_start), label="top"),
        PlotPanel(index=2, bbox=(0, offset_y + gap_end, region.shape[1], offset_y + height), label="bottom"),
    ]

    min_panel_height = max(40, int(height * 0.12))
    valid = [p for p in panels if (p.bbox[3] - p.bbox[1]) >= min_panel_height]
    if len(valid) < 2:
        return [
            PlotPanel(
                index=1,
                bbox=(0, offset_y, region.shape[1], offset_y + height),
                label=None,
            )
        ]
    return valid


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
    colored = sat > 35
    dark = gray < 85
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
    min_band = max(6, sub_height // 20)
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
    return merged if len(merged) >= 2 else [(plot_top, plot_bottom)]
