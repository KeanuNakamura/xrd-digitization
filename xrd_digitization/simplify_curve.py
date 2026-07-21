from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from xrd_digitization.types import AxisCalibrationResult, CurveData, PeakRecord

# Constant full-width at half-maximum for every reconstructed peak (degrees 2θ).
DEFAULT_PEAK_FWHM_DEG = 0.45

# Only keep peaks that stand clearly above the baseline / noise floor.
DEFAULT_PROMINENCE_FRAC = 0.08
DEFAULT_HEIGHT_FRAC = 0.08
DEFAULT_RELATIVE_HEIGHT_FLOOR = 0.20
DEFAULT_RELATIVE_HEIGHT_FLOOR_NOISY = 0.28
DEFAULT_NOISY_PEAK_COUNT = 5
DEFAULT_MERGE_DISTANCE_DEG = 3.5
DEFAULT_SEGMENT_MERGE_DISTANCE_DEG = 2.5
DEFAULT_MIN_DISTANCE_DEG = 0.6
DEFAULT_LEFT_TRIM_FRAC = 0.03
DEFAULT_RIGHT_TRIM_FRAC = 0.03
DEFAULT_MAX_PEAKS = 12
DEFAULT_MASK_PROFILE_TOLERANCE_DEG = 4.0
_AXIS_FRAME_DARK_FRAC = 0.5


def _is_axis_frame_column(column: np.ndarray, *, dark_threshold: float = 200) -> bool:
    """True for full-height border/axis lines that are not plottable trace columns."""
    if column.size == 0:
        return True
    if float(column.min()) > dark_threshold:
        return False
    dark_frac = float(np.count_nonzero(column < dark_threshold)) / float(len(column))
    return dark_frac > _AXIS_FRAME_DARK_FRAC


def _label_gap_px(plot_height: int) -> int:
    return max(15, plot_height // 15)


def _curve_top_from_dark_run(dark_indices: np.ndarray, *, label_gap_px: int) -> float | None:
    """Return the top of the lowest dark cluster, ignoring Miller labels above a gap."""
    if dark_indices.size == 0:
        return None
    ys = np.sort(dark_indices)
    if ys.size == 1:
        return float(ys[0])
    gaps = np.diff(ys)
    if gaps.size and float(gaps.max()) > label_gap_px:
        ys = ys[: int(np.argmax(gaps)) + 1]
    return float(ys.min())


def _extract_grayscale_profile(
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    band_top: int,
    band_bottom: int,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 1D profile from the darkest trace pixel in each plot column."""
    from xrd_digitization.digitize_xrd_curve import (
        _normalize_intensity,
        _pixel_to_two_theta,
        _pixel_y_to_intensity,
    )

    plot_left = calibration.plot_left
    plot_right = calibration.plot_right
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY) if cropped_bgr.ndim == 3 else cropped_bgr
    roi = gray[band_top:band_bottom, plot_left:plot_right]
    height, width = roi.shape
    if width == 0 or height == 0:
        return np.array([]), np.array([])

    label_margin = int(height * 0.05)
    scan_top = min(label_margin, height - 1)
    dark_threshold = 200
    label_gap_px = _label_gap_px(height)

    x_vals: list[float] = []
    y_vals: list[float] = []
    for col in range(width):
        column = roi[scan_top:, col]
        if column.size == 0:
            continue
        if float(column.min()) > dark_threshold:
            continue
        if _is_axis_frame_column(column, dark_threshold=dark_threshold):
            continue
        dark_indices = np.where(column < dark_threshold)[0]
        trace_top = _curve_top_from_dark_run(dark_indices, label_gap_px=label_gap_px)
        if trace_top is None:
            continue
        y_vals.append(float(trace_top + scan_top))
        x_vals.append(float(col))

    if not x_vals:
        return np.array([]), np.array([])

    x_pixels = np.array(x_vals) + plot_left
    y_pixels = np.array(y_vals) + band_top
    two_theta = _pixel_to_two_theta(x_pixels, calibration)
    intensity = _pixel_y_to_intensity(
        y_pixels,
        calibration,
        band_top=band_top,
        band_bottom=band_bottom,
        stacked=False,
    )
    if not calibration.has_y_calibration and normalize:
        intensity = _normalize_intensity(intensity)
    return two_theta, intensity


def _extract_mask_profile(
    calibration: AxisCalibrationResult,
    curve_mask: np.ndarray,
    *,
    band_top: int,
    band_bottom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 1D profile from the top of a color curve mask in each column."""
    from xrd_digitization.digitize_xrd_curve import (
        _normalize_intensity,
        _pixel_to_two_theta,
        _pixel_y_to_intensity,
    )

    plot_left = calibration.plot_left
    plot_right = calibration.plot_right
    sub = curve_mask[band_top:band_bottom, plot_left:plot_right]
    height, width = sub.shape
    if width == 0 or height == 0:
        return np.array([]), np.array([])

    label_margin = int(height * 0.12)
    scan_top = min(label_margin, height - 1)
    scan_bottom = max(scan_top + 1, int(height * 0.95))
    label_gap_px = max(20, height // 20)

    x_vals: list[float] = []
    y_vals: list[float] = []
    for col in range(width):
        ys = np.where(sub[scan_top:scan_bottom, col] > 0)[0]
        if len(ys) == 0:
            continue
        ys_sorted = np.sort(ys)
        gaps = np.diff(ys_sorted)
        if len(gaps) and float(gaps.max()) > label_gap_px:
            ys_sorted = ys_sorted[: int(np.argmax(gaps)) + 1]
        y_vals.append(float(np.min(ys_sorted) + scan_top))
        x_vals.append(float(col))

    if not x_vals:
        return np.array([]), np.array([])

    x_pixels = np.array(x_vals) + plot_left
    y_pixels = np.array(y_vals) + band_top
    two_theta = _pixel_to_two_theta(x_pixels, calibration)
    intensity = _pixel_y_to_intensity(
        y_pixels,
        calibration,
        band_top=band_top,
        band_bottom=band_bottom,
        stacked=False,
    )
    if not calibration.has_y_calibration:
        intensity = _normalize_intensity(intensity)
    return two_theta, intensity


def _colored_mask_profile_reliable(
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    curve_mask: np.ndarray,
    *,
    band_top: int,
    band_bottom: int,
    tolerance_deg: float = DEFAULT_MASK_PROFILE_TOLERANCE_DEG,
) -> bool:
    """
    Return True when a color mask trace agrees with the grayscale envelope.

    Miller-index labels printed in the curve color create tall mask columns that
    invert peak heights; comparing main-peak positions catches this generically.
    """
    mask_theta, mask_intensity = _extract_mask_profile(
        calibration,
        curve_mask,
        band_top=band_top,
        band_bottom=band_bottom,
    )
    gray_theta, gray_intensity = _extract_grayscale_profile(
        cropped_bgr,
        calibration,
        band_top=band_top,
        band_bottom=band_bottom,
    )
    if len(mask_theta) < 10 or len(gray_theta) < 10:
        return False

    plot_width = max(calibration.plot_right - calibration.plot_left, 1)
    if len(mask_theta) < 0.35 * plot_width:
        return False

    mask_peaks = detect_substantial_peaks(
        mask_theta,
        mask_intensity,
        calibration=calibration,
    )
    gray_peaks = detect_substantial_peaks(
        gray_theta,
        gray_intensity,
        calibration=calibration,
    )
    if not mask_peaks or not gray_peaks:
        return False

    if len(gray_peaks) >= 3 and len(mask_peaks) + 1 < len(gray_peaks):
        return False

    mask_main = max(mask_peaks, key=lambda peak: peak.relative_intensity)
    gray_main = max(gray_peaks, key=lambda peak: peak.relative_intensity)
    return abs(mask_main.two_theta - gray_main.two_theta) <= tolerance_deg


def _extract_curve_profile(
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    band_top: int,
    band_bottom: int,
    curve_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Always use the grayscale envelope for peak positions (labels skew color masks)."""
    del curve_mask
    return _extract_grayscale_profile(
        cropped_bgr,
        calibration,
        band_top=band_top,
        band_bottom=band_bottom,
    )


def _split_profile_segments(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_gap_deg: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a sparse profile at wide sampling gaps so peaks are not smeared away."""
    if len(x) < 3:
        return [(x, y)]

    dx = np.diff(x)
    median_dx = float(np.median(dx))
    gap_threshold = max(min_gap_deg or 0.0, max(3.0 * median_dx, 1.0))
    breaks = np.where(dx > gap_threshold)[0]
    if len(breaks) == 0:
        return [(x, y)]

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for break_idx in breaks:
        end = int(break_idx) + 1
        if end - start >= 3:
            segments.append((x[start:end], y[start:end]))
        start = end
    if len(x) - start >= 3:
        segments.append((x[start:], y[start:]))
    return segments if segments else [(x, y)]


def _profile_between(
    x: np.ndarray,
    y: np.ndarray,
    left_pos: float,
    right_pos: float,
) -> np.ndarray:
    mask = (x >= left_pos) & (x <= right_pos)
    return y[mask] if np.any(mask) else np.array([])


def _should_keep_adjacent_peaks_separate(
    x: np.ndarray,
    y: np.ndarray,
    left: tuple[int, float, float, float],
    right: tuple[int, float, float, float],
    *,
    global_y_max: float,
) -> bool:
    """Keep two nearby maxima when they are both substantial and separated by a valley."""
    left_pos, left_height = left[1], left[2]
    right_pos, right_height = right[1], right[2]
    if right_pos - left_pos < 2.0:
        return False

    ref_height = min(left_height, right_height)
    if ref_height < 0.45 * global_y_max:
        return False
    if left_height >= 0.95 * global_y_max and right_height >= 0.95 * global_y_max:
        return False

    between = _profile_between(x, y, left_pos, right_pos)
    if len(between) < 2:
        return False

    valley = float(np.min(between))
    mean_between = float(np.mean(between))
    if mean_between > 0.55 * ref_height:
        return False
    return valley < 0.20 * ref_height


def _find_peaks_on_segment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prominence_frac: float,
    height_frac: float,
    min_distance_deg: float,
    merge_distance_deg: float,
) -> list[tuple[int, float, float, float]]:
    """Return peak candidates as (grid_idx, position, height, prominence)."""
    if len(x) < 3:
        return []

    grid = np.linspace(float(x.min()), float(x.max()), max(500, len(x) * 8))
    y_grid = np.interp(grid, x, y)
    window = min(15, max(9, len(grid) // 100 * 2 + 1))
    y_smooth = savgol_filter(y_grid, window, 3) if window >= 5 and window < len(y_grid) else y_grid

    y_max = float(np.max(y_smooth))
    if y_max <= 0:
        return []

    dx = float(np.median(np.diff(grid)))
    min_distance_pts = max(1, int(min_distance_deg / max(dx, 1e-6)))
    peaks, properties = find_peaks(
        y_smooth,
        prominence=prominence_frac * y_max,
        height=height_frac * y_max,
        distance=min_distance_pts,
    )
    if len(peaks) == 0:
        peaks, properties = find_peaks(
            y_smooth,
            prominence=max(prominence_frac * 0.6, 0.05) * y_max,
            height=max(height_frac * 0.6, 0.05) * y_max,
            distance=max(1, min_distance_pts // 2),
        )
    if len(peaks) == 0:
        return []

    prominences = properties.get("prominences", np.full(len(peaks), np.nan))
    candidates: list[tuple[int, float, float, float]] = []
    for i, idx in enumerate(peaks):
        pos = float(grid[idx])
        raw_height = float(y_grid[idx])
        prom = float(prominences[i])
        if candidates and pos - candidates[-1][1] < merge_distance_deg:
            if raw_height > candidates[-1][2]:
                candidates[-1] = (int(idx), pos, raw_height, prom)
        else:
            candidates.append((int(idx), pos, raw_height, prom))
    return candidates


def _append_boundary_peak_candidates(
    segments: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[tuple[int, float, float, float]],
    *,
    height_frac: float,
    global_y_max: float,
) -> None:
    """Keep rising signal cut off at segment edges (common when the plot ends on a peak)."""
    if not segments or global_y_max <= 0:
        return

    height_cutoff = height_frac * global_y_max
    merge_guard = 0.75

    for seg_x, seg_y in (segments[0], segments[-1]):
        if len(seg_x) < 3:
            continue
        span = float(seg_x[-1] - seg_x[0])
        if span <= 0:
            continue
        edge_window = max(3, int(len(seg_x) * 0.35))

        left_height = float(seg_y[0])
        left_local = float(np.max(seg_y[:edge_window]))
        if left_local >= height_cutoff and left_local > left_height + 0.05 * global_y_max:
            pos = float(seg_x[int(np.argmax(seg_y[:edge_window]))])
            height = left_local
            close_indices = [
                idx for idx, item in enumerate(candidates) if abs(item[1] - pos) < merge_guard
            ]
            if close_indices:
                target_idx = max(close_indices, key=lambda idx: candidates[idx][2])
                if height > candidates[target_idx][2]:
                    candidates[target_idx] = (0, pos, height, height)
            else:
                candidates.append((0, pos, height, height))

        right_height = float(seg_y[-1])
        right_local = float(np.max(seg_y[-edge_window:]))
        if right_local >= height_cutoff and right_local >= right_height - 0.02 * global_y_max:
            pos = float(seg_x[-edge_window:][int(np.argmax(seg_y[-edge_window:]))])
            height = right_local
            if any(abs(item[1] - pos) < 4.0 for item in candidates):
                continue
            close_indices = [
                idx for idx, item in enumerate(candidates) if abs(item[1] - pos) < merge_guard
            ]
            if close_indices:
                target_idx = max(close_indices, key=lambda idx: candidates[idx][2])
                if height > candidates[target_idx][2]:
                    candidates[target_idx] = (0, pos, height, height)
            else:
                candidates.append((0, pos, height, height))


def _reject_calibration_edge_spikes(
    merged: list[tuple[int, float, float, float]],
    x: np.ndarray,
    y: np.ndarray,
    calibration: AxisCalibrationResult | None,
) -> list[tuple[int, float, float, float]]:
    """Drop corner spikes where the axis border or tick OCR creates a fake tall peak."""
    if calibration is None or len(merged) < 2:
        return merged

    cal_span = max(calibration.x_max - calibration.x_min, 1e-6)
    global_max = float(np.max(y))
    kept: list[tuple[int, float, float, float]] = []
    for item in merged:
        pos, height = item[1], item[2]
        if pos >= calibration.x_max - 0.02 * cal_span and height >= 0.92 * global_max:
            lookback = y[x <= pos - 0.04 * cal_span]
            if len(lookback) >= 3:
                baseline = float(np.median(lookback[-max(5, len(lookback) // 4) :]))
                if baseline < 0.18 * height:
                    continue
        kept.append(item)
    return kept if kept else merged


def _reject_weak_axis_edge_peaks(
    merged: list[tuple[int, float, float, float]],
    calibration: AxisCalibrationResult | None,
) -> list[tuple[int, float, float, float]]:
    """Drop low bumps hugging the left axis where baselines often rise."""
    if calibration is None or not merged:
        return merged

    cal_span = max(calibration.x_max - calibration.x_min, 1e-6)
    global_max = max(item[2] for item in merged)
    kept: list[tuple[int, float, float, float]] = []
    for item in merged:
        pos, height = item[1], item[2]
        if pos <= calibration.x_min + 0.07 * cal_span and height < 0.35 * global_max:
            continue
        kept.append(item)
    return kept if kept else merged


def _normalize_peak_records(peaks: list[PeakRecord]) -> list[PeakRecord]:
    """Express peak heights on a 0–100 scale relative to the tallest peak."""
    if not peaks:
        return peaks
    max_amp = max(max(float(peak.relative_intensity), 0.0) for peak in peaks)
    if max_amp <= 0:
        return peaks
    return [
        PeakRecord(
            two_theta=peak.two_theta,
            relative_intensity=float(peak.relative_intensity / max_amp * 100.0),
            prominence=peak.prominence,
        )
        for peak in peaks
    ]


def _refine_peak_amplitudes_from_grayscale(
    peaks: list[PeakRecord],
    two_theta: np.ndarray,
    raw_intensity: np.ndarray,
) -> list[PeakRecord]:
    """Rescale relative intensities using unnormalized grayscale peak heights."""
    if not peaks or len(two_theta) < 3:
        return peaks

    amps = [
        max(float(np.interp(peak.two_theta, two_theta, raw_intensity)), 0.0)
        for peak in peaks
    ]
    max_amp = max(amps)
    if max_amp <= 0:
        return peaks

    return [
        PeakRecord(
            two_theta=peak.two_theta,
            relative_intensity=float(amp / max_amp * 100.0),
            prominence=peak.prominence,
        )
        for peak, amp in zip(peaks, amps)
    ]


def detect_substantial_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    prominence_frac: float = DEFAULT_PROMINENCE_FRAC,
    height_frac: float = DEFAULT_HEIGHT_FRAC,
    min_distance_deg: float = DEFAULT_MIN_DISTANCE_DEG,
    merge_distance_deg: float = DEFAULT_MERGE_DISTANCE_DEG,
    left_trim_frac: float = DEFAULT_LEFT_TRIM_FRAC,
    right_trim_frac: float = DEFAULT_RIGHT_TRIM_FRAC,
    max_peaks: int = DEFAULT_MAX_PEAKS,
    calibration: AxisCalibrationResult | None = None,
) -> list[PeakRecord]:
    """Find major peaks only, merging nearby detections into one."""
    if len(two_theta) < 10 or len(intensity) < 10:
        return []

    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    span = x.max() - x.min()
    near_axis_left = False
    if calibration is not None:
        cal_span = max(calibration.x_max - calibration.x_min, 1e-6)
        near_axis_left = x.min() <= calibration.x_min + 0.08 * cal_span
        if calibration.has_y_calibration:
            right_trim_frac = min(right_trim_frac, 0.005)
    candidate_lo = x.min() + left_trim_frac * span
    candidate_hi = x.max() - right_trim_frac * span
    left_region = x <= candidate_lo
    left_max = float(np.max(y[left_region])) if np.any(left_region) else 0.0
    global_max = float(np.max(y))
    if near_axis_left:
        trim_lo = candidate_lo
    elif left_max > 0.25 * global_max:
        trim_lo = x.min()
    else:
        trim_lo = candidate_lo
    right_region = x >= candidate_hi
    right_max = float(np.max(y[right_region])) if np.any(right_region) else 0.0
    edge_window = max(3, int(len(x) * 0.02))
    edge_max = float(np.max(y[-edge_window:])) if len(y) >= edge_window else global_max
    if edge_max > 0.22 * global_max:
        trim_hi = x.max()
    elif right_max > 0.22 * global_max:
        trim_hi = x.max()
    else:
        trim_hi = candidate_hi
    keep = (x >= trim_lo) & (x <= trim_hi)
    x = x[keep]
    y = y[keep]
    if len(x) < 10:
        return []

    plot_width = 1
    if calibration is not None:
        plot_width = max(calibration.plot_right - calibration.plot_left, 1)
    if len(x) >= 0.55 * plot_width:
        segments = [(x, y)]
    else:
        segments = _split_profile_segments(x, y)
    global_y_max = float(np.max(y))
    candidates: list[tuple[int, float, float, float]] = []
    segment_merge = min(merge_distance_deg, DEFAULT_SEGMENT_MERGE_DISTANCE_DEG)
    for seg_x, seg_y in segments:
        candidates.extend(
            _find_peaks_on_segment(
                seg_x,
                seg_y,
                prominence_frac=prominence_frac,
                height_frac=height_frac,
                min_distance_deg=min_distance_deg,
                merge_distance_deg=segment_merge,
            )
        )
    _append_boundary_peak_candidates(
        segments,
        candidates,
        height_frac=height_frac,
        global_y_max=global_y_max,
    )
    if not candidates:
        return []

    candidates.sort(key=lambda item: item[1])
    merged: list[tuple[int, float, float, float]] = []
    for idx, pos, height, prom in candidates:
        if merged and pos - merged[-1][1] < merge_distance_deg:
            if _should_keep_adjacent_peaks_separate(
                x,
                y,
                merged[-1],
                (idx, pos, height, prom),
                global_y_max=global_y_max,
            ):
                merged.append((idx, pos, height, prom))
            elif height > merged[-1][2]:
                merged[-1] = (idx, pos, height, prom)
        else:
            merged.append((idx, pos, height, prom))

    merged.sort(key=lambda item: item[2], reverse=True)
    if merged:
        tallest = merged[0][2]
        height_floor = (
            DEFAULT_RELATIVE_HEIGHT_FLOOR_NOISY
            if len(merged) > DEFAULT_NOISY_PEAK_COUNT
            else DEFAULT_RELATIVE_HEIGHT_FLOOR
        )
        if calibration is not None and calibration.has_y_calibration:
            height_floor = min(height_floor, DEFAULT_RELATIVE_HEIGHT_FLOOR)
            y_span = max(calibration.y_max or 0.0, 1e-6)
            abs_cutoff = 0.20 * y_span
            merged = [
                item
                for item in merged
                if item[2] >= max(tallest * height_floor, abs_cutoff)
            ]
        else:
            merged = [item for item in merged if item[2] >= tallest * height_floor]
    merged = _reject_calibration_edge_spikes(merged, x, y, calibration)
    merged = _reject_weak_axis_edge_peaks(merged, calibration)
    merged = merged[:max_peaks]
    merged.sort(key=lambda item: item[1])

    return [
        PeakRecord(
            two_theta=pos,
            relative_intensity=height,
            prominence=prom if np.isfinite(prom) else height,
        )
        for _, pos, height, prom in merged
    ]


def reconstruct_constant_width_peaks(
    peaks: list[PeakRecord],
    two_theta: np.ndarray,
    *,
    fwhm_deg: float = DEFAULT_PEAK_FWHM_DEG,
    baseline_frac: float = 0.02,
) -> np.ndarray:
    """Build a synthetic curve from equal-width Gaussian peaks."""
    grid = np.asarray(two_theta, dtype=float)
    if grid.size == 0:
        return np.array([])
    if not peaks:
        return np.zeros_like(grid)

    sigma = fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    max_height = max(peak.relative_intensity for peak in peaks)
    baseline = max_height * baseline_frac
    intensity = np.full(grid.shape, baseline, dtype=float)

    for peak in peaks:
        intensity += peak.relative_intensity * np.exp(
            -0.5 * ((grid - peak.two_theta) / max(sigma, 1e-6)) ** 2
        )
    return intensity


def _build_mask_intensity_profile(
    calibration: AxisCalibrationResult,
    curve_mask: np.ndarray,
    *,
    band_top: int,
    band_bottom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smoothed raw-intensity profile from color-mask column tops."""
    from xrd_digitization.digitize_xrd_curve import _pixel_to_two_theta, _pixel_y_to_intensity

    pl, pr = calibration.plot_left, calibration.plot_right
    sub = curve_mask[band_top:band_bottom, pl:pr]
    height, width = sub.shape
    if width == 0 or height == 0:
        return np.array([]), np.array([])

    label_gap_px = max(20, height // 20)
    cols: list[float] = []
    intensities: list[float] = []
    for col in range(width):
        ys = np.where(sub[:, col] > 0)[0]
        if len(ys) == 0:
            continue
        ys_sorted = np.sort(ys)
        gaps = np.diff(ys_sorted)
        if len(gaps) and float(gaps.max()) > label_gap_px:
            ys = ys_sorted[: int(np.argmax(gaps)) + 1]
        peak_top = float(np.min(ys) + band_top)
        cols.append(float(col + pl))
        intensities.append(
            float(
                _pixel_y_to_intensity(
                    np.array([peak_top]),
                    calibration,
                    band_top=band_top,
                    band_bottom=band_bottom,
                    stacked=False,
                )[0]
            )
        )

    if len(cols) < 10:
        return np.array([]), np.array([])

    two_theta = _pixel_to_two_theta(np.array(cols), calibration)
    intensity = np.array(intensities, dtype=float)
    grid = np.linspace(float(two_theta.min()), float(two_theta.max()), max(2000, len(cols)))
    y_grid = np.interp(grid, two_theta, intensity)
    window = min(15, max(9, len(grid) // 100 * 2 + 1))
    if window >= 5 and window < len(y_grid):
        y_grid = savgol_filter(y_grid, window, 3)
    return grid, y_grid


def _wing_segment_max(
    grid: np.ndarray,
    intensity: np.ndarray,
    peak_pos: float,
    *,
    side: str,
    offset_deg: float,
    width_deg: float = 0.8,
) -> float:
    if side == "left":
        mask = (grid >= peak_pos - offset_deg - width_deg) & (grid <= peak_pos - offset_deg)
    else:
        mask = (grid >= peak_pos + offset_deg) & (grid <= peak_pos + offset_deg + width_deg)
    if not np.any(mask):
        return 0.0
    return float(np.max(intensity[mask]))


def _best_shoulder_baseline(
    grid: np.ndarray,
    intensity: np.ndarray,
    peak_pos: float,
    peak_val: float,
    *,
    side: str,
) -> float:
    best = 0.0
    for offset in np.linspace(2.5, 3.5, 11):
        baseline = _wing_segment_max(grid, intensity, peak_pos, side=side, offset_deg=float(offset))
        if baseline < peak_val * 0.98 and baseline > best:
            best = baseline
    return best


def _measure_color_mask_amplitudes(
    grid: np.ndarray,
    intensity: np.ndarray,
    peaks: list[PeakRecord],
    *,
    wing_deg: float = 2.0,
) -> list[float]:
    """Measure peak heights above local shoulders using raw (non-p99) intensity."""
    if not peaks:
        return []

    floor = float(np.percentile(intensity, 3))
    sorted_peaks = sorted(peaks, key=lambda peak: peak.two_theta)
    peak_vals = {
        peak.two_theta: float(np.interp(peak.two_theta, grid, intensity)) for peak in sorted_peaks
    }

    amplitudes: list[float] = []
    for index, peak in enumerate(sorted_peaks):
        pos = peak.two_theta
        peak_val = peak_vals[pos]
        baseline = floor
        baseline = max(
            baseline,
            _wing_segment_max(grid, intensity, pos, side="left", offset_deg=wing_deg),
            _wing_segment_max(grid, intensity, pos, side="right", offset_deg=wing_deg),
        )

        if index > 0:
            left_peak = sorted_peaks[index - 1]
            if (
                pos - left_peak.two_theta <= 10.0
                and peak_vals[left_peak.two_theta] > peak_val * 0.9
            ):
                baseline = max(
                    baseline,
                    _best_shoulder_baseline(grid, intensity, pos, peak_val, side="left"),
                )

        if index + 1 < len(sorted_peaks):
            right_peak = sorted_peaks[index + 1]
            if (
                right_peak.two_theta - pos <= 10.0
                and peak_vals[right_peak.two_theta] > peak_val * 0.9
            ):
                baseline = max(
                    baseline,
                    _best_shoulder_baseline(grid, intensity, pos, peak_val, side="right"),
                )

        amplitudes.append(max(peak_val - baseline, 0.0))
    return amplitudes


def _merge_amplitude_refinements(
    gray_peaks: list[PeakRecord],
    mask_peaks: list[PeakRecord],
    *,
    tolerance: float = 0.25,
) -> list[PeakRecord]:
    """Keep grayscale amplitudes when a color mask clearly over/under-shoots."""
    if len(gray_peaks) != len(mask_peaks):
        return gray_peaks

    merged: list[PeakRecord] = []
    for gray_peak, mask_peak in zip(gray_peaks, mask_peaks):
        gray_amp = max(float(gray_peak.relative_intensity), 0.0)
        mask_amp = max(float(mask_peak.relative_intensity), 0.0)
        if gray_amp <= 0:
            chosen = mask_peak
        elif mask_amp <= 0:
            chosen = gray_peak
        elif abs(mask_amp - gray_amp) <= tolerance * max(gray_amp, mask_amp):
            chosen = mask_peak
        else:
            chosen = gray_peak
        merged.append(
            PeakRecord(
                two_theta=gray_peak.two_theta,
                relative_intensity=float(chosen.relative_intensity),
                prominence=gray_peak.prominence,
            )
        )
    return merged


def _refine_peak_amplitudes(
    peaks: list[PeakRecord],
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    band_top: int,
    band_bottom: int,
    curve_mask: np.ndarray | None = None,
) -> list[PeakRecord]:
    """Re-measure peak heights from a color-mask profile without p99 normalization."""
    if not peaks:
        return peaks

    if curve_mask is None or not np.any(curve_mask):
        tallest = max(peak.relative_intensity for peak in peaks)
        if tallest <= 0:
            return peaks
        return [
            PeakRecord(
                two_theta=peak.two_theta,
                relative_intensity=float(peak.relative_intensity / tallest * 100.0),
                prominence=peak.prominence,
            )
            for peak in peaks
        ]

    grid, intensity = _build_mask_intensity_profile(
        calibration,
        curve_mask,
        band_top=band_top,
        band_bottom=band_bottom,
    )
    if len(grid) < 10:
        return peaks

    measured = _measure_color_mask_amplitudes(grid, intensity, peaks)
    max_amp = max(measured) if measured else 0.0
    if max_amp <= 0:
        return peaks

    return [
        PeakRecord(
            two_theta=peak.two_theta,
            relative_intensity=float(amp / max_amp * 100.0),
            prominence=peak.prominence,
        )
        for peak, amp in zip(peaks, measured)
    ]


def _drop_weak_left_edge_peaks(
    peaks: list[PeakRecord],
    calibration: AxisCalibrationResult,
) -> list[PeakRecord]:
    """Remove low bumps at the left edge after amplitude refinement."""
    if not peaks:
        return peaks
    cal_span = max(calibration.x_max - calibration.x_min, 1e-6)
    max_amp = max(float(peak.relative_intensity) for peak in peaks)
    kept = [
        peak
        for peak in peaks
        if not (
            peak.two_theta <= calibration.x_min + 0.06 * cal_span
            and float(peak.relative_intensity) < 0.40 * max_amp
        )
    ]
    return kept if kept else peaks


def simplify_single_curve(
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    band_top: int,
    band_bottom: int,
    num_points: int = 2000,
    fwhm_deg: float = DEFAULT_PEAK_FWHM_DEG,
    curve_mask: np.ndarray | None = None,
) -> tuple[CurveData, list[PeakRecord]]:
    """
    Extract a simplified single-curve XRD pattern from substantial peaks only.

    Uses a color-mask or grayscale column profile, detects major peaks, and
    reconstructs the curve with fixed-width Gaussian peaks.
    """
    two_theta, intensity = _extract_curve_profile(
        cropped_bgr,
        calibration,
        band_top=band_top,
        band_bottom=band_bottom,
        curve_mask=curve_mask,
    )
    _, raw_intensity = _extract_grayscale_profile(
        cropped_bgr,
        calibration,
        band_top=band_top,
        band_bottom=band_bottom,
        normalize=False,
    )
    warnings: list[str] = ["simplified_constant_width_peaks"]

    if len(two_theta) < 10:
        return (
            CurveData(two_theta=[], intensity=[], warnings=["profile_extraction_failed"]),
            [],
        )

    peaks = detect_substantial_peaks(two_theta, intensity, calibration=calibration)
    if not peaks:
        warnings.append("no_substantial_peaks_detected")
        return (
            CurveData(
                two_theta=two_theta.tolist(),
                intensity=intensity.tolist(),
                warnings=warnings,
            ),
            [],
        )

    if not calibration.has_y_calibration:
        gray_refined = _refine_peak_amplitudes_from_grayscale(peaks, two_theta, raw_intensity)
        if (
            curve_mask is not None
            and np.any(curve_mask)
            and _colored_mask_profile_reliable(
                cropped_bgr,
                calibration,
                curve_mask,
                band_top=band_top,
                band_bottom=band_bottom,
            )
        ):
            mask_refined = _refine_peak_amplitudes(
                gray_refined,
                cropped_bgr,
                calibration,
                band_top=band_top,
                band_bottom=band_bottom,
                curve_mask=curve_mask,
            )
            peaks = _merge_amplitude_refinements(gray_refined, mask_refined)
        else:
            peaks = gray_refined
    elif (
        curve_mask is not None
        and np.any(curve_mask)
        and _colored_mask_profile_reliable(
            cropped_bgr,
            calibration,
            curve_mask,
            band_top=band_top,
            band_bottom=band_bottom,
        )
    ):
        peaks = _refine_peak_amplitudes(
            peaks,
            cropped_bgr,
            calibration,
            band_top=band_top,
            band_bottom=band_bottom,
            curve_mask=curve_mask,
        )

    if not calibration.has_y_calibration and peaks:
        peaks = _drop_weak_left_edge_peaks(peaks, calibration)

    grid = np.linspace(float(np.min(two_theta)), float(np.max(two_theta)), num_points)
    reconstructed = reconstruct_constant_width_peaks(peaks, grid, fwhm_deg=fwhm_deg)

    return (
        CurveData(
            two_theta=grid.tolist(),
            intensity=reconstructed.tolist(),
            warnings=warnings,
        ),
        peaks,
    )
