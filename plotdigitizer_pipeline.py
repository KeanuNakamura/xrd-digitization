from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from compute_sid import compare_spectra
from plotdigitizer_extract import (
    extract_curve_autonomous,
    inset_plot_bounds,
    map_path_to_calibrated_xy,
    write_debug_outputs,
)
from xrd_digitization.calibrate_axes import calibrate_axes
from xrd_digitization.crop_plot_area import crop_plot_area
from xrd_digitization.detect_panels import detect_stacked_curve_bands
from xrd_digitization.types import AxisCalibrationResult

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
BAND_PADDING = 4
MIN_BAND_HEIGHT = 120
PATTERN_STEM_RE = re.compile(r"^pattern_(\d+)$", re.IGNORECASE)
DEFAULT_CNRS_DIGITIZED_DIR = REPO_ROOT / "data" / "CNRS_digitized"
DEFAULT_CNRS_JSON_DIR = REPO_ROOT / "data" / "CNRS"


@dataclass
class PlotDigitizerPoints:
    data_points: list[tuple[float, float]]
    locations: list[tuple[int, int]]


@dataclass
class PlotDigitizerCalibration:
    """Calibration bundle for PlotDigitizer plus post-processing remap."""

    points: PlotDigitizerPoints
    image_height: int
    x_true_range: tuple[float, float]
    y_true_range: tuple[float, float]
    x_pd_range: tuple[float, float]
    y_pd_range: tuple[float, float]
    x_anchor_ticks: tuple[float, float]
    warnings: list[str] = field(default_factory=list)


DEFAULT_Y_SCALE = 500.0
MAX_REASONABLE_Y_TICK = 200_000.0
MIN_Y_ANCHOR_ROW_SPAN = 20
# PlotDigitizer's origin finder divides by (p2.x - p1.x); keep top-left x 1px right of bottom-left.
PLOTDIGITIZER_TOP_LEFT_X_NUDGE = 1


def _x_pixel_columns(calibration: AxisCalibrationResult) -> tuple[float, float]:
    """Map plot frame edges to crop-local pixel columns."""
    if not calibration.tick_pairs:
        raise ValueError("Cannot build PlotDigitizer calibration without OCR x ticks")
    return float(calibration.plot_left), float(calibration.plot_right)


def _y_max_tick_label(calibration: AxisCalibrationResult) -> float:
    """Highest plausible y-axis tick label, used to locate the top -l anchor row."""
    if calibration.y_tick_pairs:
        from xrd_digitization.calibrate_axes import _select_arithmetic_number_sequence

        by_row: dict[int, list[float]] = {}
        for row, value in calibration.y_tick_pairs:
            by_row.setdefault(row, []).append(value)
        deduped_values = [min(values) for values in by_row.values() if values]
        sequence = _select_arithmetic_number_sequence(deduped_values)
        plausible = [value for value in sequence if 0.0 < value <= MAX_REASONABLE_Y_TICK]
        if not plausible:
            plausible = [value for value in deduped_values if 0.0 < value <= MAX_REASONABLE_Y_TICK]
        if plausible:
            return float(max(plausible))
    if calibration.y_max is not None and 0.0 < calibration.y_max <= MAX_REASONABLE_Y_TICK:
        return float(calibration.y_max)
    return DEFAULT_Y_SCALE


def _y_axis_anchor_rows(
    calibration: AxisCalibrationResult,
    *,
    full_image_bgr: np.ndarray | None,
    crop_bbox: tuple[int, int, int, int] | None,
    y_top: float,
) -> tuple[int, int]:
    """
    Return crop-local rows for the x-axis (y=0) and top y-tick anchors.

    PlotDigitizer expects the third anchor on the y-axis at the highest labeled
    tick value, not at the top of the plot frame.
    """
    row_bottom = calibration.plot_bottom
    row_top = calibration.plot_top

    if full_image_bgr is None or crop_bbox is None:
        return row_bottom, row_top

    from xrd_digitization.calibrate_axes import (
        _ocr_y_tick_labels_full_image,
        _select_arithmetic_number_sequence,
    )

    labels = _ocr_y_tick_labels_full_image(full_image_bgr, crop_bbox)
    pairs = [
        (row, value)
        for row, value, _ in labels
        if 0.0 <= value <= MAX_REASONABLE_Y_TICK
    ]
    if len(pairs) >= 2:
        by_row: dict[int, float] = {}
        for row, value in pairs:
            existing = by_row.get(row)
            if existing is None or value < existing:
                by_row[row] = value
        deduped = sorted(by_row.items(), key=lambda item: item[1])
        values = [value for _, value in deduped]
        sequence = _select_arithmetic_number_sequence(values)
        fit_pairs = [
            (row, value)
            for row, value in deduped
            if not sequence or round(value) in {round(v) for v in sequence}
        ]
        if len(fit_pairs) < 2:
            fit_pairs = deduped
        if len(fit_pairs) >= 2:
            rows = np.array([row for row, _ in fit_pairs], dtype=float)
            vals = np.array([value for _, value in fit_pairs], dtype=float)
            slope, intercept = np.polyfit(vals, rows, 1)
            row_bottom = int(round(float(intercept)))
            row_top = int(round(float(slope * y_top + intercept)))
            row_bottom = int(np.clip(row_bottom, calibration.plot_top, calibration.plot_bottom))
            row_top = int(np.clip(row_top, calibration.plot_top, calibration.plot_bottom))
            if row_top >= row_bottom:
                row_top = max(calibration.plot_top, row_bottom - 1)
            if (row_bottom - row_top) < MIN_Y_ANCHOR_ROW_SPAN:
                row_bottom = calibration.plot_bottom
                row_top = calibration.plot_top

    return row_bottom, row_top


def _plotdigitizer_locations(
    calibration: AxisCalibrationResult,
    *,
    image_height: int,
    frame_offset_x: int = 0,
    frame_offset_y: int = 0,
    full_image_bgr: np.ndarray | None = None,
    crop_bbox: tuple[int, int, int, int] | None = None,
    plot_top: int | None = None,
    plot_bottom: int | None = None,
) -> list[tuple[int, int]]:
    """Bottom-left pixel anchors aligned with OCR axis ticks (GUI-style mapping)."""
    plot_top = calibration.plot_top if plot_top is None else plot_top
    plot_bottom = calibration.plot_bottom if plot_bottom is None else plot_bottom
    y_tick_top = _y_max_tick_label(calibration)
    row_bottom, row_top = _y_axis_anchor_rows(
        calibration,
        full_image_bgr=full_image_bgr,
        crop_bbox=crop_bbox,
        y_top=y_tick_top,
    )

    col_lo, col_hi = _x_pixel_columns(calibration)
    bottom_left_x = frame_offset_x + int(round(col_lo))
    top_left_x = bottom_left_x + PLOTDIGITIZER_TOP_LEFT_X_NUDGE
    right_x = frame_offset_x + int(round(col_hi))

    def to_bottom_left(col: int, row_top_coord: int) -> tuple[int, int]:
        return _pixel_to_bottom_left(
            col,
            frame_offset_y + row_top_coord,
            image_height,
        )

    return [
        to_bottom_left(bottom_left_x, row_bottom),
        to_bottom_left(right_x, row_bottom),
        to_bottom_left(top_left_x, row_top),
    ]


def _plotdigitizer_data_points(calibration: AxisCalibrationResult) -> list[tuple[float, float]]:
    """Physical data-space anchors matching manual PlotDigitizer GUI entry."""
    x_span = max(calibration.x_max - calibration.x_min, 1.0)
    y_scale, _ = _estimate_y_scale(calibration)
    return [
        (0.0, 0.0),
        (float(x_span), 0.0),
        (0.0, float(y_scale)),
    ]


def _is_normalized_y_axis(calibration: AxisCalibrationResult) -> bool:
    if calibration.y_max is not None and calibration.y_min is not None:
        if calibration.y_max <= 1.5 and calibration.y_min >= -0.05:
            return True
    if calibration.y_tick_pairs:
        values = [value for _, value in calibration.y_tick_pairs]
        if values and max(values) <= 1.5 and min(values) >= -0.05:
            return True
    return False


def _has_reliable_y_calibration(calibration: AxisCalibrationResult) -> bool:
    if not calibration.has_y_calibration:
        return False
    if calibration.y_min is None or calibration.y_max is None:
        return False
    if calibration.y_max <= calibration.y_min:
        return False
    if calibration.y_min < 0.0:
        return False
    span = calibration.y_max - calibration.y_min
    if _is_normalized_y_axis(calibration):
        if span < 0.05:
            return False
    elif span < 1.0:
        return False
    if not _is_normalized_y_axis(calibration) and calibration.y_max > MAX_REASONABLE_Y_TICK:
        return False

    y_ticks = calibration.y_tick_pairs
    if len(y_ticks) < 2:
        return False

    pixel_rows = {row for row, _ in y_ticks}
    if len(pixel_rows) < 2:
        return False

    values = sorted(value for _, value in y_ticks)
    if values[-1] <= values[0]:
        return False

    return True


def _pixel_to_bottom_left(
    col: int,
    row_top: int,
    image_height: int,
) -> tuple[int, int]:
    return col, image_height - row_top


def _plot_corner_locations(
    *,
    image_height: int,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[tuple[int, int]]:
    """Backward-compatible frame-corner helper for band-local coordinates."""
    axis_row_top = offset_y + plot_bottom
    top_row_top = offset_y + plot_top
    bottom_left_x = offset_x + plot_left
    top_left_x = bottom_left_x + PLOTDIGITIZER_TOP_LEFT_X_NUDGE
    return [
        _pixel_to_bottom_left(bottom_left_x, axis_row_top, image_height),
        _pixel_to_bottom_left(offset_x + plot_right, axis_row_top, image_height),
        _pixel_to_bottom_left(top_left_x, top_row_top, image_height),
    ]


def _plotdigitizer_axis_transform(
    locations: list[tuple[int, int]],
    data_points: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Mirror PlotDigitizer's axis_transformation() naming and return order."""
    px = np.array([loc[0] for loc in locations], dtype=float)
    py = np.array([loc[1] for loc in locations], dtype=float)
    data_x = np.array([pt[0] for pt in data_points], dtype=float)
    data_y = np.array([pt[1] for pt in data_points], dtype=float)
    slope_x, intercept_x = np.polyfit(px, data_x, 1)
    slope_y, intercept_y = np.polyfit(py, data_y, 1)
    return (float(intercept_x), float(intercept_y)), (float(slope_x), float(slope_y))


def _plotdigitizer_x_output(
    pixel_col: float,
    *,
    locations: list[tuple[int, int]],
    data_points: list[tuple[float, float]],
) -> float:
    """PlotDigitizer raw x output at a bottom-left pixel column."""
    (scale_x, _), (offset_x, _) = _plotdigitizer_axis_transform(locations, data_points)
    if abs(scale_x) < 1e-12:
        return 0.0
    return float(-(pixel_col - offset_x) / scale_x)


def _plotdigitizer_x_output_range(points: PlotDigitizerPoints) -> tuple[float, float]:
    """Expected PlotDigitizer x output at the left and right x-axis anchors."""
    left_col = float(points.locations[0][0])
    right_col = float(points.locations[1][0])
    lo = _plotdigitizer_x_output(
        left_col,
        locations=points.locations,
        data_points=points.data_points,
    )
    hi = _plotdigitizer_x_output(
        right_col,
        locations=points.locations,
        data_points=points.data_points,
    )
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _predict_plotdigitizer_value(
    pixel_x: int,
    pixel_y_bl: int,
    points: PlotDigitizerPoints,
    image_height: int,
) -> tuple[float, float]:
    """Replicate PlotDigitizer's pixel-to-output transform for anchor pixels."""
    del image_height  # kept for backward compatibility
    raw_x = _plotdigitizer_x_output(
        pixel_x,
        locations=points.locations,
        data_points=points.data_points,
    )
    (_, scale_y), (_, offset_y) = _plotdigitizer_axis_transform(
        points.locations,
        points.data_points,
    )
    if abs(scale_y) < 1e-12:
        return raw_x, 0.0
    row_top = image_height - pixel_y_bl
    raw_y = -(row_top - offset_y) / scale_y
    return raw_x, float(raw_y)


def _remap_axis(
    values: np.ndarray,
    pd_range: tuple[float, float],
    true_range: tuple[float, float],
) -> np.ndarray:
    pd0, pd1 = pd_range
    t0, t1 = true_range
    if abs(pd1 - pd0) < 1e-9:
        return np.full_like(values, t0, dtype=float)
    return (values - pd0) / (pd1 - pd0) * (t1 - t0) + t0


def _correct_plotdigitizer_array(
    data: np.ndarray,
    calibration: PlotDigitizerCalibration,
) -> np.ndarray:
    """Remap PlotDigitizer raw coordinates without writing a CSV file."""
    if data.ndim == 1:
        data = data.reshape(1, 2)

    corrected = data.copy()
    x_pd0, x_pd1 = calibration.x_pd_range
    if abs(x_pd1 - x_pd0) < 1e-9:
        corrected[:, 0] = calibration.x_true_range[0]
    else:
        corrected[:, 0] = _remap_axis(
            data[:, 0],
            (x_pd0, x_pd1),
            calibration.x_true_range,
        )

    raw_y = data[:, 1]
    finite = np.isfinite(raw_y)
    if not finite.all():
        raw_y = raw_y[finite]
        if raw_y.size == 0:
            corrected[:, 1] = 0.0
            return corrected
    y_pd_min = float(np.nanmin(raw_y))
    y_pd_max = float(np.nanmax(raw_y))
    if abs(y_pd_max - y_pd_min) < 1e-9:
        corrected[:, 1] = 0.0
    else:
        corrected[:, 1] = _remap_axis(
            raw_y,
            (y_pd_min, y_pd_max),
            calibration.y_true_range,
        )
    corrected[:, 1] = np.maximum(corrected[:, 1], 0.0)
    return corrected


def build_plotdigitizer_calibration(
    calibration: AxisCalibrationResult,
    *,
    image_height: int,
    frame_offset_x: int = 0,
    frame_offset_y: int = 0,
    plot_top: int | None = None,
    plot_bottom: int | None = None,
    full_image_bgr: np.ndarray | None = None,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> PlotDigitizerCalibration:
    """
    Build PlotDigitizer anchors using physical axis values (-p) and tick-aligned
    pixel locations (-l), matching manual GUI digitization.
    """
    warnings: list[str] = []
    y_scale, y_warnings = _estimate_y_scale(calibration)
    warnings.extend(y_warnings)

    if full_image_bgr is not None and crop_bbox is not None:
        locations = _plotdigitizer_locations(
            calibration,
            image_height=image_height,
            frame_offset_x=frame_offset_x,
            frame_offset_y=frame_offset_y,
            full_image_bgr=full_image_bgr,
            crop_bbox=crop_bbox,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
        )
    else:
        locations = _plot_corner_locations(
            image_height=image_height,
            plot_left=calibration.plot_left,
            plot_right=calibration.plot_right,
            plot_top=calibration.plot_top if plot_top is None else plot_top,
            plot_bottom=calibration.plot_bottom if plot_bottom is None else plot_bottom,
            offset_x=frame_offset_x,
            offset_y=frame_offset_y,
        )
        warnings.append("using_frame_corners_without_tick_y_anchor")

    data_points = _plotdigitizer_data_points(calibration)
    points = PlotDigitizerPoints(data_points=data_points, locations=locations)
    x_pd_lo, x_pd_hi = _plotdigitizer_x_output_range(points)

    return PlotDigitizerCalibration(
        points=points,
        image_height=image_height,
        x_true_range=(calibration.x_min, calibration.x_max),
        y_true_range=(0.0, y_scale),
        x_pd_range=(x_pd_lo, x_pd_hi),
        y_pd_range=(0.0, 0.0),
        x_anchor_ticks=(calibration.x_min, calibration.x_max),
        warnings=warnings,
    )


def calibration_to_plotdigitizer_points(
    calibration: AxisCalibrationResult,
    *,
    image_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> PlotDigitizerPoints:
    """Backward-compatible wrapper returning only PlotDigitizer anchor triples."""
    if offset_x or offset_y:
        raise ValueError(
            "calibration_to_plotdigitizer_points now expects crop-local coordinates; "
            "use build_plotdigitizer_calibration() on a cropped plot image instead."
        )
    return build_plotdigitizer_calibration(
        calibration,
        image_height=image_height,
        frame_offset_x=offset_x,
        frame_offset_y=offset_y,
    ).points


def _estimate_y_scale(calibration: AxisCalibrationResult) -> tuple[float, list[str]]:
    """Choose a physical y-axis span for post-processing PlotDigitizer output."""
    warnings: list[str] = []
    if _is_normalized_y_axis(calibration):
        if _has_reliable_y_calibration(calibration):
            return float(calibration.y_max - calibration.y_min), warnings
        if calibration.y_tick_pairs:
            values = [value for _, value in calibration.y_tick_pairs]
            hi, lo = max(values), min(values)
            if hi > lo:
                return float(hi - lo), warnings
        return 1.0, warnings

    if _has_reliable_y_calibration(calibration):
        return float(calibration.y_max - calibration.y_min), warnings

    if calibration.y_tick_pairs:
        from xrd_digitization.calibrate_axes import _select_arithmetic_number_sequence

        by_row: dict[int, list[float]] = {}
        for row, value in calibration.y_tick_pairs:
            by_row.setdefault(row, []).append(value)
        deduped_values = [min(values) for values in by_row.values() if values]

        sequence = _select_arithmetic_number_sequence(deduped_values)
        plausible = [value for value in sequence if 0.0 < value <= MAX_REASONABLE_Y_TICK]
        if not plausible:
            plausible = [value for value in deduped_values if 0.0 < value <= MAX_REASONABLE_Y_TICK]
        if len(plausible) >= 2:
            lo, hi = min(plausible), max(plausible)
            if hi > lo:
                warnings.append("using_y_tick_span_for_scale")
                return float(hi - lo), warnings

    warnings.append("using_relative_y_scale")
    return DEFAULT_Y_SCALE, warnings


def correct_plotdigitizer_csv(
    csv_path: Path,
    calibration: PlotDigitizerCalibration,
) -> np.ndarray:
    """Remap PlotDigitizer raw CSV coordinates onto OCR-calibrated axis ranges."""
    data = np.loadtxt(csv_path)
    corrected = _correct_plotdigitizer_array(data, calibration)

    with csv_path.open("w", encoding="utf-8") as handle:
        for x_val, y_val in corrected:
            handle.write(f"{x_val:g} {y_val:g}\n")

    return corrected


def _csv_quality_score(csv_path: Path, calibration: PlotDigitizerCalibration) -> float:
    """Higher is better: reward point count and y resolution after correction."""
    try:
        raw = np.loadtxt(csv_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, 2)
    except Exception:
        return -1.0

    try:
        data = _correct_plotdigitizer_array(raw, calibration)
    except Exception:
        return -1.0
    if len(data) < 20:
        return float(len(data))
    y_unique = len(np.unique(np.round(data[:, 1], 1)))
    if y_unique < 8:
        return float(y_unique)
    y_span = float(data[:, 1].max() - data[:, 1].min())
    if y_span < max(1.0, calibration.y_true_range[1] * 0.05):
        return float(y_unique) * 0.1
    # Grid artifacts inflate point count; prefer typical XRD curve sizes (~200-800 pts).
    count_penalty = 0.0
    if len(data) > 900:
        count_penalty = (len(data) - 900) * 0.5
    raw_y_span = float(raw[:, 1].max() - raw[:, 1].min())
    return 100.0 * y_unique + len(data) * 0.01 - count_penalty + raw_y_span * 10.0


def save_digitized_preview(
    csv_path: Path,
    plot_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Write a simple preview plot from corrected CSV data."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available; skipping preview plot %s", plot_path)
        return

    data = np.loadtxt(csv_path)
    if data.ndim == 1:
        data = data.reshape(1, 2)

    fig, axis = plt.subplots(figsize=(8, 4))
    # Break the line on large x-gaps so sparse traces do not draw fake ramps.
    if len(data) >= 2:
        x = data[:, 0]
        y = data[:, 1]
        dx = np.diff(x)
        typical = float(np.median(dx[dx > 0])) if np.any(dx > 0) else 1.0
        breaks = np.where(dx > max(3.0 * typical, 1.0))[0]
        start = 0
        for b in list(breaks) + [len(data) - 1]:
            end = b + 1
            axis.plot(x[start:end], y[start:end], color="black", linewidth=1.0)
            start = end
    else:
        axis.plot(data[:, 0], data[:, 1], color="black", linewidth=1.0)
    axis.set_xlabel("2θ (degrees)")
    axis.set_ylabel("Intensity (a.u.)")
    if title:
        axis.set_title(title)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


@dataclass
class BandDigitizationResult:
    stem: str
    source_image: Path
    csv_path: Path
    plot_path: Path | None
    data_points: list[tuple[float, float]]
    locations: list[tuple[int, int]]
    calibration: dict[str, Any]
    warnings: list[str]
    success: bool
    error: str | None = None


@dataclass
class FigureDigitizationResult:
    figure_id: str
    source_image: Path
    bands: list[BandDigitizationResult]
    warnings: list[str]


def detect_figure_bands(
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
) -> list[tuple[int, int]]:
    """Return horizontal band ranges within the cropped plot area."""
    bands = detect_stacked_curve_bands(
        cropped_bgr,
        calibration.plot_top,
        calibration.plot_bottom,
        calibration.plot_left,
        calibration.plot_right,
    )
    if len(bands) <= 1:
        return [(calibration.plot_top, calibration.plot_bottom)]

    plot_height = max(1, calibration.plot_bottom - calibration.plot_top)
    # Drop tiny false bands (e.g. a thick tick / label strip mistaken for a curve).
    substantial = [
        (top, bottom)
        for top, bottom in bands
        if (bottom - top) >= max(40, int(plot_height * 0.15))
    ]
    if len(substantial) <= 1:
        return [(calibration.plot_top, calibration.plot_bottom)]
    return substantial


def _expand_band_range(
    band_top: int,
    band_bottom: int,
    *,
    max_height: int,
) -> tuple[int, int]:
    """Expand thin bands so PlotDigitizer has enough plot area to extract a curve."""
    height = band_bottom - band_top
    if height >= MIN_BAND_HEIGHT:
        return band_top, band_bottom

    center = (band_top + band_bottom) // 2
    half = MIN_BAND_HEIGHT // 2
    expanded_top = max(0, center - half)
    expanded_bottom = min(max_height, center + half)
    if expanded_bottom - expanded_top < MIN_BAND_HEIGHT:
        expanded_top = max(0, expanded_bottom - MIN_BAND_HEIGHT)
    return expanded_top, expanded_bottom


def should_invert_image(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < 128.0


def _plot_area_mask(
    height: int,
    width: int,
    *,
    plot_bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Mask excluding outer margins where axis labels and spines live."""
    if plot_bounds is not None:
        left, top, right, bottom = plot_bounds
        mask = np.zeros((height, width), dtype=bool)
        left = int(np.clip(left, 0, width))
        right = int(np.clip(right, 0, width))
        top = int(np.clip(top, 0, height))
        bottom = int(np.clip(bottom, 0, height))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
        return mask

    margin_mask = np.ones((height, width), dtype=bool)
    margin_mask[: int(height * 0.08), :] = False
    margin_mask[int(height * 0.92) :, :] = False
    margin_mask[:, : int(width * 0.08)] = False
    margin_mask[:, int(width * 0.96) :] = False
    return margin_mask


def _prepare_colored_curve_image(
    image_bgr: np.ndarray,
    *,
    plot_bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Isolate saturated colored curves on a white background."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    height, width = image_bgr.shape[:2]

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    colored_curve = (sat > 40) & (val > 30) & (val < 250)
    curve_mask = colored_curve & _plot_area_mask(
        height,
        width,
        plot_bounds=plot_bounds,
    )

    if int(curve_mask.sum()) < max(500, int(height * width * 0.001)):
        return None

    prepared = np.full_like(image_bgr, 255)
    prepared[curve_mask] = (0, 0, 0)
    return prepared


def _prepare_dark_curve_image(
    image_bgr: np.ndarray,
    *,
    plot_bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """
    Isolate thin black/gray matplotlib curves and thicken them for PlotDigitizer.

    Anti-aliased matplotlib strokes sit around grayscale 50-90. Border-touching
    axis/frame components are removed via connected-component filtering.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    plot_mask = _plot_area_mask(height, width, plot_bounds=plot_bounds)

    dark = (gray < 95) & plot_mask
    if int(dark.sum()) < max(200, int(height * width * 0.0003)):
        return None

    dark_u8 = dark.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_u8)
    curve_mask = np.zeros_like(dark_u8)
    for label_index in range(1, num_labels):
        x, y, comp_width, comp_height, _area = stats[label_index]
        touches_border = (
            x <= 2
            or y <= 2
            or x + comp_width >= width - 3
            or y + comp_height >= height - 3
        )
        spans_width = comp_width > width * 0.85
        spans_height = comp_height > height * 0.85
        is_thin_h_line = comp_height <= 4 and comp_width > width * 0.4
        is_thin_v_line = comp_width <= 4 and comp_height > height * 0.4
        if touches_border and (
            spans_width or spans_height or is_thin_h_line or is_thin_v_line
        ):
            continue
        curve_mask[labels == label_index] = 255

    curve = cv2.dilate(curve_mask, np.ones((3, 3), np.uint8), iterations=2)
    if int((curve > 0).sum()) < max(150, int(height * width * 0.0002)):
        return None

    prepared = np.full_like(image_bgr, 255)
    prepared[curve > 0] = (0, 0, 0)
    return prepared


def _full_image_plot_bounds(
    calibration: AxisCalibrationResult,
    crop_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map crop-local plot frame to full-image (left, top, right, bottom) rows/cols."""
    x0, y0, _, _ = crop_bbox
    return (
        x0 + calibration.plot_left,
        y0 + calibration.plot_top,
        x0 + calibration.plot_right,
        y0 + calibration.plot_bottom,
    )


def prepare_images_for_plotdigitizer(
    image_bgr: np.ndarray,
    *,
    plot_bounds: tuple[int, int, int, int] | None = None,
) -> list[np.ndarray]:
    """Return one or more preprocessed images to try with PlotDigitizer."""
    prepared: list[np.ndarray] = []
    colored = _prepare_colored_curve_image(image_bgr, plot_bounds=plot_bounds)
    if colored is not None:
        prepared.append(colored)
    dark = _prepare_dark_curve_image(image_bgr, plot_bounds=plot_bounds)
    if dark is not None:
        prepared.append(dark)
    return prepared


def prepare_image_for_plotdigitizer(image_bgr: np.ndarray) -> np.ndarray | None:
    """Backward-compatible helper returning the first prepared candidate."""
    prepared = prepare_images_for_plotdigitizer(image_bgr)
    return prepared[0] if prepared else None


def should_remove_grid(
    image_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
) -> bool:
    plot = image_bgr[
        calibration.plot_top : calibration.plot_bottom,
        calibration.plot_left : calibration.plot_right,
    ]
    if plot.size == 0:
        return False

    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY) if plot.ndim == 3 else plot
    edges = cv2.Canny(gray, 50, 150)
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(10, gray.shape[1] // 10), 1),
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(10, gray.shape[0] // 10)),
    )
    h_lines = float(cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel).sum())
    v_lines = float(cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel).sum())
    return h_lines > gray.size * 0.05 and v_lines > gray.size * 0.02


def _plotdigitizer_executable() -> list[str]:
    if shutil.which("plotdigitizer"):
        return ["plotdigitizer"]
    return [sys.executable, "-m", "plotdigitizer.plotdigitizer"]


def _plotdigitizer_quality_acceptable(
    csv_path: Path,
    pd_calibration: PlotDigitizerCalibration,
) -> bool:
    """Reject PlotDigitizer runs that only recovered a flat baseline trace."""
    try:
        raw = np.loadtxt(csv_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, 2)
        corrected = _correct_plotdigitizer_array(raw, pd_calibration)
    except Exception:
        return False

    y_span = float(corrected[:, 1].max() - corrected[:, 1].min())
    target = float(pd_calibration.y_true_range[1] - pd_calibration.y_true_range[0])
    if target > 0 and y_span < target * 0.08:
        return False
    return len(corrected) >= 20


def build_plotdigitizer_command(
    infile: Path,
    points: PlotDigitizerPoints,
    *,
    output: Path,
    plot_file: Path | None = None,
    remove_grid: bool = False,
    invert_image: bool = False,
) -> list[str]:
    cmd = [*_plotdigitizer_executable(), str(infile)]
    for x_val, y_val in points.data_points:
        cmd.extend(["-p", f"{x_val},{y_val}"])
    for col, row in points.locations:
        cmd.extend(["-l", f"{col},{row}"])
    cmd.extend(["-o", str(output)])
    if plot_file is not None:
        cmd.extend(["--plot", str(plot_file)])
    if remove_grid:
        cmd.append("--remove-grid")
    if invert_image:
        cmd.append("--invert-image")
    return cmd


def _calibration_summary(
    calibration: AxisCalibrationResult,
    pd_calibration: PlotDigitizerCalibration | None = None,
) -> dict[str, Any]:
    summary = {
        "method": calibration.method,
        "x_min": calibration.x_min,
        "x_max": calibration.x_max,
        "y_min": calibration.y_min,
        "y_max": calibration.y_max,
        "y_method": calibration.y_method,
        "confidence": calibration.confidence,
        "plot_left": calibration.plot_left,
        "plot_right": calibration.plot_right,
        "plot_top": calibration.plot_top,
        "plot_bottom": calibration.plot_bottom,
        "warnings": calibration.warnings,
    }
    if pd_calibration is not None:
        summary["plotdigitizer"] = {
            "data_points": pd_calibration.points.data_points,
            "locations": pd_calibration.points.locations,
            "x_true_range": pd_calibration.x_true_range,
            "y_true_range": pd_calibration.y_true_range,
            "x_pd_range": pd_calibration.x_pd_range,
            "y_pd_range": pd_calibration.y_pd_range,
            "x_anchor_ticks": pd_calibration.x_anchor_ticks,
            "warnings": pd_calibration.warnings,
        }
    return summary


def _output_stem(figure_id: str, band_index: int, num_bands: int) -> str:
    base = figure_id.replace(".png", "")
    if num_bands <= 1:
        return base
    return f"{base}_curve_{band_index}"


def run_plotdigitizer(
    infile: Path,
    points: PlotDigitizerPoints,
    *,
    output: Path,
    plot_file: Path | None = None,
    remove_grid: bool = False,
    invert_image: bool = False,
) -> tuple[bool, str | None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if plot_file is not None:
        plot_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_plotdigitizer_command(
        infile,
        points,
        output=output,
        plot_file=plot_file,
        remove_grid=remove_grid,
        invert_image=invert_image,
    )
    LOGGER.debug("Running: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return False, str(exc)

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        return False, stderr or f"plotdigitizer exited with code {completed.returncode}"

    if not output.exists():
        return False, f"PlotDigitizer did not create output file: {output}"

    return True, None


def _ocr_tick_boxes_in_image(
    calibration: AxisCalibrationResult,
    *,
    image_height: int,
    image_width: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Approximate OCR tick label boxes in image coordinates (secondary text mask)."""
    boxes: list[tuple[int, int, int, int]] = []
    # X tick labels sit just below the plot frame.
    for col, _value in calibration.tick_pairs:
        x = int(col) - offset_x
        y = int(calibration.plot_bottom) - offset_y
        boxes.append((x - 12, y + 2, x + 12, min(image_height, y + 28)))
    # Y tick labels sit just left of the plot frame.
    for row, _value in calibration.y_tick_pairs:
        x = int(calibration.plot_left) - offset_x
        y = int(row) - offset_y
        boxes.append((max(0, x - 48), y - 8, max(0, x - 2), y + 8))
    clipped: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        xa = int(np.clip(min(x0, x1), 0, image_width))
        xb = int(np.clip(max(x0, x1), 0, image_width))
        ya = int(np.clip(min(y0, y1), 0, image_height))
        yb = int(np.clip(max(y0, y1), 0, image_height))
        if xb > xa and yb > ya:
            clipped.append((xa, ya, xb, yb))
    return clipped


def _save_xy_csv(csv_path: Path, data: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as handle:
        for x_val, y_val in data:
            handle.write(f"{x_val:g} {y_val:g}\n")


def _try_autonomous_band_extraction(
    *,
    extract_image: np.ndarray,
    band_calibration: AxisCalibrationResult,
    stem: str,
    output_dir: Path,
    csv_path: Path,
    plot_path: Path,
) -> BandDigitizationResult | None:
    """
    Run Lab/CC/DP extraction. Return a successful BandDigitizationResult, or
    None so the caller can fall back to PlotDigitizer.
    """
    height, width = extract_image.shape[:2]
    plot_bounds = (
        band_calibration.plot_left,
        band_calibration.plot_top,
        band_calibration.plot_right,
        band_calibration.plot_bottom,
    )
    text_boxes = _ocr_tick_boxes_in_image(
        band_calibration,
        image_height=height,
        image_width=width,
    )
    # Shift tick boxes into the inset ROI used by extract_curve_autonomous.
    inset = inset_plot_bounds(
        *plot_bounds,
        image_shape=(height, width),
    )
    left, top, _right, _bottom = inset
    roi_boxes: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in text_boxes:
        roi_boxes.append((x0 - left, y0 - top, x1 - left, y1 - top))

    try:
        extracted = extract_curve_autonomous(
            extract_image,
            plot_bounds=plot_bounds,
            text_boxes_roi=roi_boxes,
            build_debug=True,
        )
    except Exception as exc:
        LOGGER.debug("Autonomous extraction failed for %s: %s", stem, exc)
        return None

    band_warnings = list(extracted.warnings)
    band_warnings.append(f"extract_mode={extracted.mode}")
    band_warnings.append(f"extract_coverage={extracted.coverage:.2f}")

    debug_dir = output_dir / "debug"
    if extracted.debug is not None:
        write_debug_outputs(extracted.debug, debug_dir, stem)

    if not extracted.ok:
        band_warnings.append("autonomous_extract_quality_failed")
        LOGGER.info(
            "Autonomous extract below quality gate for %s; falling back to PlotDigitizer",
            stem,
        )
        return None

    y_scale, y_warnings = _estimate_y_scale(band_calibration)
    band_warnings.extend(y_warnings)
    data = map_path_to_calibrated_xy(
        extracted.rows,
        calibration=band_calibration,
        plot_bounds=extracted.plot_bounds,
        y_scale=y_scale,
    )
    if len(data) < 20:
        band_warnings.append("autonomous_extract_too_few_points")
        return None

    y_span = float(data[:, 1].max() - data[:, 1].min())
    if y_span < max(1.0, y_scale * 0.05):
        band_warnings.append("autonomous_extract_flat_trace")
        return None

    _save_xy_csv(csv_path, data)
    save_digitized_preview(csv_path, plot_path, title=stem)
    band_warnings.append(
        f"autonomous_extract_points={len(data)},"
        f"y_unique={len(np.unique(np.round(data[:, 1], 1)))}"
    )
    LOGGER.info("Autonomous extract digitized %s -> %s", stem, csv_path.name)

    return BandDigitizationResult(
        stem=stem,
        source_image=Path(),  # filled by caller
        csv_path=csv_path,
        plot_path=plot_path,
        data_points=[],
        locations=[],
        calibration=_calibration_summary(band_calibration),
        warnings=band_warnings,
        success=True,
        error=None,
    )


def digitize_figure_image(
    image_path: Path,
    output_dir: Path,
    *,
    figure_id: str | None = None,
) -> FigureDigitizationResult:
    """Calibrate, split stacked bands, and digitize one figure PNG.

    Tries autonomous Lab/CC/DP extraction first; falls back to PlotDigitizer.
    """
    figure_id = figure_id or image_path.stem
    warnings: list[str] = []

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    plot_crop = crop_plot_area(image_bgr)
    warnings.extend(plot_crop.warnings)

    calibration = calibrate_axes(plot_crop, full_image_bgr=image_bgr)
    warnings.extend(calibration.warnings)

    cropped = plot_crop.cropped_bgr
    bands = detect_figure_bands(cropped, calibration)
    num_bands = len(bands)

    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = Path(tempfile.mkdtemp(prefix=f"{figure_id}_pd_attempts_"))

    band_results: list[BandDigitizationResult] = []
    temp_paths: list[Path] = []

    for band_index, (band_top, band_bottom) in enumerate(bands, start=1):
        stem = _output_stem(figure_id, band_index, num_bands)
        band_warnings: list[str] = []
        csv_path = output_dir / f"{stem}.csv"
        plot_path = output_dir / f"{stem}_digitized.png"

        if num_bands <= 1:
            extract_image = cropped
            band_calibration = calibration
            x0, y0, _, _ = plot_crop.bbox
            plot_bounds = _full_image_plot_bounds(calibration, plot_crop.bbox)
            digitize_sources: list[tuple[Path, int, int, int, np.ndarray]] = []
            for prepared_index, prepared in enumerate(
                prepare_images_for_plotdigitizer(image_bgr, plot_bounds=plot_bounds)
            ):
                with tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False,
                    prefix=f"{stem}_prepared_{prepared_index}_",
                ) as handle:
                    prepared_path = Path(handle.name)
                temp_paths.append(prepared_path)
                cv2.imwrite(str(prepared_path), prepared)
                digitize_sources.append(
                    (prepared_path, x0, y0, image_bgr.shape[0], prepared),
                )
            digitize_sources.append(
                (image_path, x0, y0, image_bgr.shape[0], image_bgr),
            )
            grid_source = cropped
        else:
            y_start = max(0, band_top - BAND_PADDING)
            y_end = min(cropped.shape[0], band_bottom + BAND_PADDING)
            expanded_top, expanded_bottom = _expand_band_range(
                y_start,
                y_end,
                max_height=cropped.shape[0],
            )
            band_img = cropped[expanded_top:expanded_bottom, :].copy()
            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
                prefix=f"{stem}_",
            ) as handle:
                digitize_path = Path(handle.name)
            temp_paths.append(digitize_path)
            cv2.imwrite(str(digitize_path), band_img)

            band_height = band_img.shape[0]
            margin = min(BAND_PADDING, max(1, band_height // 20))
            band_calibration = replace(
                calibration,
                plot_top=margin,
                plot_bottom=max(margin + 1, band_height - margin),
                y_min=None,
                y_max=None,
                y_method="relative",
                y_tick_pairs=[],
            )
            extract_image = band_img
            grid_source = band_img
            digitize_sources = [
                (digitize_path, 0, 0, band_height, band_img),
            ]

        # Primary: autonomous Lab / connected-component / DP extraction.
        auto_result = _try_autonomous_band_extraction(
            extract_image=extract_image,
            band_calibration=band_calibration,
            stem=stem,
            output_dir=output_dir,
            csv_path=csv_path,
            plot_path=plot_path,
        )
        if auto_result is not None:
            auto_result.source_image = image_path
            band_results.append(auto_result)
            continue

        band_warnings.append("autonomous_extract_fallback_to_plotdigitizer")

        best_csv: Path | None = None
        best_score = -1.0
        best_pd_calibration: PlotDigitizerCalibration | None = None
        error: str | None = None

        for source_index, (source_path, frame_offset_x, frame_offset_y, image_height, invert_img) in enumerate(digitize_sources):
            try:
                pd_calibration = build_plotdigitizer_calibration(
                    band_calibration,
                    image_height=image_height,
                    frame_offset_x=frame_offset_x,
                    frame_offset_y=frame_offset_y,
                    full_image_bgr=image_bgr if num_bands <= 1 else None,
                    crop_bbox=plot_crop.bbox if num_bands <= 1 else None,
                )
            except ValueError as exc:
                error = str(exc)
                continue

            points = pd_calibration.points
            invert_image = should_invert_image(invert_img)
            grid_detected = should_remove_grid(grid_source, band_calibration)

            # Try without grid removal first; matplotlib/database plots often
            # trace grid lines or axis spines when --remove-grid is enabled.
            attempt_configs: list[tuple[bool, bool]] = [
                (False, invert_image),
                (True, invert_image),
            ]
            if not invert_image:
                attempt_configs.extend([(False, True), (True, True)])
            if grid_detected:
                band_warnings.append("grid_detected")

            for attempt_index, (remove_grid, invert) in enumerate(attempt_configs):
                attempt_csv = attempts_dir / f"{stem}.s{source_index}.a{attempt_index}.csv"
                attempt_ok, attempt_error = run_plotdigitizer(
                    source_path,
                    points,
                    output=attempt_csv,
                    plot_file=None,
                    remove_grid=remove_grid,
                    invert_image=invert,
                )
                if not attempt_ok:
                    error = attempt_error
                    attempt_csv.unlink(missing_ok=True)
                    continue
                score = _csv_quality_score(attempt_csv, pd_calibration)
                if score > best_score:
                    best_score = score
                    if best_csv is not None and best_csv != attempt_csv:
                        best_csv.unlink(missing_ok=True)
                    best_csv = attempt_csv
                    best_pd_calibration = pd_calibration
                    error = None
                elif attempt_csv != best_csv:
                    attempt_csv.unlink(missing_ok=True)

        if (
            best_csv is not None
            and best_pd_calibration is not None
            and not _plotdigitizer_quality_acceptable(best_csv, best_pd_calibration)
        ):
            band_warnings.append("plotdigitizer_low_quality_trace")
            best_csv.unlink(missing_ok=True)
            best_csv = None
            best_pd_calibration = None
            best_score = -1.0

        if best_pd_calibration is None and error is not None:
            band_warnings.append(error)
            band_results.append(
                BandDigitizationResult(
                    stem=stem,
                    source_image=image_path,
                    csv_path=csv_path,
                    plot_path=None,
                    data_points=[],
                    locations=[],
                    calibration=_calibration_summary(band_calibration),
                    warnings=band_warnings,
                    success=False,
                    error=error,
                )
            )
            continue

        if best_pd_calibration is not None:
            band_warnings.extend(best_pd_calibration.warnings)

        success = best_csv is not None
        if success and best_csv is not None and best_pd_calibration is not None:
            if best_csv != csv_path:
                shutil.copy2(best_csv, csv_path)
                best_csv.unlink(missing_ok=True)
            try:
                corrected = correct_plotdigitizer_csv(csv_path, best_pd_calibration)
                save_digitized_preview(csv_path, plot_path, title=stem)
                band_warnings.append(
                    f"plotdigitizer_quality_score={best_score:.0f},"
                    f"points={len(corrected)},"
                    f"y_unique={len(np.unique(np.round(corrected[:, 1], 1)))}"
                )
            except Exception as exc:
                success = False
                error = f"post_processing_failed: {exc}"
                band_warnings.append(str(exc))
        elif error is None:
            error = "plotdigitizer_failed"

        if not success:
            band_warnings.append(error or "plotdigitizer_failed")
            LOGGER.warning(
                "PlotDigitizer failed for %s band %d: %s",
                figure_id,
                band_index,
                error,
            )
        else:
            LOGGER.info("Digitized %s -> %s", figure_id, csv_path.name)

        band_results.append(
            BandDigitizationResult(
                stem=stem,
                source_image=image_path,
                csv_path=csv_path,
                plot_path=plot_path if success else None,
                data_points=(
                    best_pd_calibration.points.data_points
                    if best_pd_calibration is not None
                    else []
                ),
                locations=(
                    best_pd_calibration.points.locations
                    if best_pd_calibration is not None
                    else []
                ),
                calibration=_calibration_summary(
                    band_calibration,
                    best_pd_calibration,
                ),
                warnings=band_warnings,
                success=success,
                error=error,
            )
        )

    for temp_path in temp_paths:
        temp_path.unlink(missing_ok=True)
    shutil.rmtree(attempts_dir, ignore_errors=True)

    return FigureDigitizationResult(
        figure_id=figure_id,
        source_image=image_path,
        bands=band_results,
        warnings=sorted(set(warnings)),
    )


def resolve_figure_image(image_path_str: str, paper_dir: Path) -> Path | None:
    raw = Path(image_path_str)
    candidates = [
        raw,
        paper_dir / raw.name,
        paper_dir / "figures" / raw.name,
        REPO_ROOT / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def write_digitization_manifest(
    paper_dir: Path,
    results: list[FigureDigitizationResult],
) -> Path:
    manifest_path = paper_dir / "figures_digitized" / "digitization_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "paper_directory": str(paper_dir.resolve()),
        "figures": [
            {
                "figure_id": result.figure_id,
                "source_image": str(result.source_image),
                "warnings": result.warnings,
                "bands": [
                    {
                        "stem": band.stem,
                        "success": band.success,
                        "error": band.error,
                        "csv": str(band.csv_path),
                        "plot": str(band.plot_path) if band.plot_path else None,
                        "data_points": band.data_points,
                        "locations": band.locations,
                        "calibration": band.calibration,
                        "warnings": band.warnings,
                    }
                    for band in result.bands
                ],
            }
            for result in results
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def digitize_parsed_paper(
    paper_dir: Path,
    *,
    xrd_only: bool = True,
) -> list[FigureDigitizationResult]:
    """Digitize extracted figure PNGs for one parsed paper directory."""
    paper_dir = paper_dir.resolve()
    extra_dir = paper_dir / "extra"
    records_paths = sorted(extra_dir.glob("*.xrd_records.json"))
    if not records_paths:
        LOGGER.warning("No xrd_records.json found in %s", extra_dir)
        return []

    records = json.loads(records_paths[0].read_text(encoding="utf-8"))
    output_dir = paper_dir / "figures_digitized"
    results: list[FigureDigitizationResult] = []

    for record in records:
        figure = record.get("figure") or {}
        figure_id = figure.get("figure_id") or "unknown"
        image_paths = figure.get("image_paths") or []

        if xrd_only and not figure.get("is_likely_xrd"):
            continue
        if not image_paths:
            LOGGER.info("Skipping %s: no extracted image", figure_id)
            continue

        image_path = resolve_figure_image(image_paths[0], paper_dir)
        if image_path is None:
            LOGGER.warning(
                "Skipping %s: image not found (%s)",
                figure_id,
                image_paths[0],
            )
            continue

        try:
            result = digitize_figure_image(
                image_path,
                output_dir,
                figure_id=figure_id,
            )
        except Exception as exc:
            LOGGER.exception("Digitization failed for %s: %s", figure_id, exc)
            result = FigureDigitizationResult(
                figure_id=figure_id,
                source_image=image_path,
                bands=[
                    BandDigitizationResult(
                        stem=figure_id,
                        source_image=image_path,
                        csv_path=output_dir / f"{figure_id}.csv",
                        plot_path=None,
                        data_points=[],
                        locations=[],
                        calibration={},
                        warnings=[str(exc)],
                        success=False,
                        error=str(exc),
                    )
                ],
                warnings=[str(exc)],
            )

        results.append(result)

    if results:
        manifest_path = write_digitization_manifest(paper_dir, results)
        LOGGER.info("Wrote digitization manifest: %s", manifest_path)

    return results


def _normalize_intensity(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def pattern_index_from_stem(stem: str) -> int | None:
    match = PATTERN_STEM_RE.match(stem)
    if match is None:
        return None
    return int(match.group(1))


def save_sid_overlay(
    json_path: Path,
    csv_path: Path,
    overlay_path: Path,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot original vs digitized spectra and annotate with symmetric SID."""
    import matplotlib.pyplot as plt

    comparison = compare_spectra(json_path, csv_path)
    true_x = comparison["true_x"]
    true_y = _normalize_intensity(comparison["true_y"])
    approx_x = comparison["approx_x"]
    approx_y = _normalize_intensity(comparison["approx_y"])
    sid = comparison["symmetric_sid"]

    fig, axis = plt.subplots(figsize=(10, 4.5), dpi=150)
    axis.plot(true_x, true_y, color="#0072B2", linewidth=1.4, label="Original (JSON)")
    axis.plot(
        approx_x,
        approx_y,
        color="#D55E00",
        linewidth=1.2,
        alpha=0.9,
        label="Digitized (CSV)",
    )
    axis.set_xlabel("2θ (degrees)")
    axis.set_ylabel("Normalized intensity")
    axis.set_ylim(-0.02, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", fontsize=9)

    sid_text = (
        f"SID = {sid:.6g}\n"
        f"D(true||approx) = {comparison['forward']:.6g}\n"
        f"D(approx||true) = {comparison['reverse']:.6g}"
    )
    plot_title = title or overlay_path.stem
    axis.set_title(f"{plot_title}  |  SID = {sid:.6g}")
    axis.text(
        0.02,
        0.98,
        sid_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)
    return comparison


def digitize_png_directory(
    png_dir: Path,
    *,
    output_dir: Path = DEFAULT_CNRS_DIGITIZED_DIR,
    json_dir: Path = DEFAULT_CNRS_JSON_DIR,
    skip_existing: bool = True,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Digitize every pattern_N.png into figure_N_digitized/ with SID overlay."""
    png_dir = png_dir.resolve()
    output_dir = output_dir.resolve()
    json_dir = json_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(png_dir.glob("pattern_*.png"))
    if limit is not None:
        png_files = png_files[: max(0, limit)]

    if not png_files:
        raise FileNotFoundError(f"No pattern_*.png files found in {png_dir}")

    counts = {"succeeded": 0, "failed": 0, "skipped": 0, "total": len(png_files)}

    for index, png_path in enumerate(png_files, start=1):
        pattern_index = pattern_index_from_stem(png_path.stem)
        if pattern_index is None:
            LOGGER.warning("Skipping unexpected PNG name: %s", png_path.name)
            counts["failed"] += 1
            continue

        figure_id = f"figure_{pattern_index}"
        figure_dir = output_dir / f"{figure_id}_digitized"
        csv_path = figure_dir / f"{figure_id}.csv"
        digitized_png = figure_dir / f"{figure_id}_digitized.png"
        overlay_path = figure_dir / f"{figure_id}_overlay.png"
        original_copy = figure_dir / png_path.name
        json_path = json_dir / f"pattern_{pattern_index}.json"

        already_done = (
            csv_path.is_file()
            and digitized_png.is_file()
            and overlay_path.is_file()
            and original_copy.is_file()
        )
        if already_done and skip_existing and not overwrite:
            counts["skipped"] += 1
            print(f"[{index}/{counts['total']}] skip (exists): {figure_dir.name}")
            continue

        if not json_path.is_file():
            counts["failed"] += 1
            print(
                f"[{index}/{counts['total']}] FAILED {png_path.name}: "
                f"missing truth JSON {json_path}"
            )
            continue

        try:
            if overwrite and figure_dir.exists():
                shutil.rmtree(figure_dir)
            figure_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png_path, original_copy)

            result = digitize_figure_image(
                png_path,
                figure_dir,
                figure_id=figure_id,
            )
            if not result.bands:
                raise RuntimeError("digitize_figure_image returned no bands")

            band = result.bands[0]
            if not band.success or band.csv_path is None or not band.csv_path.is_file():
                raise RuntimeError(band.error or "digitization failed")

            if band.csv_path.resolve() != csv_path.resolve():
                shutil.copy2(band.csv_path, csv_path)

            if band.plot_path is not None and band.plot_path.is_file():
                if band.plot_path.resolve() != digitized_png.resolve():
                    shutil.copy2(band.plot_path, digitized_png)
            elif not digitized_png.is_file():
                save_digitized_preview(csv_path, digitized_png, title=figure_id)

            comparison = save_sid_overlay(
                json_path,
                csv_path,
                overlay_path,
                title=figure_id,
            )
            counts["succeeded"] += 1
            print(
                f"[{index}/{counts['total']}] ok {figure_dir.name} "
                f"SID={comparison['symmetric_sid']:.6g}"
            )
        except Exception as exc:
            counts["failed"] += 1
            LOGGER.exception("Failed digitizing %s", png_path.name)
            print(f"[{index}/{counts['total']}] FAILED {png_path.name}: {exc}")

    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run PlotDigitizer on a parsed paper directory, or on a directory "
            "of pattern_*.png files (--png-dir)."
        ),
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "Parsed paper directory (extra/ + figures/), or a PNG directory "
            "when --png-dir is set."
        ),
    )
    parser.add_argument(
        "--png-dir",
        action="store_true",
        help="Treat input_dir as a folder of pattern_*.png files to digitize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CNRS_DIGITIZED_DIR,
        help="Output root for --png-dir mode (default: data/CNRS_digitized).",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_CNRS_JSON_DIR,
        help="Ground-truth JSON directory for SID overlays (default: data/CNRS).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip figure folders that already have complete outputs (default: on).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and redo existing figure_N_digitized folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N PNGs in --png-dir mode.",
    )
    parser.add_argument(
        "--all-figures",
        action="store_true",
        help="Digitize all figures with images, not only likely XRD figures.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input_dir.exists():
        raise SystemExit(f"Directory not found: {args.input_dir}")

    if args.png_dir:
        counts = digitize_png_directory(
            args.input_dir,
            output_dir=args.output_dir,
            json_dir=args.json_dir,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        print(
            "PNG batch complete: "
            f"succeeded={counts['succeeded']}, "
            f"skipped={counts['skipped']}, "
            f"failed={counts['failed']}, "
            f"total={counts['total']} -> {args.output_dir}"
        )
        return

    results = digitize_parsed_paper(
        args.input_dir,
        xrd_only=not args.all_figures,
    )
    succeeded = sum(
        1
        for result in results
        for band in result.bands
        if band.success
    )
    total = sum(len(result.bands) for result in results)
    print(f"Digitized {succeeded}/{total} band(s) across {len(results)} figure(s).")


if __name__ == "__main__":
    main()
