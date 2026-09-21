"""Stacked multi-curve XRD digitization helpers.

Separates vertically offset traces into bands, digitizes each with continuous
path tracing (PlotDigitizer fallback) under a shared x-axis calibration, and
writes product outputs: combined JSON/CSV, a clean digitized stacked PNG, a
reconstructed overlay for validation, plus optional debug visualizations.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from plotdigitizer_pipeline import (
    BAND_PADDING,
    BandDigitizationResult,
    digitize_band_with_shared_x,
)
from xrd_digitization.calibrate_axes import calibrate_axes
from xrd_digitization.crop_plot_area import crop_plot_area
from xrd_digitization.detect_panels import detect_stacked_curve_bands
from xrd_digitization.stacked_path_trace import (
    asymmetric_search_bounds,
    build_ink_mask,
    column_trace_curve,
    detect_annotation_mask,
    detect_persistent_baselines,
    high_confidence_trace,
    pixel_path_to_calibrated_xy,
    usable_column_trace,
    viterbi_trace_curve,
)
from xrd_digitization.types import AxisCalibrationResult, CurveData

LOGGER = logging.getLogger(__name__)

IsolationMode = Literal["rect", "mask"]  # "mask" reserved for a later revision


@dataclass
class CurveLabelHint:
    text: str
    vertical_order: int


@dataclass
class CurveIsolation:
    """One isolated curve ready for single-curve digitization.

    V1 uses rectangular bands. ``mask`` is reserved so callers can later switch
    to pixel-level isolation without changing the digitize loop.
    """

    curve_id: int
    band_top: int
    band_bottom: int
    baseline_px: float
    image_bgr: np.ndarray
    crop_y0: int
    crop_y1: int
    mode: IsolationMode = "rect"
    mask: np.ndarray | None = None
    label: str | None = None


@dataclass
class BandValidation:
    status: Literal["success", "low_confidence"]
    reason: str | None = None
    detected_count: int = 0
    estimated_count: int | None = None


@dataclass
class StackedCurveRecord:
    curve_id: int
    label: str | None
    vertical_order: int
    baseline_px: float
    band_top: int
    band_bottom: int
    csv_path: str | None
    plot_path: str | None
    success: bool
    error: str | None
    warnings: list[str] = field(default_factory=list)
    xy: list[list[float]] = field(default_factory=list)
    baseline_corrected_xy: list[list[float]] = field(default_factory=list)
    raw_pixel_xy: list[list[float]] = field(default_factory=list)
    trace_method: str | None = None
    trace_quality: dict[str, Any] | None = None


@dataclass
class StackedDigitizationResult:
    figure_id: str
    plot_type: str
    multi_curve_status: Literal["success", "low_confidence"]
    reason: str | None
    source_image: Path
    cleaned_image: Path | None
    x_axis: dict[str, Any]
    curves: list[StackedCurveRecord]
    warnings: list[str] = field(default_factory=list)
    confidence: float | None = None
    stacked_json_path: Path | None = None
    stacked_csv_path: Path | None = None
    digitized_png_path: Path | None = None
    reconstructed_overlay_path: Path | None = None
    debug_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Full diagnostic payload (includes paths and internal fields)."""
        return {
            "figure_id": self.figure_id,
            "plot_type": self.plot_type,
            "multi_curve_status": self.multi_curve_status,
            "status": self.multi_curve_status,
            "reason": self.reason,
            "confidence": self.confidence,
            "source_image": str(self.source_image),
            "cleaned_image": str(self.cleaned_image) if self.cleaned_image else None,
            "x_axis": self.x_axis,
            "curves": [asdict(curve) for curve in self.curves],
            "warnings": self.warnings,
            "stacked_json_path": str(self.stacked_json_path) if self.stacked_json_path else None,
            "stacked_csv_path": str(self.stacked_csv_path) if self.stacked_csv_path else None,
            "digitized_png_path": str(self.digitized_png_path) if self.digitized_png_path else None,
            "reconstructed_overlay_path": (
                str(self.reconstructed_overlay_path) if self.reconstructed_overlay_path else None
            ),
            "debug_dir": str(self.debug_dir) if self.debug_dir else None,
        }

    def to_canonical_dict(self) -> dict[str, Any]:
        """Canonical structured output for ``{figure_id}_stacked.json``."""
        curves_out: list[dict[str, Any]] = []
        for curve in self.curves:
            quality = dict(curve.trace_quality or {})
            conf = quality.get("confidence")
            if conf is None and curve.success:
                conf = 1.0 if curve.trace_method == "continuous_path" else 0.5
            xy = curve.baseline_corrected_xy or curve.xy
            curves_out.append(
                {
                    "curve_id": curve.curve_id,
                    "label": curve.label,
                    "vertical_order": curve.vertical_order,
                    "baseline_px": curve.baseline_px,
                    "success": curve.success,
                    "trace_method": curve.trace_method,
                    "confidence": conf,
                    "quality": quality or None,
                    "xy": [[float(x), float(y)] for x, y in xy],
                    "error": curve.error,
                    "warnings": list(curve.warnings),
                }
            )
        return {
            "figure_id": self.figure_id,
            "plot_type": self.plot_type,
            "status": self.multi_curve_status,
            "confidence": self.confidence,
            "reason": self.reason,
            "x_axis": self.x_axis,
            "curves": curves_out,
            "warnings": list(self.warnings),
        }


def remove_vertical_guides(image_bgr: np.ndarray) -> np.ndarray:
    """
    Remove long near-vertical dashed/solid reference guides (V1 heuristic).

    Only targets thin, tall, low-saturation (gray) strokes so colored/black
    XRD peaks are preserved.
    """
    if image_bgr.size == 0:
        return image_bgr

    out = image_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    height, width = gray.shape

    # Gray ink only (guides), not saturated curve colors.
    grayish = (hsv[:, :, 1] < 40) & (gray < 200) & (gray > 40)
    ink = grayish.astype(np.uint8) * 255

    # Require a long vertical run that is only 1-3 px wide.
    v_len = max(40, height // 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (vertical > 0).astype(np.uint8),
        connectivity=8,
    )
    guide_mask = np.zeros((height, width), dtype=bool)
    min_h = max(50, int(height * 0.35))
    max_w = 3
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if h < min_h or w > max_w:
            continue
        if h / max(w, 1) < 15.0:
            continue
        guide_mask[labels == label_id] = True

    if not guide_mask.any():
        return out

    dilate = cv2.dilate(guide_mask.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1)
    out[dilate > 0] = (255, 255, 255)
    return out


def _peak_based_bands(
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    estimated_curve_count: int | None = None,
) -> list[tuple[int, int]]:
    """Split stacked traces using peaks in the horizontal occupancy profile."""
    plot_top = int(calibration.plot_top)
    plot_bottom = int(calibration.plot_bottom)
    plot_left = int(calibration.plot_left)
    plot_right = int(calibration.plot_right)
    sub_height = plot_bottom - plot_top
    if sub_height < 40:
        return [(plot_top, plot_bottom)]

    # Slightly tighter margins so top/bottom traces are not clipped away.
    margin_top = int(sub_height * 0.03)
    margin_bottom = int(sub_height * 0.05)
    scan_top = plot_top + margin_top
    scan_bottom = plot_bottom - margin_bottom
    if scan_bottom - scan_top < 30:
        scan_top, scan_bottom = plot_top, plot_bottom

    from xrd_digitization.detect_panels import _row_occupancy_profile

    row_counts = _row_occupancy_profile(
        cropped_bgr,
        plot_left,
        plot_right,
        scan_top,
        scan_bottom,
    )
    if row_counts.size == 0 or float(row_counts.max()) <= 0:
        return [(plot_top, plot_bottom)]

    kernel = max(5, sub_height // 40)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = np.convolve(row_counts, np.ones(kernel) / kernel, mode="same")

    try:
        from scipy.signal import find_peaks
    except ImportError:  # pragma: no cover
        find_peaks = None

    peaks: np.ndarray
    if find_peaks is not None:
        distance = max(8, len(smoothed) // max(12, (estimated_curve_count or 8) * 2))
        peaks, _ = find_peaks(
            smoothed,
            height=max(2.0, float(smoothed.max()) * 0.12),
            distance=distance,
        )
    else:
        peaks = np.array([], dtype=int)

    if estimated_curve_count is not None and estimated_curve_count >= 2 and len(peaks) > estimated_curve_count:
        order = np.argsort(smoothed[peaks])[::-1][:estimated_curve_count]
        peaks = np.sort(peaks[order])

    if len(peaks) < 2:
        return [(plot_top, plot_bottom)]

    # Convert peak centers into contiguous bands via midpoints.
    centers = [scan_top + int(p) for p in peaks]
    edges = [plot_top]
    for left, right in zip(centers, centers[1:]):
        edges.append((left + right) // 2)
    edges.append(plot_bottom)

    bands = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    bands = [(t, b) for t, b in bands if b - t >= max(6, sub_height // 40)]
    return bands if len(bands) >= 2 else [(plot_top, plot_bottom)]


def _equal_split_bands(
    calibration: AxisCalibrationResult,
    count: int,
) -> list[tuple[int, int]]:
    """Last-resort equal-height bands when image detection fails."""
    top = int(calibration.plot_top)
    bottom = int(calibration.plot_bottom)
    count = max(2, int(count))
    height = bottom - top
    if height < count * 8:
        return [(top, bottom)]
    edges = [top + int(round(i * height / count)) for i in range(count)] + [bottom]
    return [(edges[i], edges[i + 1]) for i in range(count)]


def _bands_are_comparable(bands: list[tuple[int, int]]) -> bool:
    """True when band heights look like stacked traces rather than one giant band."""
    if len(bands) < 2:
        return False
    heights = [bottom - top for top, bottom in bands]
    largest = max(heights)
    if largest <= 0:
        return False
    comparable = sum(1 for height in heights if height >= 0.35 * largest)
    return comparable >= 2


def detect_curve_bands(
    cropped_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    estimated_curve_count: int | None = None,
) -> list[tuple[int, int]]:
    """Detect horizontal bands; ``estimated_curve_count`` is a soft hint only."""
    bands = detect_stacked_curve_bands(
        cropped_bgr,
        calibration.plot_top,
        calibration.plot_bottom,
        calibration.plot_left,
        calibration.plot_right,
    )

    peak_bands = _peak_based_bands(
        cropped_bgr,
        calibration,
        estimated_curve_count=estimated_curve_count,
    )

    candidates = [
        b
        for b in (bands, peak_bands)
        if len(b) >= 2 and _bands_are_comparable(b)
    ]
    if candidates:
        if estimated_curve_count is None:
            return max(candidates, key=len)
        return min(
            candidates,
            key=lambda b: (abs(len(b) - estimated_curve_count), -len(b)),
        )

    if estimated_curve_count is not None and estimated_curve_count >= 2:
        LOGGER.info(
            "Band detector lacked comparable stacked bands; "
            "falling back to equal split using estimate=%d",
            estimated_curve_count,
        )
        return _equal_split_bands(calibration, estimated_curve_count)

    # Last resort: accept non-comparable multi bands if that is all we have.
    weak = [b for b in (bands, peak_bands) if len(b) >= 2]
    if weak:
        return max(weak, key=len)

    return bands if bands else [(int(calibration.plot_top), int(calibration.plot_bottom))]


def validate_band_separation(
    bands: list[tuple[int, int]],
    *,
    estimated_curve_count: int | None = None,
    plot_height: int | None = None,
) -> BandValidation:
    """Sanity-check detected bands before digitizing."""
    detected = len(bands)
    if detected < 2:
        return BandValidation(
            status="low_confidence",
            reason=f"detected {detected} curve band(s); need at least 2 for stacked",
            detected_count=detected,
            estimated_count=estimated_curve_count,
        )

    # Vertical order / non-overlap
    ordered = sorted(bands, key=lambda b: b[0])
    for index in range(1, len(ordered)):
        prev_top, prev_bottom = ordered[index - 1]
        top, bottom = ordered[index]
        if top < prev_top:
            return BandValidation(
                status="low_confidence",
                reason="bands are not vertically ordered",
                detected_count=detected,
                estimated_count=estimated_curve_count,
            )
        overlap = min(prev_bottom, bottom) - max(prev_top, top)
        height = max(1, min(prev_bottom - prev_top, bottom - top))
        if overlap > 0.35 * height:
            return BandValidation(
                status="low_confidence",
                reason=f"bands overlap excessively (overlap={overlap}px)",
                detected_count=detected,
                estimated_count=estimated_curve_count,
            )

    if estimated_curve_count is not None and estimated_curve_count >= 2:
        delta = abs(detected - estimated_curve_count)
        # Allow off-by-one; larger mismatches are low confidence.
        if delta > max(1, int(round(0.25 * estimated_curve_count))):
            return BandValidation(
                status="low_confidence",
                reason=(
                    f"detected {detected} curves but vision model estimated "
                    f"{estimated_curve_count}"
                ),
                detected_count=detected,
                estimated_count=estimated_curve_count,
            )

    if plot_height is not None and plot_height > 0:
        heights = [bottom - top for top, bottom in bands]
        if min(heights) < max(6, int(plot_height * 0.04)):
            return BandValidation(
                status="low_confidence",
                reason="at least one band is too thin",
                detected_count=detected,
                estimated_count=estimated_curve_count,
            )

    return BandValidation(
        status="success",
        reason=None,
        detected_count=detected,
        estimated_count=estimated_curve_count,
    )


def estimate_baseline_px(
    cropped_bgr: np.ndarray,
    band_top: int,
    band_bottom: int,
    plot_left: int,
    plot_right: int,
) -> float:
    """Estimate baseline row inside a band from foreground row occupancy."""
    sub = cropped_bgr[band_top:band_bottom, plot_left:plot_right]
    if sub.size == 0:
        return float((band_top + band_bottom) / 2.0)

    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    foreground = ((hsv[:, :, 1] > 30) & (gray < 245)) | (gray < 90)
    row_counts = foreground.sum(axis=1).astype(float)
    if row_counts.max() <= 0:
        return float(band_bottom - 1)

    # Baseline tends to sit near the densest lower portion of the band.
    lower = row_counts[len(row_counts) // 2 :]
    if lower.max() <= 0:
        local = int(np.argmax(row_counts))
    else:
        local = len(row_counts) // 2 + int(np.argmax(lower))
    return float(band_top + local)


def isolate_curve_band(
    cropped_bgr: np.ndarray,
    band_top: int,
    band_bottom: int,
    *,
    curve_id: int,
    baseline_px: float,
    mode: IsolationMode = "rect",
    padding: int = BAND_PADDING,
    label: str | None = None,
    min_height: int = 48,
) -> CurveIsolation:
    """
    Isolate one curve band.

    ``mode="rect"`` (V1): rectangular crop with padding.
    ``mode="mask"``: reserved; currently falls back to rect with a full-True mask.
    """
    height = cropped_bgr.shape[0]
    y0 = max(0, band_top - padding)
    y1 = min(height, band_bottom + padding)
    # Expand modestly for PlotDigitizer, but avoid swallowing neighboring bands.
    band_h = y1 - y0
    if band_h < min_height:
        center = (y0 + y1) // 2
        half = min_height // 2
        y0 = max(0, center - half)
        y1 = min(height, center + half)
        if y1 - y0 < min_height:
            y0 = max(0, y1 - min_height)
    band_img = cropped_bgr[y0:y1, :].copy()

    mask: np.ndarray | None = None
    if mode == "mask":
        # Placeholder for future pixel-level isolation.
        mask = np.ones(band_img.shape[:2], dtype=bool)
        mode_used: IsolationMode = "mask"
    else:
        mode_used = "rect"

    return CurveIsolation(
        curve_id=curve_id,
        band_top=band_top,
        band_bottom=band_bottom,
        baseline_px=baseline_px,
        image_bgr=band_img,
        crop_y0=y0,
        crop_y1=y1,
        mode=mode_used,
        mask=mask,
        label=label,
    )


def associate_labels_with_bands(
    bands: list[tuple[int, int]],
    baselines: list[float],
    labels: Sequence[CurveLabelHint] | Sequence[dict[str, Any]] | None,
) -> list[str | None]:
    """Map triage labels (top→bottom vertical_order) onto detected bands."""
    n = len(bands)
    assigned: list[str | None] = [None] * n
    if not labels:
        return assigned

    normalized: list[CurveLabelHint] = []
    for index, item in enumerate(labels):
        if isinstance(item, CurveLabelHint):
            normalized.append(item)
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                order = int(item.get("vertical_order", index))
            except (TypeError, ValueError):
                order = index
            normalized.append(CurveLabelHint(text=text, vertical_order=order))
        else:
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            order = int(getattr(item, "vertical_order", index))
            normalized.append(CurveLabelHint(text=text, vertical_order=order))

    if not normalized:
        return assigned

    normalized.sort(key=lambda item: item.vertical_order)
    # Prefer 1:1 by order when counts match; otherwise nearest baseline fraction.
    if len(normalized) == n:
        for index, hint in enumerate(normalized):
            assigned[index] = hint.text
        return assigned

    # Map label order fraction onto band order fraction.
    for hint in normalized:
        if len(normalized) <= 1:
            frac = 0.0
        else:
            frac = hint.vertical_order / max(1, max(h.vertical_order for h in normalized))
        target = int(round(frac * (n - 1)))
        target = max(0, min(n - 1, target))
        if assigned[target] is None:
            assigned[target] = hint.text
        else:
            # Find nearest empty slot.
            for offset in range(1, n):
                for candidate in (target - offset, target + offset):
                    if 0 <= candidate < n and assigned[candidate] is None:
                        assigned[candidate] = hint.text
                        break
                else:
                    continue
                break
    return assigned


def baseline_corrected_xy(
    xy: list[list[float]],
    *,
    intensity_is_up: bool = True,
) -> list[list[float]]:
    """
    Convert intensities so baselines sit near zero without 0–1 normalization.

    PlotDigitizer relative y already tends to put baseline near 0 and peaks
    positive; this subtracts the robust lower baseline estimate.
    """
    if not xy:
        return []
    ys = np.array([row[1] for row in xy], dtype=float)
    if intensity_is_up:
        baseline = float(np.percentile(ys, 10))
        return [[float(row[0]), float(row[1] - baseline)] for row in xy]
    baseline = float(np.percentile(ys, 90))
    return [[float(row[0]), float(baseline - row[1])] for row in xy]


def trace_curve_pixels_columnwise(
    band_bgr: np.ndarray,
    *,
    foreground_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """
    Experimental direct pixel tracer (NOT used by the default pipeline).

    For each x column, pick the foreground y nearest the previous column's y,
    producing a continuous left-to-right trace in pixel space.

    TODO: convert pixel x via shared calibration and evaluate against
    PlotDigitizer on stacked fixtures before enabling as a fallback.
    """
    height, width = band_bgr.shape[:2]
    if foreground_mask is None:
        gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        foreground_mask = ((hsv[:, :, 1] > 30) & (gray < 245)) | (gray < 90)

    points: list[tuple[int, int]] = []
    prev_y: int | None = None
    for x in range(width):
        ys = np.flatnonzero(foreground_mask[:, x])
        if ys.size == 0:
            continue
        if prev_y is None:
            y = int(np.median(ys))
        else:
            y = int(ys[np.argmin(np.abs(ys - prev_y))])
        points.append((x, y))
        prev_y = y
    return points


def save_stacked_debug(
    *,
    debug_dir: Path,
    original_bgr: np.ndarray | None,
    cleaned_bgr: np.ndarray,
    cropped_bgr: np.ndarray,
    bands: list[tuple[int, int]],
    baselines: list[float],
    isolations: list[CurveIsolation],
    digitized_xy: list[list[list[float]]],
    calibration: AxisCalibrationResult,
    coverage_profile: np.ndarray | None = None,
    annotation_mask: np.ndarray | None = None,
    pixel_paths: list[list[tuple[int, int]]] | None = None,
    search_regions: list[tuple[int, int]] | None = None,
) -> Path:
    """Write debug overlays for stacked-curve development."""
    debug_dir = Path(debug_dir)
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if original_bgr is not None:
        cv2.imwrite(str(debug_dir / "original.png"), original_bgr)
    cv2.imwrite(str(debug_dir / "cleaned.png"), cleaned_bgr)

    # baseline_candidates.png — coverage heatmap + candidate lines
    if coverage_profile is not None and coverage_profile.size == cropped_bgr.shape[0]:
        cand = cropped_bgr.copy()
        cov = coverage_profile.astype(float)
        if cov.max() > 0:
            norm = (255.0 * cov / cov.max()).astype(np.uint8)
            # Expand 1D coverage to a full-width strip for visualization.
            strip = np.repeat(norm[:, None], cropped_bgr.shape[1], axis=1)
            heat = cv2.applyColorMap(strip, cv2.COLORMAP_TURBO)
            cand = cv2.addWeighted(cand, 0.55, heat, 0.45, 0)
        for y in baselines:
            yy = int(round(y))
            cv2.line(
                cand,
                (calibration.plot_left, yy),
                (calibration.plot_right, yy),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(debug_dir / "baseline_candidates.png"), cand)

    baselines_vis = cropped_bgr.copy()
    for index, ((top, bottom), baseline) in enumerate(zip(bands, baselines)):
        color = (
            int(40 + (37 * index) % 180),
            int(80 + (53 * index) % 140),
            int(200 - (29 * index) % 160),
        )
        cv2.rectangle(
            baselines_vis,
            (calibration.plot_left, top),
            (calibration.plot_right, bottom),
            color,
            2,
        )
        y = int(round(baseline))
        cv2.line(
            baselines_vis,
            (calibration.plot_left, y),
            (calibration.plot_right, y),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            baselines_vis,
            f"c{index}",
            (calibration.plot_left + 4, max(top + 14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "detected_baselines.png"), baselines_vis)

    if annotation_mask is not None and annotation_mask.shape[:2] == cropped_bgr.shape[:2]:
        ann_vis = cropped_bgr.copy()
        ann_vis[annotation_mask] = (0, 0, 255)
        cv2.imwrite(str(debug_dir / "annotation_mask.png"), ann_vis)

    for isolation in isolations:
        cv2.imwrite(
            str(debug_dir / f"curve_{isolation.curve_id:02d}.png"),
            isolation.image_bgr,
        )

    if search_regions is not None:
        for index, (top, bottom) in enumerate(search_regions):
            region = cropped_bgr.copy()
            overlay = region.copy()
            cv2.rectangle(
                overlay,
                (calibration.plot_left, top),
                (calibration.plot_right, bottom),
                (0, 255, 0),
                -1,
            )
            region = cv2.addWeighted(region, 0.65, overlay, 0.35, 0)
            if index < len(baselines):
                y = int(round(baselines[index]))
                cv2.line(
                    region,
                    (calibration.plot_left, y),
                    (calibration.plot_right, y),
                    (0, 0, 255),
                    1,
                )
            cv2.imwrite(str(debug_dir / f"curve_{index:02d}_search_region.png"), region)

    if pixel_paths is not None:
        for index, path in enumerate(pixel_paths):
            traced = cropped_bgr.copy()
            color = (
                int(20 + (41 * index) % 200),
                int(60 + (73 * index) % 160),
                int(220 - (31 * index) % 180),
            )
            for p0, p1 in zip(path, path[1:]):
                cv2.line(traced, p0, p1, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(debug_dir / f"curve_{index:02d}_traced.png"), traced)

    overlay = render_reconstructed_overlay(
        cropped_bgr,
        calibration=calibration,
        isolations=isolations,
        digitized_xy=digitized_xy,
        pixel_paths=pixel_paths,
    )
    cv2.imwrite(str(debug_dir / "reconstructed_overlay.png"), overlay)
    return debug_dir


def _load_xy_from_csv(csv_path: Path) -> list[list[float]]:
    if not csv_path.is_file():
        return []
    try:
        data = np.loadtxt(csv_path)
    except Exception:
        return []
    if data.size == 0:
        return []
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        return []
    return [[float(row[0]), float(row[1])] for row in data]


def _x_axis_payload(calibration: AxisCalibrationResult) -> dict[str, Any]:
    return {
        "label": "2 Theta",
        "unit": "degree",
        "min": float(calibration.x_min),
        "max": float(calibration.x_max),
        "pixel_min": int(calibration.plot_left),
        "pixel_max": int(calibration.plot_right),
        "plot_top": int(calibration.plot_top),
        "plot_bottom": int(calibration.plot_bottom),
        "method": calibration.method,
        "confidence": float(calibration.confidence),
    }


def _save_curve_csv(path: Path, xy: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not xy:
        path.write_text("", encoding="utf-8")
        return
    data = np.asarray(xy, dtype=float)
    np.savetxt(path, data, delimiter=",", fmt="%.6f")


def _curve_color_bgr(index: int) -> tuple[int, int, int]:
    """Stable distinct BGR colors matching debug overlays."""
    return (
        int(20 + (41 * index) % 200),
        int(60 + (73 * index) % 160),
        int(220 - (31 * index) % 180),
    )


def overall_stacked_confidence(curves: Sequence[StackedCurveRecord]) -> float:
    """Aggregate figure-level confidence from per-curve trace quality."""
    if not curves:
        return 0.0
    confs: list[float] = []
    for curve in curves:
        quality = curve.trace_quality or {}
        if "confidence" in quality:
            confs.append(float(quality["confidence"]))
        elif curve.success:
            confs.append(0.55 if curve.trace_method == "plotdigitizer_fallback" else 0.4)
        else:
            confs.append(0.0)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    success_frac = sum(1 for c in curves if c.success) / max(1, len(curves))
    return float(max(0.0, min(1.0, 0.7 * mean_conf + 0.3 * success_frac)))


def write_stacked_combined_csv(path: Path, curves: Sequence[StackedCurveRecord]) -> Path:
    """Write all curves in long format: curve_id,label,x,y."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["curve_id,label,x,y"]
    for curve in curves:
        if not curve.success:
            continue
        xy = curve.baseline_corrected_xy or curve.xy
        label = (curve.label or "").replace(",", " ")
        for x_val, y_val in xy:
            lines.append(
                f"{curve.curve_id},{label},{float(x_val):.6f},{float(y_val):.6f}"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def render_reconstructed_overlay(
    cropped_bgr: np.ndarray,
    *,
    calibration: AxisCalibrationResult,
    isolations: Sequence[CurveIsolation],
    digitized_xy: Sequence[Sequence[Sequence[float]]],
    pixel_paths: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """Overlay digitized traces on the cropped source image (debug/validation)."""
    overlay = cropped_bgr.copy()
    if pixel_paths:
        for index, path in enumerate(pixel_paths):
            if not path:
                continue
            color = _curve_color_bgr(index)
            for p0, p1 in zip(path, path[1:]):
                cv2.line(overlay, p0, p1, color, 2, cv2.LINE_AA)
        return overlay

    x_min = float(calibration.x_min)
    x_max = float(calibration.x_max)
    plot_left = int(calibration.plot_left)
    plot_right = int(calibration.plot_right)
    span = max(1e-9, x_max - x_min)
    for index, (isolation, xy) in enumerate(zip(isolations, digitized_xy)):
        if not xy:
            continue
        color = _curve_color_bgr(index)
        ys = np.array([pt[1] for pt in xy], dtype=float)
        if ys.size == 0:
            continue
        y_lo = float(np.percentile(ys, 5))
        y_hi = float(np.percentile(ys, 99))
        amp = max(1e-6, y_hi - y_lo)
        band_h = max(1, isolation.band_bottom - isolation.band_top)
        pts: list[tuple[int, int]] = []
        for x_val, y_val in xy:
            px = int(
                round(
                    plot_left
                    + (float(x_val) - x_min) / span * (plot_right - plot_left)
                )
            )
            rel = (float(y_val) - y_lo) / amp
            py = int(round(isolation.baseline_px - rel * 0.85 * band_h))
            pts.append((px, py))
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
    return overlay


def render_digitized_stacked_png(
    *,
    calibration: AxisCalibrationResult,
    curves: Sequence[StackedCurveRecord],
    pixel_paths: Sequence[Sequence[tuple[int, int]]] | None = None,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """
    Clean digitized-only stacked plot (no source image).

    Preserves vertical baseline layout in pixel space and draws the shared x-axis.
    """
    plot_left = int(calibration.plot_left)
    plot_right = int(calibration.plot_right)
    plot_top = int(calibration.plot_top)
    plot_bottom = int(calibration.plot_bottom)
    width = int(width or max(plot_right + 40, 400))
    height = int(height or max(plot_bottom + 50, 300))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # Plot frame + baseline guides.
    cv2.rectangle(
        canvas,
        (plot_left, plot_top),
        (plot_right, plot_bottom),
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    for index, curve in enumerate(curves):
        if not curve.success:
            continue
        color = _curve_color_bgr(index)
        path: list[tuple[int, int]] = []
        if pixel_paths is not None and index < len(pixel_paths) and pixel_paths[index]:
            path = [(int(x), int(y)) for x, y in pixel_paths[index]]
        elif curve.raw_pixel_xy:
            path = [(int(round(x)), int(round(y))) for x, y in curve.raw_pixel_xy]
        else:
            xy = curve.baseline_corrected_xy or curve.xy
            span = max(1e-9, float(calibration.x_max) - float(calibration.x_min))
            for theta, intensity in xy:
                px = int(
                    round(
                        plot_left
                        + (float(theta) - float(calibration.x_min)) / span * (plot_right - plot_left)
                    )
                )
                py = int(round(float(curve.baseline_px) - float(intensity)))
                path.append((px, py))

        for p0, p1 in zip(path, path[1:]):
            if 0 <= p0[0] < width and 0 <= p0[1] < height and 0 <= p1[0] < width and 0 <= p1[1] < height:
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)

        # Soft baseline reference.
        by = int(round(curve.baseline_px))
        if plot_top <= by <= plot_bottom:
            cv2.line(
                canvas,
                (plot_left, by),
                (plot_right, by),
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        if curve.label:
            label_x = max(4, plot_left - 36)
            label_y = int(round(curve.baseline_px))
            cv2.putText(
                canvas,
                str(curve.label),
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    # Shared x-axis ticks / labels.
    x_min = float(calibration.x_min)
    x_max = float(calibration.x_max)
    span = max(1e-9, x_max - x_min)
    tick_count = 6
    axis_y = min(height - 8, plot_bottom + 18)
    cv2.line(
        canvas,
        (plot_left, plot_bottom),
        (plot_right, plot_bottom),
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    for i in range(tick_count + 1):
        frac = i / tick_count
        theta = x_min + frac * span
        px = int(round(plot_left + frac * (plot_right - plot_left)))
        cv2.line(canvas, (px, plot_bottom), (px, plot_bottom + 5), (40, 40, 40), 1)
        label = f"{theta:.0f}" if abs(theta - round(theta)) < 1e-6 else f"{theta:.1f}"
        cv2.putText(
            canvas,
            label,
            (px - 10, axis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "2theta (deg)",
        ((plot_left + plot_right) // 2 - 40, min(height - 4, axis_y + 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_stacked_product_outputs(
    result: StackedDigitizationResult,
    output_dir: Path,
    *,
    calibration: AxisCalibrationResult,
    cropped_bgr: np.ndarray,
    isolations: Sequence[CurveIsolation],
    digitized_xy: Sequence[Sequence[Sequence[float]]],
    pixel_paths: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> StackedDigitizationResult:
    """
    Write canonical stacked products (no tracing changes):

    * ``{figure_id}_stacked.json``
    * ``{figure_id}_stacked.csv``
    * ``digitized_stacked.png``
    * ``reconstructed_overlay.png``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.confidence = overall_stacked_confidence(result.curves)

    json_path = output_dir / f"{result.figure_id}_stacked.json"
    json_path.write_text(
        json.dumps(result.to_canonical_dict(), indent=2),
        encoding="utf-8",
    )
    result.stacked_json_path = json_path

    csv_path = write_stacked_combined_csv(
        output_dir / f"{result.figure_id}_stacked.csv",
        result.curves,
    )
    result.stacked_csv_path = csv_path

    overlay = render_reconstructed_overlay(
        cropped_bgr,
        calibration=calibration,
        isolations=isolations,
        digitized_xy=digitized_xy,
        pixel_paths=pixel_paths,
    )
    overlay_path = output_dir / "reconstructed_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    result.reconstructed_overlay_path = overlay_path

    clean = render_digitized_stacked_png(
        calibration=calibration,
        curves=result.curves,
        pixel_paths=pixel_paths,
        height=cropped_bgr.shape[0],
        width=cropped_bgr.shape[1],
    )
    dig_path = output_dir / "digitized_stacked.png"
    cv2.imwrite(str(dig_path), clean)
    result.digitized_png_path = dig_path
    return result


def digitize_stacked_figure(
    source_image: Path,
    output_dir: Path,
    *,
    figure_id: str | None = None,
    cleaned_image: Path | None = None,
    estimated_curve_count: int | None = None,
    curve_labels: Sequence[CurveLabelHint] | Sequence[dict[str, Any]] | None = None,
    debug: bool = False,
    isolation_mode: IsolationMode = "rect",
    remove_guides: bool = True,
    prefer_path_trace: bool = True,
    axes_sidecar_path: Path | None = None,
    prefer_colored_ink: bool = False,
    require_usable_axes: bool = True,
) -> StackedDigitizationResult:
    """
    Full stacked-curve pipeline for one figure.

    Calibrates shared X once, detects persistent baselines, traces each curve
    with a continuous-path (Viterbi) extractor, falls back to PlotDigitizer per
    band when path confidence is low, and writes product outputs:

    ``{figure_id}_stacked.json``, ``{figure_id}_stacked.csv``,
    ``digitized_stacked.png``, and ``reconstructed_overlay.png``.

    When ``axes_sidecar_path`` (or a sibling ``*.axes.json``) is available and
    usable, crop/calibration are reused from the original image instead of
    re-OCR on a cleaned PNG.
    """
    source_image = Path(source_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_id = figure_id or source_image.stem

    original_bgr = cv2.imread(str(source_image))
    if original_bgr is None:
        raise ValueError(f"Could not load image: {source_image}")

    cleaned_path = Path(cleaned_image) if cleaned_image is not None else source_image
    cleaned_bgr = cv2.imread(str(cleaned_path))
    if cleaned_bgr is None:
        raise ValueError(f"Could not load cleaned image: {cleaned_path}")

    working = cleaned_bgr.copy()
    if remove_guides:
        working = remove_vertical_guides(working)

    from xrd_digitization.axis_sidecar import AxisSidecarError, load_sidecar_for_digitize

    warnings: list[str] = []
    try:
        loaded = load_sidecar_for_digitize(
            cleaned_path if cleaned_path.is_file() else source_image,
            working,
            axes_sidecar_path=axes_sidecar_path,
            require_usable_x=True,
        )
    except AxisSidecarError as exc:
        if require_usable_axes and axes_sidecar_path is not None:
            raise RuntimeError(f"calibration_failed: {exc}") from exc
        LOGGER.warning("Axis sidecar unusable (%s); recalibrating on ORIGINAL image", exc)
        loaded = None
    if loaded is not None:
        sidecar, plot_crop = loaded
        calibration = sidecar.calibration
        warnings.extend(sidecar.warnings)
        LOGGER.info("Using axis sidecar for stacked digitize (%s)", figure_id)
    else:
        orig_crop = crop_plot_area(original_bgr)
        calibration = calibrate_axes(orig_crop, full_image_bgr=original_bgr)
        from xrd_digitization.axis_sidecar import x_calibration_is_usable

        usable, reasons = x_calibration_is_usable(calibration)
        if require_usable_axes and not usable:
            raise RuntimeError(
                "calibration_failed: "
                + ", ".join(reasons)
                + f" method={calibration.method} "
                f"x=[{calibration.x_min}, {calibration.x_max}]"
            )
        from xrd_digitization.axis_sidecar import crop_from_saved_bbox

        plot_crop = crop_from_saved_bbox(working, orig_crop.bbox)
        warnings.extend(list(orig_crop.warnings) + list(calibration.warnings))
    cropped = plot_crop.cropped_bgr

    # Optional legend region exclusion from the baseline allow-mask later.
    legend_region = None
    try:
        from xrd_digitization.legend_detect import detect_legend_bbox

        legend_region = detect_legend_bbox(cropped, calibration)
        if legend_region is not None:
            warnings.append("legend_detected")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Legend detect skipped: %s", exc)

    # --- Baseline detection (horizontal coverage preferred) ---
    coverage_profile: np.ndarray | None = None
    annotation_mask = np.zeros(cropped.shape[:2], dtype=bool)
    baseline_candidates, coverage_profile, allow_mask = detect_persistent_baselines(
        cropped,
        calibration,
        estimated_curve_count=estimated_curve_count,
    )
    if legend_region is not None:
        try:
            from xrd_digitization.legend_detect import apply_legend_mask

            allow_mask = apply_legend_mask(allow_mask, legend_region)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Legend mask skipped: %s", exc)
    annotation_mask = detect_annotation_mask(cropped, allow_mask)

    if len(baseline_candidates) >= 2:
        baselines = [float(c.y) for c in baseline_candidates]
        # Build asymmetric search regions as "bands" for bookkeeping / PD fallback.
        bands = [
            asymmetric_search_bounds(
                baselines,
                index,
                plot_top=int(calibration.plot_top),
                plot_bottom=int(calibration.plot_bottom),
            )
            for index in range(len(baselines))
        ]
        validation = validate_band_separation(
            bands,
            estimated_curve_count=estimated_curve_count,
            plot_height=calibration.plot_bottom - calibration.plot_top,
        )
        # Coverage-based baselines are trusted even if band heights differ a lot.
        if len(baselines) >= 2:
            validation = BandValidation(
                status=(
                    "success"
                    if (
                        estimated_curve_count is None
                        or abs(len(baselines) - estimated_curve_count)
                        <= max(1, int(round(0.25 * estimated_curve_count)))
                    )
                    else "low_confidence"
                ),
                reason=(
                    None
                    if estimated_curve_count is None
                    or abs(len(baselines) - estimated_curve_count)
                    <= max(1, int(round(0.25 * estimated_curve_count)))
                    else (
                        f"detected {len(baselines)} baselines but vision model "
                        f"estimated {estimated_curve_count}"
                    )
                ),
                detected_count=len(baselines),
                estimated_count=estimated_curve_count,
            )
    else:
        bands = detect_curve_bands(
            cropped,
            calibration,
            estimated_curve_count=estimated_curve_count,
        )
        validation = validate_band_separation(
            bands,
            estimated_curve_count=estimated_curve_count,
            plot_height=calibration.plot_bottom - calibration.plot_top,
        )
        baselines = [
            estimate_baseline_px(
                cropped,
                top,
                bottom,
                calibration.plot_left,
                calibration.plot_right,
            )
            for top, bottom in bands
        ]
        warnings.append("fallback_occupancy_band_detection")

    if validation.reason:
        warnings.append(validation.reason)

    labels = associate_labels_with_bands(bands, baselines, curve_labels)
    search_regions = list(bands)

    isolations = [
        isolate_curve_band(
            cropped,
            top,
            bottom,
            curve_id=index,
            baseline_px=baselines[index],
            mode=isolation_mode,
            label=labels[index],
        )
        for index, (top, bottom) in enumerate(bands)
    ]

    curve_records: list[StackedCurveRecord] = []
    digitized_xy: list[list[list[float]]] = []
    pixel_paths: list[list[tuple[int, int]]] = []
    ink = build_ink_mask(cropped, colored_only=bool(prefer_colored_ink))
    if prefer_colored_ink and not ink.any():
        warnings.append("colored_ink_empty_fallback_all_ink")
        ink = build_ink_mask(cropped, colored_only=False)

    if len(bands) < 2:
        result = StackedDigitizationResult(
            figure_id=figure_id,
            plot_type="stacked_xrd",
            multi_curve_status="low_confidence",
            reason=validation.reason or "fewer than 2 baselines detected",
            source_image=source_image,
            cleaned_image=cleaned_path if cleaned_image is not None else None,
            x_axis=_x_axis_payload(calibration),
            curves=[],
            warnings=warnings,
        )
        write_stacked_product_outputs(
            result,
            output_dir,
            calibration=calibration,
            cropped_bgr=cropped,
            isolations=isolations,
            digitized_xy=[],
            pixel_paths=[],
        )
        if debug:
            result.debug_dir = save_stacked_debug(
                debug_dir=output_dir / "debug",
                original_bgr=original_bgr,
                cleaned_bgr=cleaned_bgr,
                cropped_bgr=cropped,
                bands=bands,
                baselines=baselines,
                isolations=isolations,
                digitized_xy=[],
                calibration=calibration,
                coverage_profile=coverage_profile,
                annotation_mask=annotation_mask,
                pixel_paths=[],
                search_regions=search_regions,
            )
        return result

    x0 = int(calibration.plot_left) + max(8, int(0.045 * (calibration.plot_right - calibration.plot_left)))
    x1 = int(calibration.plot_right) - 2

    for index, isolation in enumerate(isolations):
        stem = f"{figure_id}_curve_{isolation.curve_id}"
        iso_path = output_dir / f"{stem}_band.png"
        cv2.imwrite(str(iso_path), isolation.image_bgr)

        search_top, search_bottom = search_regions[index]
        record_warnings: list[str] = []
        trace_method = "none"
        raw_pixel: list[list[float]] = []
        xy: list[list[float]] = []
        corrected: list[list[float]] = []
        pixel_path: list[tuple[int, int]] = []
        quality_dict: dict[str, Any] | None = None
        csv_path = output_dir / f"{stem}.csv"
        plot_path: Path | None = output_dir / f"{stem}_digitized.png"
        success = False
        error: str | None = None

        if prefer_path_trace:
            path, _missing, quality = viterbi_trace_curve(
                ink,
                annotation_mask,
                baseline_px=float(isolation.baseline_px),
                search_top=search_top,
                search_bottom=search_bottom,
                x0=x0,
                x1=x1,
            )
            quality_dict = quality.to_dict()
            pixel_path = path
            raw_pixel, xy, corrected = pixel_path_to_calibrated_xy(
                path,
                calibration,
                baseline_px=float(isolation.baseline_px),
            )
            if high_confidence_trace(quality) and len(xy) >= 20:
                trace_method = "continuous_path"
                success = True
                _save_curve_csv(csv_path, xy)
                # Simple preview overlay on band crop.
                preview = isolation.image_bgr.copy()
                color = (0, 0, 220)
                local_pts = [
                    (x, y - isolation.crop_y0)
                    for x, y in path
                    if isolation.crop_y0 <= y < isolation.crop_y1
                ]
                for p0, p1 in zip(local_pts, local_pts[1:]):
                    cv2.line(preview, p0, p1, color, 1, cv2.LINE_AA)
                cv2.imwrite(str(plot_path), preview)
                record_warnings.append(
                    f"path_confidence={quality.confidence:.2f},"
                    f"coverage={quality.coverage:.2f},"
                    f"gap={quality.longest_gap_px}"
                )
            else:
                record_warnings.append(
                    f"path_low_confidence:{','.join(quality.flags) or 'weak'}"
                )
                trace_method = "continuous_path_rejected"
                col_path, col_missing, col_q = column_trace_curve(
                    cropped,
                    baseline_px=float(isolation.baseline_px),
                    search_top=search_top,
                    search_bottom=search_bottom,
                    x0=x0,
                    x1=x1,
                    annotation_mask=annotation_mask,
                )
                col_raw, col_xy, col_corr = pixel_path_to_calibrated_xy(
                    col_path,
                    calibration,
                    baseline_px=float(isolation.baseline_px),
                    missing=col_missing,
                )
                n_obs = sum(1 for m in col_missing if not m)
                if usable_column_trace(col_q, n_points=n_obs) and len(col_xy) >= 40:
                    path = col_path
                    quality = col_q
                    quality_dict = col_q.to_dict()
                    pixel_path = col_path
                    raw_pixel, xy, corrected = col_raw, col_xy, col_corr
                    trace_method = "column_path"
                    success = True
                    _save_curve_csv(csv_path, xy)
                    preview = isolation.image_bgr.copy()
                    color = (0, 140, 255)
                    local_pts = [
                        (x, y - isolation.crop_y0)
                        for x, y in col_path
                        if isolation.crop_y0 <= y < isolation.crop_y1
                    ]
                    for p0, p1 in zip(local_pts, local_pts[1:]):
                        cv2.line(preview, p0, p1, color, 1, cv2.LINE_AA)
                    cv2.imwrite(str(plot_path), preview)
                    record_warnings.append(
                        f"column_confidence={col_q.confidence:.2f},"
                        f"coverage={col_q.coverage:.2f},"
                        f"obs={n_obs}"
                    )
                else:
                    record_warnings.append(
                        f"column_low_confidence:cov={col_q.coverage:.2f},obs={n_obs}"
                    )

        if not success:
            # Fallback: existing PlotDigitizer on the asymmetric search crop.
            band_img = cropped[search_top:search_bottom, :].copy()
            # Blank annotation pixels so PD is less likely to follow markers.
            local_ann = annotation_mask[search_top:search_bottom, :]
            band_img[local_ann] = (255, 255, 255)
            cv2.imwrite(str(iso_path), band_img)
            band_result = digitize_band_with_shared_x(
                band_img,
                calibration,
                output_dir=output_dir,
                stem=stem,
                source_image=source_image,
            )
            if band_result.success:
                success = True
                trace_method = "plotdigitizer_fallback"
                xy = _load_xy_from_csv(band_result.csv_path)
                corrected = baseline_corrected_xy(xy) if xy else []
                csv_path = band_result.csv_path
                plot_path = band_result.plot_path
                record_warnings.extend(band_result.warnings)
                # Approximate pixel path from calibrated xy for overlay.
                pixel_path = []
                p0 = float(calibration.plot_left)
                p1 = float(calibration.plot_right)
                span = max(1e-9, p1 - p0)
                for theta, intensity in xy:
                    x_px = int(
                        round(
                            p0
                            + (float(theta) - float(calibration.x_min))
                            / max(1e-9, float(calibration.x_max) - float(calibration.x_min))
                            * span
                        )
                    )
                    y_px = int(round(float(isolation.baseline_px) - float(intensity)))
                    pixel_path.append((x_px, y_px))
                raw_pixel = [[float(x), float(y)] for x, y in pixel_path]
            else:
                error = band_result.error or "digitization_failed"
                record_warnings.extend(band_result.warnings)
                if trace_method == "continuous_path_rejected" and xy:
                    # Keep weak path rather than nothing, marked low-confidence.
                    success = True
                    trace_method = "continuous_path_low_confidence"
                    _save_curve_csv(csv_path, xy)
                    record_warnings.append("kept_low_confidence_path")
                    error = None

        digitized_xy.append(xy)
        pixel_paths.append(pixel_path)
        curve_records.append(
            StackedCurveRecord(
                curve_id=isolation.curve_id,
                label=isolation.label,
                vertical_order=isolation.curve_id,
                baseline_px=float(isolation.baseline_px),
                band_top=int(search_top),
                band_bottom=int(search_bottom),
                csv_path=str(csv_path) if success and csv_path else None,
                plot_path=str(plot_path) if success and plot_path else None,
                success=bool(success),
                error=error,
                warnings=record_warnings,
                xy=xy,
                baseline_corrected_xy=corrected or xy,
                raw_pixel_xy=raw_pixel,
                trace_method=trace_method,
                trace_quality=quality_dict,
            )
        )

    succeeded = sum(1 for record in curve_records if record.success)
    high_ok = sum(
        1
        for record in curve_records
        if record.success and record.trace_method in {"continuous_path", "column_path"}
    )
    status = validation.status
    reason = validation.reason
    if succeeded == 0:
        status = "low_confidence"
        reason = reason or "no curves digitized successfully"
    elif succeeded < len(curve_records):
        status = "low_confidence"
        reason = reason or f"only {succeeded}/{len(curve_records)} curves digitized"
    elif high_ok < len(curve_records):
        status = "low_confidence"
        reason = reason or (
            f"{high_ok}/{len(curve_records)} high-confidence path traces "
            f"({succeeded} total successes incl. fallbacks)"
        )

    curve_data = [
        CurveData(
            two_theta=[row[0] for row in record.baseline_corrected_xy or record.xy],
            intensity=[row[1] for row in record.baseline_corrected_xy or record.xy],
            curve_id=f"curve_{record.curve_id}",
            label=record.label,
        )
        for record in curve_records
        if record.success and (record.xy or record.baseline_corrected_xy)
    ]
    if curve_data:
        from xrd_digitization.plot_digitized_curve import save_multi_column_xy

        save_multi_column_xy(curve_data, output_dir / f"{figure_id}_stacked.xy")

    result = StackedDigitizationResult(
        figure_id=figure_id,
        plot_type="stacked_xrd",
        multi_curve_status=status,
        reason=reason,
        source_image=source_image,
        cleaned_image=cleaned_path if cleaned_image is not None else None,
        x_axis=_x_axis_payload(calibration),
        curves=curve_records,
        warnings=sorted(set(warnings)),
    )
    write_stacked_product_outputs(
        result,
        output_dir,
        calibration=calibration,
        cropped_bgr=cropped,
        isolations=isolations,
        digitized_xy=digitized_xy,
        pixel_paths=pixel_paths,
    )

    if debug:
        result.debug_dir = save_stacked_debug(
            debug_dir=output_dir / "debug",
            original_bgr=original_bgr,
            cleaned_bgr=cleaned_bgr,
            cropped_bgr=cropped,
            bands=bands,
            baselines=baselines,
            isolations=isolations,
            digitized_xy=digitized_xy,
            calibration=calibration,
            coverage_profile=coverage_profile,
            annotation_mask=annotation_mask,
            pixel_paths=pixel_paths,
            search_regions=search_regions,
        )
        if result.stacked_json_path is not None:
            result.stacked_json_path.write_text(
                json.dumps(result.to_canonical_dict(), indent=2),
                encoding="utf-8",
            )

    LOGGER.info(
        "Stacked digitize %s: status=%s curves=%d succeeded=%d path_ok=%d",
        figure_id,
        status,
        len(curve_records),
        succeeded,
        high_ok,
    )
    return result
