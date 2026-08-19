"""Hybrid digitization: agent semantic guidance + selective path tracing.

The existing pixel digitizer remains the primary extractor. Hybrid features
(text cleaning, DP tracing) are enhancements for text-containing figures, not
a replacement for clean XRD plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xrd_digitization.agent_guidance import (
    AgentFigureMetadata,
    extract_agent_metadata,
    offset_text_regions_to_crop,
    run_agent_guidance,
)
from xrd_digitization.coords import (
    boxes_overlap_annotation_text,
    render_bbox_debug_overlay,
)
from xrd_digitization.digitize_xrd_curve import digitize_raw_xrd_curve
from xrd_digitization.hybrid_text_removal import create_text_mask, remove_text_preserve_curve
from xrd_digitization.simplify_curve import reconstruct_constant_width_peaks
from xrd_digitization.trace_curve import digitize_via_global_trace
from xrd_digitization.types import (
    AxisCalibrationResult,
    CurveData,
    PeakRecord,
    PlotCropResult,
)

# Source function name recorded for passthrough metadata / debugging.
ORIGINAL_DIGITIZER_SOURCE_FUNCTION = "digitize_raw_xrd_curve"

LOGGER = logging.getLogger(__name__)

DEFAULT_FUSION_THRESHOLD = 0.45
JUMP_FRAC = 0.12
CURVE_INK_THRESH = 170
MIN_PEAK_AGREEMENT_OK = 0.75
MIN_APEX_COVERAGE_OK = 0.70
MAX_PEAK_HEIGHT_ERROR_OK = 0.35
MAX_INK_ABOVE_PATH_OK = 0.25


@dataclass
class HybridDigitizationArtifacts:
    agent_meta: AgentFigureMetadata
    cleaned_bgr: np.ndarray
    text_mask: np.ndarray
    removal_mask: np.ndarray
    original_curve: CurveData
    cleaned_curve: CurveData
    agent_prior: CurveData
    fused_curve: CurveData
    dp_curve: CurveData = field(
        default_factory=lambda: CurveData(two_theta=[], intensity=[], source="dp")
    )
    hybrid_mode_used: str = "original_passthrough"
    validation: dict[str, Any] = field(default_factory=dict)
    candidate_scores: dict[str, Any] = field(default_factory=dict)
    passthrough_debug: dict[str, Any] = field(default_factory=dict)
    overlay_bgr: np.ndarray | None = None
    bbox_debug_bgr: np.ndarray | None = None
    crop_regions: list = field(default_factory=list)
    axis_mask: np.ndarray | None = None


def _empty_curve(source: str = "empty") -> CurveData:
    return CurveData(two_theta=[], intensity=[], source=source, confidence=[], warnings=[])


def hash_array(values: np.ndarray | list[float]) -> str:
    """Stable content hash for curve arrays (passthrough equality checks)."""
    arr = np.asarray(values, dtype=np.float64)
    # Round-trip through tobytes so NaNs / -0.0 are represented consistently.
    payload = arr.tobytes()
    import hashlib

    return hashlib.sha256(payload).hexdigest()[:16]


def copy_curve_arrays(curve: CurveData, *, source: str | None = None) -> CurveData:
    """Exact array copy of a curve — no resampling, smoothing, or peak rebuild."""
    return CurveData(
        two_theta=list(curve.two_theta),
        intensity=list(curve.intensity),
        curve_id=curve.curve_id,
        color=curve.color,
        label=curve.label,
        warnings=list(curve.warnings or []),
        detected_peaks=list(curve.detected_peaks or []),
        confidence=list(curve.confidence) if curve.confidence is not None else None,
        source=source if source is not None else curve.source,
        point_sources=list(curve.point_sources) if curve.point_sources is not None else None,
    )


def assert_passthrough_arrays_equal(original: CurveData, final: CurveData) -> dict[str, Any]:
    """Verify final arrays are numerically identical to the original digitizer output."""
    orig_y = np.asarray(original.intensity, dtype=np.float64)
    final_y = np.asarray(final.intensity, dtype=np.float64)
    orig_x = np.asarray(original.two_theta, dtype=np.float64)
    final_x = np.asarray(final.two_theta, dtype=np.float64)
    original_hash = hash_array(orig_y)
    final_hash = hash_array(final_y)
    arrays_equal = bool(
        orig_x.shape == final_x.shape
        and orig_y.shape == final_y.shape
        and np.allclose(final_x, orig_x, rtol=0.0, atol=1e-10, equal_nan=True)
        and np.allclose(final_y, orig_y, rtol=0.0, atol=1e-10, equal_nan=True)
    )
    if not arrays_equal:
        raise AssertionError(
            "original_passthrough arrays diverge from digitizer output: "
            f"original_hash={original_hash} final_hash={final_hash} "
            f"len_orig={len(orig_y)} len_final={len(final_y)}"
        )
    simplified = "simplified_constant_width_peaks" in (final.warnings or [])
    if simplified:
        raise AssertionError(
            "original_passthrough emitted simplified_constant_width_peaks — "
            "peak reconstruction is forbidden on the passthrough path"
        )
    return {
        "selected_candidate": "original_passthrough",
        "original_curve_hash": original_hash,
        "final_curve_hash": final_hash,
        "passthrough_arrays_equal": True,
        "final_curve_source_function": ORIGINAL_DIGITIZER_SOURCE_FUNCTION,
    }


def _intensity_to_pixel_y(
    intensity: np.ndarray,
    calibration: AxisCalibrationResult,
) -> np.ndarray:
    top, bottom = calibration.plot_top, calibration.plot_bottom
    span = max(bottom - top, 1)
    if calibration.has_y_calibration and calibration.y_min is not None and calibration.y_max is not None:
        y_span = max(calibration.y_max - calibration.y_min, 1e-9)
        frac = (intensity - calibration.y_min) / y_span
        frac = np.clip(frac, 0.0, 1.0)
        return bottom - frac * span
    max_i = float(np.max(intensity)) if intensity.size else 1.0
    max_i = max(max_i, 1e-9)
    frac = np.clip(intensity / max_i, 0.0, 1.0)
    return bottom - frac * span


def _two_theta_to_pixel_x(
    two_theta: np.ndarray,
    calibration: AxisCalibrationResult,
) -> np.ndarray:
    span_x = max(calibration.x_max - calibration.x_min, 1e-9)
    frac = (two_theta - calibration.x_min) / span_x
    return calibration.plot_left + frac * (calibration.plot_right - calibration.plot_left)


def build_agent_prior_curve(
    agent_meta: AgentFigureMetadata,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
    grid: np.ndarray | None = None,
) -> CurveData:
    """Synthetic prior from approximate curve path or peak list (fallback only)."""
    if grid is None:
        grid = np.linspace(calibration.x_min, calibration.x_max, num_points)
    else:
        grid = np.asarray(grid, dtype=float)

    if agent_meta.approximate_curve and agent_meta.approximate_curve.two_theta:
        xs = np.asarray(agent_meta.approximate_curve.two_theta, dtype=float)
        ys = np.asarray(agent_meta.approximate_curve.intensity, dtype=float)
        order = np.argsort(xs)
        intensity = np.interp(grid, xs[order], ys[order], left=ys[order][0], right=ys[order][-1])
        return CurveData(
            two_theta=grid.tolist(),
            intensity=intensity.tolist(),
            curve_id="agent_prior",
            source="agent_curve",
            confidence=[0.25] * len(grid),
            warnings=["agent_approximate_curve_prior"],
        )

    peaks = [
        PeakRecord(two_theta=float(p), relative_intensity=1.0, prominence=1.0)
        for p in agent_meta.approximate_peaks
    ]
    intensity = reconstruct_constant_width_peaks(peaks, grid)
    if intensity.size and float(np.max(intensity)) > 0:
        intensity = intensity / float(np.max(intensity))
    return CurveData(
        two_theta=grid.tolist(),
        intensity=intensity.tolist() if intensity.size else [0.0] * len(grid),
        curve_id="agent_prior",
        source="agent_peaks",
        confidence=[0.2] * len(grid),
        warnings=["agent_peak_prior"],
        detected_peaks=peaks,
    )


def _sample_darkness(
    gray: np.ndarray,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    *,
    radius: int = 1,
) -> np.ndarray:
    h, w = gray.shape
    xi = np.clip(np.round(x_pix).astype(int), 0, w - 1)
    yi = np.clip(np.round(y_pix).astype(int), 0, h - 1)
    scores = np.zeros(len(xi), dtype=np.float32)
    for i, (x, y) in enumerate(zip(xi, yi)):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        patch = gray[y0:y1, x0:x1]
        scores[i] = float(np.clip((255.0 - patch.min()) / 255.0, 0.0, 1.0))
    return scores


def digitize_with_original_digitizer(
    plot_crop: PlotCropResult,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
    text_mask: np.ndarray | None = None,
    source: str = "original",
) -> CurveData:
    """Run the sophisticated digitizer's raw mask-trace (never peak reconstruction)."""
    curve = digitize_raw_xrd_curve(
        plot_crop,
        calibration,
        num_points=num_points,
        text_mask=text_mask,
    )
    if not curve.two_theta:
        empty = _empty_curve(source)
        empty.warnings = list(curve.warnings or []) + ["original_digitizer"]
        return empty

    if "simplified_constant_width_peaks" in (curve.warnings or []):
        raise RuntimeError(
            "digitize_raw_xrd_curve returned a simplified peak reconstruction; "
            "this is a logic bug in the raw digitizer path"
        )

    curve.source = source
    warnings = list(curve.warnings or [])
    if "original_digitizer" not in warnings:
        warnings.append("original_digitizer")
    curve.warnings = warnings
    return curve


def score_curve_confidence(
    curve: CurveData,
    image_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    agent_meta: AgentFigureMetadata | None = None,
    text_mask: np.ndarray | None = None,
    agent_prior: CurveData | None = None,
    other_curve: CurveData | None = None,
    axis_mask: np.ndarray | None = None,
) -> list[float]:
    """
    Per-point confidence requiring continuity, low jumps, path agreement,
    and no axis/text-mask intersection — not merely proximity to dark pixels.
    """
    n = len(curve.two_theta)
    if n == 0:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tt = np.asarray(curve.two_theta, dtype=float)
    inten = np.asarray(curve.intensity, dtype=float)
    x_pix = _two_theta_to_pixel_x(tt, calibration)
    y_pix = _intensity_to_pixel_y(inten, calibration)
    darkness = _sample_darkness(gray, x_pix, y_pix, radius=1)

    dy = np.abs(np.diff(inten, prepend=inten[0]))
    scale = max(float(np.percentile(inten, 95) - np.percentile(inten, 5)), 1e-6)
    jump_norm = dy / scale
    continuity = np.clip(1.0 - jump_norm / max(JUMP_FRAC, 1e-6), 0.0, 1.0)
    abrupt = jump_norm > JUMP_FRAC

    pred = inten.copy()
    if n >= 3:
        pred[1:-1] = 0.5 * (inten[:-2] + inten[2:])
    local_agree = np.clip(1.0 - np.abs(inten - pred) / scale, 0.0, 1.0)

    path_agree = np.ones(n, dtype=float)
    if other_curve is not None and other_curve.two_theta:
        other_y = np.interp(
            tt,
            np.asarray(other_curve.two_theta, dtype=float),
            np.asarray(other_curve.intensity, dtype=float),
        )
        path_agree = np.clip(1.0 - np.abs(inten - other_y) / scale, 0.0, 1.0)

    prior_agree = np.ones(n, dtype=float)
    if agent_prior is not None and agent_prior.two_theta:
        prior_y = np.interp(
            tt,
            np.asarray(agent_prior.two_theta, dtype=float),
            np.asarray(agent_prior.intensity, dtype=float),
        )
        prior_scale = max(float(np.max(np.abs(prior_y))), 1e-6)
        prior_agree = np.clip(1.0 - np.abs(inten - prior_y) / prior_scale, 0.0, 1.0)
    elif agent_meta is not None and agent_meta.approximate_peaks:
        peaks = np.asarray(agent_meta.approximate_peaks, dtype=float)
        dist = np.min(np.abs(tt[:, None] - peaks[None, :]), axis=1)
        prior_agree = np.clip(1.0 - dist / 5.0, 0.35, 1.0)

    xi = np.clip(np.round(x_pix).astype(int), 0, gray.shape[1] - 1)
    yi = np.clip(np.round(y_pix).astype(int), 0, gray.shape[0] - 1)

    on_text = np.zeros(n, dtype=bool)
    if text_mask is not None and text_mask.shape[:2] == gray.shape:
        on_text = text_mask[yi, xi] > 0

    on_axis = np.zeros(n, dtype=bool)
    if axis_mask is not None and axis_mask.shape[:2] == gray.shape:
        on_axis = axis_mask[yi, xi] > 0
    on_axis = on_axis | (yi >= calibration.plot_bottom - 3)

    scores = (
        0.20 * darkness
        + 0.25 * continuity
        + 0.15 * local_agree
        + 0.20 * path_agree
        + 0.10 * prior_agree
        + 0.10 * (1.0 - on_text.astype(float))
    )
    scores = scores - 0.35 * abrupt.astype(float)
    scores = scores - 0.45 * on_text.astype(float)
    scores = scores - 0.55 * on_axis.astype(float)
    weak = (darkness < 0.35) | (continuity < 0.4)
    scores = scores - 0.25 * weak.astype(float)
    return np.clip(scores, 0.0, 1.0).tolist()


def digitize_with_confidence(
    plot_crop: PlotCropResult,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
    text_mask: np.ndarray | None = None,
    agent_meta: AgentFigureMetadata | None = None,
    agent_prior: CurveData | None = None,
    source: str = "image",
    other_curve: CurveData | None = None,
    use_dp: bool = False,
) -> tuple[CurveData, np.ndarray | None]:
    """Digitize with per-point confidence.

    By default uses the existing sophisticated digitizer. Set ``use_dp=True`` to
    run the optional DP path tracer instead.
    """
    axis_mask = None
    if use_dp:
        peaks = agent_meta.approximate_peaks if agent_meta is not None else None
        curve, _curve_y, axis_mask = digitize_via_global_trace(
            plot_crop.cropped_bgr,
            calibration,
            num_points=num_points,
            text_mask=text_mask,
            peak_two_thetas=peaks,
            prior_curve=agent_prior,
            source=source,
        )
    else:
        curve = digitize_with_original_digitizer(
            plot_crop,
            calibration,
            num_points=num_points,
            text_mask=text_mask,
            source=source,
        )

    if not curve.two_theta:
        curve.source = source
        curve.confidence = []
        return curve, axis_mask

    prior = agent_prior
    if prior is None and agent_meta is not None:
        prior = build_agent_prior_curve(
            agent_meta,
            calibration,
            grid=np.asarray(curve.two_theta, dtype=float),
        )

    curve.confidence = score_curve_confidence(
        curve,
        plot_crop.cropped_bgr,
        calibration,
        agent_meta=agent_meta,
        text_mask=text_mask,
        agent_prior=prior,
        other_curve=other_curve,
        axis_mask=axis_mask,
    )
    curve.source = source
    return curve, axis_mask


def fuse_extractions(
    cleaned: CurveData,
    original: CurveData,
    agent_prior: CurveData,
    *,
    threshold: float = DEFAULT_FUSION_THRESHOLD,
    text_mask_columns: np.ndarray | None = None,
) -> CurveData:
    """Prefer cleaned when confident; else original; else agent prior.

    Inside text-mask columns, never trust the original path (it follows glyphs).
    Kept for tests / optional candidate use — hybrid selection does not always fuse.
    """
    if cleaned.two_theta:
        grid = np.asarray(cleaned.two_theta, dtype=float)
    elif original.two_theta:
        grid = np.asarray(original.two_theta, dtype=float)
    elif agent_prior.two_theta:
        grid = np.asarray(agent_prior.two_theta, dtype=float)
    else:
        return CurveData(
            two_theta=[],
            intensity=[],
            warnings=["fusion_empty"],
            source="fused",
            confidence=[],
            point_sources=[],
        )

    def _on_grid(curve: CurveData) -> tuple[np.ndarray, np.ndarray]:
        if not curve.two_theta:
            return np.full(grid.shape, np.nan), np.zeros(grid.shape)
        xs = np.asarray(curve.two_theta, dtype=float)
        ys = np.asarray(curve.intensity, dtype=float)
        conf = (
            np.asarray(curve.confidence, dtype=float)
            if curve.confidence and len(curve.confidence) == len(xs)
            else np.zeros_like(xs)
        )
        order = np.argsort(xs)
        y_i = np.interp(grid, xs[order], ys[order], left=np.nan, right=np.nan)
        c_i = np.interp(grid, xs[order], conf[order], left=0.0, right=0.0)
        return y_i, c_i

    y_c, c_c = _on_grid(cleaned)
    y_o, c_o = _on_grid(original)
    y_a, c_a = _on_grid(agent_prior)

    in_text = np.zeros(len(grid), dtype=bool)
    if text_mask_columns is not None and len(text_mask_columns) == len(grid):
        in_text = np.asarray(text_mask_columns, dtype=bool)

    final = np.zeros_like(grid)
    conf = np.zeros_like(grid)
    sources: list[str] = []
    for i in range(len(grid)):
        if in_text[i]:
            if np.isfinite(y_c[i]):
                final[i] = y_c[i]
                conf[i] = max(float(c_c[i]), 0.35)
                sources.append("cleaned_text")
            elif np.isfinite(y_a[i]):
                final[i] = y_a[i]
                conf[i] = max(float(c_a[i]), 0.15)
                sources.append("agent")
            elif np.isfinite(y_o[i]):
                final[i] = y_o[i]
                conf[i] = float(c_o[i]) * 0.3
                sources.append("original_text_fallback")
            else:
                final[i] = 0.0
                conf[i] = 0.0
                sources.append("empty")
            continue

        if np.isfinite(y_c[i]) and c_c[i] > threshold:
            final[i] = y_c[i]
            conf[i] = float(c_c[i])
            sources.append("cleaned")
        elif np.isfinite(y_o[i]) and c_o[i] > threshold:
            final[i] = y_o[i]
            conf[i] = float(c_o[i])
            sources.append("original")
        elif np.isfinite(y_c[i]) and np.isfinite(y_o[i]):
            final[i] = 0.5 * (y_c[i] + y_o[i])
            conf[i] = 0.5 * (float(c_c[i]) + float(c_o[i]))
            sources.append("averaged")
        elif np.isfinite(y_c[i]):
            final[i] = y_c[i]
            conf[i] = float(c_c[i])
            sources.append("cleaned_low")
        elif np.isfinite(y_o[i]):
            final[i] = y_o[i]
            conf[i] = float(c_o[i])
            sources.append("original_low")
        elif np.isfinite(y_a[i]):
            final[i] = y_a[i]
            conf[i] = max(float(c_a[i]), 0.15)
            sources.append("agent")
        else:
            final[i] = 0.0
            conf[i] = 0.0
            sources.append("empty")

    warnings = sorted(
        set(
            (cleaned.warnings or [])
            + (original.warnings or [])
            + (agent_prior.warnings or [])
            + ["hybrid_fused"]
        )
    )
    return CurveData(
        two_theta=grid.tolist(),
        intensity=final.tolist(),
        curve_id="curve_1",
        warnings=warnings,
        confidence=conf.tolist(),
        source="fused",
        point_sources=sources,
    )


def _find_curve_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:
    if len(intensity) < 5:
        return np.array([], dtype=float)
    from scipy.signal import find_peaks

    y_s = intensity.copy()
    if len(y_s) > 15:
        window = min(11, len(y_s) // 40 * 2 + 1)
        if window >= 5:
            y_s = np.convolve(y_s, np.ones(window) / window, mode="same")
    prominence = max(0.05 * float(np.max(y_s)), 1e-6)
    peaks_idx, _ = find_peaks(
        y_s, prominence=prominence, distance=max(3, len(intensity) // 80)
    )
    return two_theta[peaks_idx] if peaks_idx.size else np.array([], dtype=float)


def detect_artificial_plateaus(
    curve: CurveData,
    image_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    agent_peaks: list[float] | None = None,
) -> dict[str, Any]:
    """Flag long flat regions with abrupt ends where the image has a narrow peak."""
    result = {
        "count": 0,
        "rectangular_transitions": 0,
        "locations": [],
    }
    if not curve.two_theta:
        return result

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tt = np.asarray(curve.two_theta, dtype=float)
    inten = np.asarray(curve.intensity, dtype=float)
    if len(inten) < 10:
        return result

    scale = max(float(np.percentile(inten, 95) - np.percentile(inten, 5)), 1e-6)
    peak_level = max(float(np.max(inten)), 1e-6)
    y_pix = _intensity_to_pixel_y(inten, calibration)
    x_pix = _two_theta_to_pixel_x(tt, calibration)

    d1 = np.abs(np.diff(inten))
    flat = d1 < (0.003 * scale)
    # Abrupt vertical transitions in intensity space.
    abrupt = (np.abs(np.diff(inten)) / scale) > 0.12

    i = 0
    n = len(flat)
    while i < n:
        if not flat[i] or inten[i] < 0.12 * peak_level:
            i += 1
            continue
        j = i
        while j < n and flat[j] and inten[min(j, len(inten) - 1)] > 0.12 * peak_level:
            j += 1
        run_len = j - i
        # ~1° wide or longer on a typical 70° scan with 2000 points ≈ 28 samples.
        min_run = max(18, len(inten) // 90)
        if run_len >= min_run:
            left_abrupt = i > 0 and abrupt[i - 1]
            right_abrupt = j < len(abrupt) and abrupt[j]
            # Image evidence: tall narrow ink above the plateau path.
            xs = np.clip(np.round(x_pix[i : j + 1]).astype(int), 0, gray.shape[1] - 1)
            ys = np.clip(np.round(y_pix[i : j + 1]).astype(int), 0, gray.shape[0] - 1)
            ink_above = 0
            tall_col = 0
            for x, y in zip(xs, ys):
                top = calibration.plot_top + 2
                bot = max(top + 1, y - 2)
                col = gray[top:bot, x]
                if col.size == 0:
                    continue
                dark = col < CURVE_INK_THRESH
                if not np.any(dark):
                    continue
                tip = int(np.where(dark)[0][0]) + top
                rise = y - tip
                if rise > 25:
                    ink_above += 1
                    if rise > 60:
                        tall_col += 1
            frac_above = ink_above / max(len(xs), 1)
            near_agent_peak = False
            if agent_peaks:
                mid_tt = float(np.mean(tt[i : j + 1]))
                near_agent_peak = any(abs(mid_tt - p) < 2.0 for p in agent_peaks)

            # Require image evidence of a taller narrow peak above the flat path.
            # Flat baseline segments and genuine broad peaks must not trigger this.
            is_artificial = (
                frac_above > 0.25
                and tall_col >= 2
                and (
                    (left_abrupt and right_abrupt)
                    or (near_agent_peak and run_len >= min_run)
                )
            )
            if is_artificial:
                result["count"] += 1
                result["locations"].append(
                    {
                        "two_theta_min": float(tt[i]),
                        "two_theta_max": float(tt[min(j, len(tt) - 1)]),
                        "ink_above_fraction": float(frac_above),
                    }
                )
            if left_abrupt and right_abrupt and frac_above > 0.25 and tall_col >= 1:
                result["rectangular_transitions"] += 1
        i = max(j, i + 1)
    return result


def _image_peak_apex_metrics(
    gray: np.ndarray,
    curve: CurveData,
    calibration: AxisCalibrationResult,
    peak_tt: float,
    *,
    window: float = 1.5,
) -> dict[str, float]:
    """Compare extracted path height to connected dark apex near ``peak_tt``."""
    tt = np.asarray(curve.two_theta, dtype=float)
    inten = np.asarray(curve.intensity, dtype=float)
    mask = (tt >= peak_tt - window) & (tt <= peak_tt + window)
    if not np.any(mask):
        return {
            "path_height": 0.0,
            "image_apex_height": 0.0,
            "apex_coverage": 0.0,
            "ink_above_fraction": 0.0,
            "height_error": 1.0,
        }

    path_height = float(np.max(inten[mask]))
    x_pix = _two_theta_to_pixel_x(tt[mask], calibration)
    y_pix = _intensity_to_pixel_y(inten[mask], calibration)
    xs = np.clip(np.round(x_pix).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip(np.round(y_pix).astype(int), 0, gray.shape[0] - 1)

    apex_ys: list[float] = []
    above = 0
    for x, y in zip(xs, ys):
        top = calibration.plot_top + 2
        bot = min(calibration.plot_bottom - 2, max(y + 6, top + 8))
        col = gray[top:bot, x]
        dark = col < CURVE_INK_THRESH
        if not np.any(dark):
            continue
        # Highest connected dark run tip (smallest image y).
        tip_local = int(np.where(dark)[0][0])
        tip = top + tip_local
        apex_ys.append(float(tip))
        if tip < y - 8:
            above += 1

    if apex_ys:
        apex_y = float(np.min(apex_ys))
        apex_inten = _pixel_y_to_intensity(apex_y, calibration)
    else:
        apex_inten = path_height

    height_denom = max(apex_inten, path_height, 1e-6)
    height_error = max(0.0, (apex_inten - path_height) / height_denom)
    coverage = 1.0 - height_error
    return {
        "path_height": path_height,
        "image_apex_height": float(apex_inten),
        "apex_coverage": float(np.clip(coverage, 0.0, 1.0)),
        "ink_above_fraction": float(above / max(len(xs), 1)),
        "height_error": float(height_error),
    }


def _pixel_y_to_intensity(y: float, calibration: AxisCalibrationResult) -> float:
    top, bottom = calibration.plot_top, calibration.plot_bottom
    span = max(bottom - top, 1)
    frac = (bottom - y) / span
    if calibration.has_y_calibration and calibration.y_min is not None and calibration.y_max is not None:
        return float(calibration.y_min + frac * (calibration.y_max - calibration.y_min))
    return float(max(frac, 0.0) * 100.0)


def score_candidate_curve(
    curve: CurveData,
    image_bgr: np.ndarray,
    agent_meta: AgentFigureMetadata,
    calibration: AxisCalibrationResult,
    *,
    axis_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Rich candidate score — ink distance alone is not enough."""
    report = validate_extraction(
        image_bgr,
        curve,
        agent_meta,
        calibration,
        axis_mask=axis_mask,
    )
    # Scalar utility for ranking (higher is better).
    peak_agr = float(report.get("peak_agreement") or 0.0)
    apex = float(report.get("mean_apex_coverage") or 0.0)
    height_err = float(report.get("mean_peak_height_error") or 1.0)
    ink_above = float(report.get("mean_ink_above_fraction") or 1.0)
    plateaus = int(report.get("artificial_plateau_count") or 0)
    rect = int(report.get("rectangular_transitions") or 0)
    ink_dist = float(report.get("mean_ink_distance_px") or 10.0)
    baseline = float(report.get("baseline_agreement") or 0.0)

    score = (
        3.5 * peak_agr
        + 2.2 * apex
        + 1.0 * baseline
        + 0.6 * max(0.0, 1.0 - ink_dist / 4.0)
        - 1.6 * height_err
        - 1.2 * ink_above
        - 1.0 * min(plateaus, 3)
        - 0.6 * min(rect, 3)
    )
    if report.get("missed_tall_peak") and peak_agr < 0.75:
        score -= 2.5
    if not curve.two_theta:
        score = -1e9
    report["selection_score"] = float(score)
    return report


def validate_extraction(
    image_bgr: np.ndarray,
    curve: CurveData,
    agent_meta: AgentFigureMetadata,
    calibration: AxisCalibrationResult,
    *,
    threshold: float = DEFAULT_FUSION_THRESHOLD,
    axis_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate curve against ink, peaks, apex coverage, and artificial plateaus."""
    result: dict[str, Any] = {
        "ok": True,
        "mean_ink_distance_px": None,
        "jump_count": 0,
        "gap_count": 0,
        "peak_agreement": None,
        "peak_height_agreement": None,
        "mean_apex_coverage": None,
        "mean_peak_height_error": None,
        "mean_ink_above_fraction": None,
        "baseline_agreement": None,
        "artificial_plateau_count": 0,
        "rectangular_transitions": 0,
        "missed_tall_peak": False,
        "uncertain_fraction": 0.0,
        "uncertain_segments": [],
        "axis_intersection_fraction": 0.0,
        "flags": [],
    }
    if not curve.two_theta:
        result["ok"] = False
        result["flags"].append("empty_curve")
        return result

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tt = np.asarray(curve.two_theta, dtype=float)
    inten = np.asarray(curve.intensity, dtype=float)
    x_pix = _two_theta_to_pixel_x(tt, calibration)
    y_pix = _intensity_to_pixel_y(inten, calibration)

    dark = (gray < CURVE_INK_THRESH).astype(np.uint8) * 255
    dist = cv2.distanceTransform(255 - dark, cv2.DIST_L2, 3)
    xi = np.clip(np.round(x_pix).astype(int), 0, gray.shape[1] - 1)
    yi = np.clip(np.round(y_pix).astype(int), 0, gray.shape[0] - 1)
    ink_dist = dist[yi, xi]
    result["mean_ink_distance_px"] = float(np.mean(ink_dist))
    if result["mean_ink_distance_px"] > 4.0:
        result["flags"].append("far_from_dark_pixels")
        result["ok"] = False

    scale = max(float(np.percentile(inten, 95) - np.percentile(inten, 5)), 1e-6)
    jumps = np.where(np.abs(np.diff(inten)) / scale > JUMP_FRAC)[0]
    result["jump_count"] = int(jumps.size)
    if jumps.size > max(3, len(inten) // 50):
        result["flags"].append("sudden_jumps")

    near_axis = yi >= (calibration.plot_bottom - 4)
    result["axis_intersection_fraction"] = float(np.mean(near_axis))
    if result["axis_intersection_fraction"] > 0.08:
        result["flags"].append("axis_tracking")
        result["ok"] = False
    if axis_mask is not None and axis_mask.shape[:2] == gray.shape:
        on_axis = axis_mask[yi, xi] > 0
        if float(np.mean(on_axis)) > 0.05:
            result["flags"].append("axis_mask_intersection")
            result["ok"] = False

    plateau_info = detect_artificial_plateaus(
        curve,
        image_bgr,
        calibration,
        agent_peaks=list(agent_meta.approximate_peaks or []),
    )
    result["artificial_plateau_count"] = int(plateau_info["count"])
    result["rectangular_transitions"] = int(plateau_info["rectangular_transitions"])
    result["plateau_locations"] = plateau_info["locations"]
    if plateau_info["count"] >= 1:
        result["flags"].append("rectangular_plateaus")
        result["ok"] = False

    # Baseline agreement outside peaks: path should stay near lower-envelope ink.
    if agent_meta.approximate_peaks:
        peaks = np.asarray(agent_meta.approximate_peaks, dtype=float)
        outside = np.min(np.abs(tt[:, None] - peaks[None, :]), axis=1) > 2.0
    else:
        outside = np.ones(len(tt), dtype=bool)
    if np.any(outside):
        base_dist = ink_dist[outside]
        result["baseline_agreement"] = float(
            np.clip(1.0 - np.mean(base_dist) / 3.0, 0.0, 1.0)
        )
    else:
        result["baseline_agreement"] = 1.0

    conf = (
        np.asarray(curve.confidence, dtype=float)
        if curve.confidence and len(curve.confidence) == len(tt)
        else np.ones(len(tt))
    )
    uncertain = conf < threshold
    result["uncertain_fraction"] = float(np.mean(uncertain)) if len(uncertain) else 0.0
    segments: list[dict[str, float]] = []
    if np.any(uncertain):
        idx = np.where(uncertain)[0]
        start = int(idx[0])
        prev = start
        for i in idx[1:]:
            if int(i) == prev + 1:
                prev = int(i)
                continue
            segments.append(
                {
                    "two_theta_min": float(tt[start]),
                    "two_theta_max": float(tt[prev]),
                }
            )
            start = int(i)
            prev = start
        segments.append(
            {"two_theta_min": float(tt[start]), "two_theta_max": float(tt[prev])}
        )
    result["uncertain_segments"] = segments
    if result["uncertain_fraction"] > 0.25:
        result["flags"].append("high_uncertainty")
        result["ok"] = False

    if agent_meta.approximate_peaks:
        extracted = _find_curve_peaks(tt, inten)
        matches = 0
        apex_coverages: list[float] = []
        height_errors: list[float] = []
        ink_above_fracs: list[float] = []
        height_agreements: list[float] = []
        missed_tall = False
        for ap in agent_meta.approximate_peaks:
            if extracted.size and float(np.min(np.abs(extracted - ap))) <= 1.5:
                matches += 1
            metrics = _image_peak_apex_metrics(
                gray, curve, calibration, float(ap), window=1.5
            )
            apex_coverages.append(metrics["apex_coverage"])
            height_errors.append(metrics["height_error"])
            ink_above_fracs.append(metrics["ink_above_fraction"])
            height_agreements.append(1.0 - metrics["height_error"])
            # Tall narrow image peak while path stays near its base.
            if (
                metrics["image_apex_height"] > 0.35 * max(float(np.max(inten)), 1e-6)
                and metrics["height_error"] > 0.45
                and metrics["ink_above_fraction"] > 0.3
            ):
                missed_tall = True

        agreement = matches / max(len(agent_meta.approximate_peaks), 1)
        result["peak_agreement"] = float(agreement)
        result["extracted_peak_count"] = int(extracted.size)
        result["mean_apex_coverage"] = float(np.mean(apex_coverages))
        result["mean_peak_height_error"] = float(np.mean(height_errors))
        result["mean_ink_above_fraction"] = float(np.mean(ink_above_fracs))
        result["peak_height_agreement"] = float(np.mean(height_agreements))
        result["missed_tall_peak"] = bool(missed_tall)

        if agreement < MIN_PEAK_AGREEMENT_OK:
            result["flags"].append("peak_mismatch")
            result["ok"] = False
        if result["mean_apex_coverage"] < MIN_APEX_COVERAGE_OK:
            result["flags"].append("low_apex_coverage")
            result["ok"] = False
        if result["mean_peak_height_error"] > MAX_PEAK_HEIGHT_ERROR_OK:
            result["flags"].append("peak_height_underestimate")
            result["ok"] = False
        if result["mean_ink_above_fraction"] > MAX_INK_ABOVE_PATH_OK:
            result["flags"].append("ink_above_path_near_peaks")
            result["ok"] = False
        if missed_tall:
            result["flags"].append("missed_tall_narrow_peak")
            result["ok"] = False

    return result


def select_best_candidate(
    candidates: dict[str, CurveData],
    image_bgr: np.ndarray,
    agent_meta: AgentFigureMetadata,
    calibration: AxisCalibrationResult,
    *,
    axis_mask: np.ndarray | None = None,
    prefer_order: tuple[str, ...] = ("original", "cleaned", "dp"),
) -> tuple[str, CurveData, dict[str, Any]]:
    """Pick a whole candidate via validation scores — do not always fuse."""
    scores: dict[str, Any] = {}
    for name in prefer_order:
        curve = candidates.get(name)
        if curve is None or not curve.two_theta:
            continue
        scores[name] = score_candidate_curve(
            curve, image_bgr, agent_meta, calibration, axis_mask=axis_mask
        )

    if not scores:
        return "empty", _empty_curve(), scores

    def _rank(name: str) -> tuple[float, float, float]:
        report = scores[name]
        peak = float(report.get("peak_agreement") or 0.0)
        apex = float(report.get("mean_apex_coverage") or 0.0)
        score = float(report.get("selection_score") or -1e9)
        # Prefer original/cleaned when quality is comparable.
        if name == "original":
            score += 0.15
        elif name == "cleaned":
            score += 0.10
        # Peak agreement dominates — a text-corrupted original must not beat
        # a DP path that recovers substantially more major peaks.
        return (peak, apex, score)

    best_name = max(scores.keys(), key=_rank)
    best_curve = candidates[best_name]

    # DP must be demonstrably better on peaks/apex to override original/cleaned.
    if best_name == "dp":
        dp_peak = float(scores["dp"].get("peak_agreement") or 0.0)
        dp_apex = float(scores["dp"].get("mean_apex_coverage") or 0.0)
        alt_names = [n for n in ("cleaned", "original") if n in scores]
        if alt_names:
            alt = max(alt_names, key=_rank)
            alt_peak = float(scores[alt].get("peak_agreement") or 0.0)
            alt_apex = float(scores[alt].get("mean_apex_coverage") or 0.0)
            alt_score = float(scores[alt].get("selection_score") or -1e9)
            dp_score = float(scores["dp"].get("selection_score") or -1e9)
            clearly_better_peaks = (dp_peak >= alt_peak + 0.2) or (
                dp_peak >= 0.75 and alt_peak < 0.6
            )
            clearly_better_apex = dp_apex >= alt_apex + 0.1
            if not (clearly_better_peaks or clearly_better_apex or dp_score >= alt_score + 0.75):
                return alt, candidates[alt], scores

    # If a non-DP winner has poor peak agreement and DP is much better, take DP.
    if best_name in {"original", "cleaned"} and "dp" in scores:
        best_peak = float(scores[best_name].get("peak_agreement") or 0.0)
        dp_peak = float(scores["dp"].get("peak_agreement") or 0.0)
        dp_apex = float(scores["dp"].get("mean_apex_coverage") or 0.0)
        if dp_peak >= 0.75 and (dp_peak >= best_peak + 0.2 or best_peak < 0.6):
            if dp_apex >= 0.55:
                return "dp", candidates["dp"], scores

    return best_name, best_curve, scores


def _extraction_corrupted_by_text(
    validation: dict[str, Any],
    *,
    text_mask: np.ndarray | None,
) -> bool:
    if text_mask is None or not np.any(text_mask):
        return False
    flags = set(validation.get("flags") or [])
    corruption_flags = {
        "peak_mismatch",
        "low_apex_coverage",
        "peak_height_underestimate",
        "missed_tall_narrow_peak",
        "ink_above_path_near_peaks",
        "axis_tracking",
    }
    if flags & corruption_flags:
        return True
    if float(validation.get("peak_agreement") or 1.0) < 0.6:
        return True
    return False


def render_hybrid_overlay(
    image_bgr: np.ndarray,
    curve: CurveData,
    calibration: AxisCalibrationResult,
    agent_meta: AgentFigureMetadata,
    *,
    text_mask: np.ndarray | None = None,
    threshold: float = DEFAULT_FUSION_THRESHOLD,
) -> np.ndarray:
    """Overlay selected curve, low-confidence segments, and agent text boxes."""
    overlay = image_bgr.copy()
    if text_mask is not None and np.any(text_mask):
        tint = overlay.copy()
        tint[text_mask > 0] = (40, 40, 200)
        overlay = cv2.addWeighted(overlay, 0.75, tint, 0.25, 0)

    for region in agent_meta.text_regions:
        x1, y1, x2, y2 = [int(v) for v in region.bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 1)

    if not curve.two_theta:
        return overlay

    tt = np.asarray(curve.two_theta, dtype=float)
    inten = np.asarray(curve.intensity, dtype=float)
    x_pix = _two_theta_to_pixel_x(tt, calibration)
    y_pix = _intensity_to_pixel_y(inten, calibration)
    conf = (
        np.asarray(curve.confidence, dtype=float)
        if curve.confidence and len(curve.confidence) == len(tt)
        else np.ones(len(tt))
    )

    pts_lo: list[tuple[int, int]] = []
    for x, y, c in zip(x_pix, y_pix, conf):
        pt = (int(round(x)), int(round(y)))
        if c < threshold:
            pts_lo.append(pt)

    all_pts = np.column_stack(
        [np.round(x_pix).astype(int), np.round(y_pix).astype(int)]
    )
    if len(all_pts) >= 2:
        cv2.polylines(
            overlay,
            [all_pts.reshape(-1, 1, 2)],
            False,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    for pt in pts_lo:
        cv2.circle(overlay, pt, 2, (0, 0, 255), -1, cv2.LINE_AA)
    return overlay


def run_hybrid_digitization(
    image_bgr: np.ndarray,
    plot_crop: PlotCropResult,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
    agent_meta: AgentFigureMetadata | None = None,
    agent_metadata_path: Path | None = None,
    fusion_threshold: float = DEFAULT_FUSION_THRESHOLD,
    http_post: Any | None = None,
    require_box_overlap: bool = True,
    force_dp_tracing: bool = False,
) -> HybridDigitizationArtifacts:
    """Hybrid pipeline with original-digitizer primacy.

    Pipeline:
      original image → optional text removal → existing sophisticated digitizer

    DP tracing is an optional fallback only when text clearly corrupts extraction,
    or when ``force_dp_tracing`` is set.
    """
    if agent_meta is None:
        agent_meta = run_agent_guidance(
            image_bgr,
            metadata_path=agent_metadata_path,
            http_post=http_post,
        )
    else:
        agent_meta = extract_agent_metadata(
            agent_meta, image_shape=image_bgr.shape[:2]
        )

    actual_h, actual_w = image_bgr.shape[:2]
    crop_h, crop_w = plot_crop.cropped_bgr.shape[:2]
    crop_regions = offset_text_regions_to_crop(
        agent_meta,
        plot_crop.bbox,
        target_image_size=(crop_w, crop_h),
        actual_image_size=(actual_w, actual_h),
    )

    bbox_debug = render_bbox_debug_overlay(
        plot_crop.cropped_bgr,
        [r.bbox for r in crop_regions],
        labels=[r.text or r.type for r in crop_regions],
    )

    overlap_ok, overlap_stats = boxes_overlap_annotation_text(
        plot_crop.cropped_bgr,
        [r.bbox for r in crop_regions],
    )
    LOGGER.info(
        "Transformed text boxes overlap check: ok=%s stats=%s",
        overlap_ok,
        {k: overlap_stats[k] for k in overlap_stats if k != "fractions"},
    )

    text_mask = create_text_mask(plot_crop.cropped_bgr.shape[:2], crop_regions)
    has_text_regions = bool(agent_meta.text_regions)

    agent_prior = build_agent_prior_curve(
        agent_meta, calibration, num_points=num_points
    )

    # ------------------------------------------------------------------
    # 1) No-text passthrough: keep the original sophisticated digitizer.
    # ------------------------------------------------------------------
    if not has_text_regions and not force_dp_tracing:
        LOGGER.info(
            "No text regions — original_passthrough via %s (no simplify/peak rebuild)",
            ORIGINAL_DIGITIZER_SOURCE_FUNCTION,
        )
        # Exact digitizer arrays — never simplify_single_curve / reconstruct_from_peaks.
        original_curve = digitize_with_original_digitizer(
            plot_crop,
            calibration,
            num_points=num_points,
            text_mask=None,
            source="original",
        )
        if original_curve.two_theta:
            original_curve.confidence = score_curve_confidence(
                original_curve,
                plot_crop.cropped_bgr,
                calibration,
                agent_meta=agent_meta,
                text_mask=None,
                agent_prior=agent_prior,
                axis_mask=None,
            )

        cleaned_bgr = plot_crop.cropped_bgr.copy()
        removal_mask = np.zeros(plot_crop.cropped_bgr.shape[:2], dtype=np.uint8)
        text_mask = np.zeros_like(removal_mask)
        cleaned_curve = copy_curve_arrays(original_curve, source="cleaned")
        dp_curve = _empty_curve("dp")

        # final_x/y must be exact copies of the digitizer arrays.
        final_curve = copy_curve_arrays(original_curve, source="original_passthrough")
        hybrid_mode_used = "original_passthrough"
        passthrough_debug = assert_passthrough_arrays_equal(original_curve, final_curve)

        validation = validate_extraction(
            plot_crop.cropped_bgr,
            final_curve,
            agent_meta,
            calibration,
            threshold=fusion_threshold,
            axis_mask=None,
        )
        validation["bbox_overlap"] = overlap_stats
        validation["hybrid_mode_used"] = hybrid_mode_used
        validation.update(passthrough_debug)
        candidate_scores = {
            "original": score_candidate_curve(
                original_curve,
                plot_crop.cropped_bgr,
                agent_meta,
                calibration,
                axis_mask=None,
            )
        }

        return _finalize_artifacts(
            image_bgr=image_bgr,
            plot_crop=plot_crop,
            calibration=calibration,
            agent_meta=agent_meta,
            crop_regions=crop_regions,
            cleaned_bgr=cleaned_bgr,
            text_mask=text_mask,
            removal_mask=removal_mask,
            original_curve=original_curve,
            cleaned_curve=cleaned_curve,
            dp_curve=dp_curve,
            agent_prior=agent_prior,
            final_curve=final_curve,
            hybrid_mode_used=hybrid_mode_used,
            validation=validation,
            candidate_scores=candidate_scores,
            passthrough_debug=passthrough_debug,
            bbox_debug=bbox_debug,
            axis_mask=None,
            fusion_threshold=fusion_threshold,
        )

    # ------------------------------------------------------------------
    # 2) Text figures: clean, then digitize with the original digitizer.
    # ------------------------------------------------------------------
    if require_box_overlap and not overlap_ok:
        LOGGER.warning(
            "Skipping text removal: transformed boxes do not overlap annotation ink"
        )
        cleaned_bgr = plot_crop.cropped_bgr.copy()
        removal_mask = np.zeros(plot_crop.cropped_bgr.shape[:2], dtype=np.uint8)
        text_mask = np.zeros_like(text_mask)
    else:
        cleaned_bgr, removal_mask = remove_text_preserve_curve(
            plot_crop.cropped_bgr,
            text_mask,
            text_regions=crop_regions,
        )

    cleaned_crop = PlotCropResult(
        cropped_bgr=cleaned_bgr,
        bbox=plot_crop.bbox,
        confidence=plot_crop.confidence,
        warnings=list(plot_crop.warnings),
    )

    original_curve, axis_mask = digitize_with_confidence(
        plot_crop,
        calibration,
        num_points=num_points,
        text_mask=text_mask,
        agent_meta=agent_meta,
        agent_prior=agent_prior,
        source="original",
        use_dp=False,
    )
    cleaned_curve, axis_mask_c = digitize_with_confidence(
        cleaned_crop,
        calibration,
        num_points=num_points,
        text_mask=text_mask,
        agent_meta=agent_meta,
        agent_prior=agent_prior,
        source="cleaned",
        other_curve=original_curve,
        use_dp=False,
    )
    if axis_mask is None:
        axis_mask = axis_mask_c

    if original_curve.two_theta:
        original_curve.confidence = score_curve_confidence(
            original_curve,
            plot_crop.cropped_bgr,
            calibration,
            agent_meta=agent_meta,
            text_mask=text_mask,
            agent_prior=agent_prior,
            other_curve=cleaned_curve,
            axis_mask=axis_mask,
        )

    orig_val = validate_extraction(
        plot_crop.cropped_bgr,
        original_curve,
        agent_meta,
        calibration,
        threshold=fusion_threshold,
        axis_mask=axis_mask,
    )
    clean_val = validate_extraction(
        cleaned_bgr,
        cleaned_curve,
        agent_meta,
        calibration,
        threshold=fusion_threshold,
        axis_mask=axis_mask,
    )

    need_dp = force_dp_tracing or (
        has_text_regions
        and _extraction_corrupted_by_text(orig_val, text_mask=text_mask)
        and (
            _extraction_corrupted_by_text(clean_val, text_mask=text_mask)
            or not clean_val.get("ok", False)
        )
    )

    dp_curve = _empty_curve("dp")
    if need_dp:
        LOGGER.info("Running DP tracer as optional fallback (force=%s)", force_dp_tracing)
        dp_curve, axis_mask_dp = digitize_with_confidence(
            cleaned_crop if np.any(removal_mask) else plot_crop,
            calibration,
            num_points=num_points,
            text_mask=text_mask,
            agent_meta=agent_meta,
            agent_prior=agent_prior,
            source="dp",
            other_curve=cleaned_curve if cleaned_curve.two_theta else original_curve,
            use_dp=True,
        )
        if axis_mask is None:
            axis_mask = axis_mask_dp
    else:
        LOGGER.info("Skipping DP tracer — original/cleaned digitizer sufficient")

    grid = None
    if cleaned_curve.two_theta:
        grid = np.asarray(cleaned_curve.two_theta, dtype=float)
    elif original_curve.two_theta:
        grid = np.asarray(original_curve.two_theta, dtype=float)
    if grid is not None:
        agent_prior = build_agent_prior_curve(agent_meta, calibration, grid=grid)

    mode_name, final_curve, candidate_scores = select_best_candidate(
        {
            "original": original_curve,
            "cleaned": cleaned_curve,
            "dp": dp_curve,
        },
        plot_crop.cropped_bgr,
        agent_meta,
        calibration,
        axis_mask=axis_mask,
    )
    hybrid_mode_used = mode_name if mode_name else "empty"
    # Preserve selected candidate arrays exactly — do not regenerate from peaks.
    selected = (
        original_curve
        if hybrid_mode_used == "original"
        else cleaned_curve
        if hybrid_mode_used == "cleaned"
        else dp_curve
        if hybrid_mode_used == "dp"
        else final_curve
    )
    final_curve = copy_curve_arrays(selected, source=hybrid_mode_used)

    if final_curve.two_theta:
        final_curve.confidence = score_curve_confidence(
            final_curve,
            plot_crop.cropped_bgr,
            calibration,
            agent_meta=agent_meta,
            text_mask=text_mask,
            agent_prior=agent_prior,
            other_curve=cleaned_curve if cleaned_curve.two_theta else original_curve,
            axis_mask=axis_mask,
        )

    validation = validate_extraction(
        plot_crop.cropped_bgr,
        final_curve,
        agent_meta,
        calibration,
        threshold=fusion_threshold,
        axis_mask=axis_mask,
    )
    validation["bbox_overlap"] = overlap_stats
    validation["hybrid_mode_used"] = hybrid_mode_used
    validation["selected_candidate"] = hybrid_mode_used
    validation["final_curve_source_function"] = (
        ORIGINAL_DIGITIZER_SOURCE_FUNCTION
        if hybrid_mode_used in {"original", "cleaned"}
        else "digitize_via_global_trace"
        if hybrid_mode_used == "dp"
        else "unknown"
    )
    if not overlap_ok:
        validation["flags"] = list(validation.get("flags") or []) + [
            "text_boxes_misaligned"
        ]

    return _finalize_artifacts(
        image_bgr=image_bgr,
        plot_crop=plot_crop,
        calibration=calibration,
        agent_meta=agent_meta,
        crop_regions=crop_regions,
        cleaned_bgr=cleaned_bgr,
        text_mask=text_mask,
        removal_mask=removal_mask,
        original_curve=original_curve,
        cleaned_curve=cleaned_curve,
        dp_curve=dp_curve,
        agent_prior=agent_prior,
        final_curve=final_curve,
        hybrid_mode_used=hybrid_mode_used,
        validation=validation,
        candidate_scores=candidate_scores,
        passthrough_debug={},
        bbox_debug=bbox_debug,
        axis_mask=axis_mask,
        fusion_threshold=fusion_threshold,
    )


def _finalize_artifacts(
    *,
    image_bgr: np.ndarray,
    plot_crop: PlotCropResult,
    calibration: AxisCalibrationResult,
    agent_meta: AgentFigureMetadata,
    crop_regions: list,
    cleaned_bgr: np.ndarray,
    text_mask: np.ndarray,
    removal_mask: np.ndarray,
    original_curve: CurveData,
    cleaned_curve: CurveData,
    dp_curve: CurveData,
    agent_prior: CurveData,
    final_curve: CurveData,
    hybrid_mode_used: str,
    validation: dict[str, Any],
    candidate_scores: dict[str, Any],
    passthrough_debug: dict[str, Any],
    bbox_debug: np.ndarray,
    axis_mask: np.ndarray | None,
    fusion_threshold: float,
) -> HybridDigitizationArtifacts:
    crop_h, crop_w = plot_crop.cropped_bgr.shape[:2]
    actual_h, actual_w = image_bgr.shape[:2]
    crop_meta = AgentFigureMetadata(
        text_regions=crop_regions,
        approximate_peaks=agent_meta.approximate_peaks,
        image_width=crop_w,
        image_height=crop_h,
        coordinate_space="plot_crop_pixels",
    )
    crop_overlay = render_hybrid_overlay(
        plot_crop.cropped_bgr,
        final_curve,
        calibration,
        crop_meta,
        text_mask=text_mask,
        threshold=fusion_threshold,
    )
    full_overlay = image_bgr.copy()
    x0, y0, x1, y1 = plot_crop.bbox
    slice_h, slice_w = y1 - y0, x1 - x0
    if crop_overlay.shape[0] == slice_h and crop_overlay.shape[1] == slice_w:
        full_overlay[y0:y1, x0:x1] = crop_overlay
    else:
        resized = cv2.resize(crop_overlay, (slice_w, slice_h), interpolation=cv2.INTER_AREA)
        full_overlay[y0:y1, x0:x1] = resized

    src_w = int(agent_meta.image_width or actual_w)
    src_h = int(agent_meta.image_height or actual_h)
    sx = actual_w / max(src_w, 1)
    sy = actual_h / max(src_h, 1)
    for region in agent_meta.text_regions:
        bx1 = int(round(region.bbox[0] * sx))
        by1 = int(round(region.bbox[1] * sy))
        bx2 = int(round(region.bbox[2] * sx))
        by2 = int(round(region.bbox[3] * sy))
        cv2.rectangle(full_overlay, (bx1, by1), (bx2, by2), (0, 165, 255), 1)

    return HybridDigitizationArtifacts(
        agent_meta=agent_meta,
        cleaned_bgr=cleaned_bgr,
        text_mask=text_mask,
        removal_mask=removal_mask,
        original_curve=original_curve,
        cleaned_curve=cleaned_curve,
        agent_prior=agent_prior,
        fused_curve=final_curve,
        dp_curve=dp_curve,
        hybrid_mode_used=hybrid_mode_used,
        validation=validation,
        candidate_scores=candidate_scores,
        passthrough_debug=passthrough_debug,
        overlay_bgr=full_overlay,
        bbox_debug_bgr=bbox_debug,
        crop_regions=crop_regions,
        axis_mask=axis_mask,
    )
