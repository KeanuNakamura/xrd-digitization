"""Autonomous curve extraction for PlotDigitizer pipeline.

Crops to the inner plot rectangle, segments curve pixels (Lab when colored,
darkness+continuity when black), rejects localized text via connected
components, and extracts one y per x-column with dynamic programming.
Does not rely on OCR/inpainting to remove or invent peak data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from xrd_digitization.types import AxisCalibrationResult

DEFAULT_SPINE_INSET = 4
MIN_CURVE_WIDTH_FRACTION = 0.12
DP_LAMBDA1 = 0.25
DP_LAMBDA2 = 0.10
DP_SCORE_WEIGHT = 5.0
MAX_GAP_INTERPOLATE = 8
MIN_PATH_COVERAGE = 0.35
MIN_Y_SPAN_FRACTION = 0.06
DP_MAX_COLUMN_GAP = 80


@dataclass
class ExtractDebugImages:
    plot_area_bgr: np.ndarray
    candidates_u8: np.ndarray
    rejected_text_u8: np.ndarray
    path_overlay_bgr: np.ndarray


@dataclass
class CurveExtractResult:
    """Pixel-space path and intermediate masks (ROI-local coordinates)."""

    rows: np.ndarray
    cols: np.ndarray
    candidate_mask: np.ndarray
    rejected_text_mask: np.ndarray
    score: np.ndarray
    plot_bounds: tuple[int, int, int, int]
    mode: str
    coverage: float
    y_span_fraction: float
    warnings: list[str] = field(default_factory=list)
    debug: ExtractDebugImages | None = None

    @property
    def ok(self) -> bool:
        return (
            self.coverage >= MIN_PATH_COVERAGE
            and self.y_span_fraction >= MIN_Y_SPAN_FRACTION
            and int(np.isfinite(self.rows).sum()) >= 20
        )


def inset_plot_bounds(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    inset: int = DEFAULT_SPINE_INSET,
    image_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Shrink plot rectangle past spines and inward tick stubs."""
    left_i = left + inset
    top_i = top + inset
    right_i = right - inset
    bottom_i = bottom - inset
    if image_shape is not None:
        height, width = image_shape
        left_i = int(np.clip(left_i, 0, width - 1))
        right_i = int(np.clip(right_i, left_i + 1, width))
        top_i = int(np.clip(top_i, 0, height - 1))
        bottom_i = int(np.clip(bottom_i, top_i + 1, height))
    if right_i - left_i < 8 or bottom_i - top_i < 8:
        return left, top, right, bottom
    return left_i, top_i, right_i, bottom_i


def detect_inner_plot_rectangle(
    image_bgr: np.ndarray,
    *,
    suggested: tuple[int, int, int, int] | None = None,
    inset: int = DEFAULT_SPINE_INSET,
) -> tuple[int, int, int, int]:
    """Detect the inner plotting rectangle from long axis spines."""
    height, width = image_bgr.shape[:2]
    if suggested is not None:
        left, top, right, bottom = suggested
        return inset_plot_bounds(
            left,
            top,
            right,
            bottom,
            inset=inset,
            image_shape=(height, width),
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 20)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 20), 1))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_kernel)
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)

    left_scores = vertical[:, : int(width * 0.35)].sum(axis=0)
    right_scores = vertical[:, int(width * 0.65) :].sum(axis=0)
    top_scores = horizontal[: int(height * 0.35), :].sum(axis=1)
    bottom_scores = horizontal[int(height * 0.65) :, :].sum(axis=1)

    left = int(np.argmax(left_scores)) if left_scores.size and left_scores.max() > 0 else int(width * 0.08)
    right = (
        int(width * 0.65 + np.argmax(right_scores))
        if right_scores.size and right_scores.max() > 0
        else int(width * 0.96)
    )
    top = int(np.argmax(top_scores)) if top_scores.size and top_scores.max() > 0 else int(height * 0.08)
    bottom = (
        int(height * 0.65 + np.argmax(bottom_scores))
        if bottom_scores.size and bottom_scores.max() > 0
        else int(height * 0.88)
    )
    return inset_plot_bounds(
        left,
        top,
        right,
        bottom,
        inset=inset,
        image_shape=(height, width),
    )


def mask_axes_and_ticks(
    height: int,
    width: int,
    *,
    band: int = 3,
) -> np.ndarray:
    """Boolean keep-mask that zeros a band along ROI borders (axes / tick stubs)."""
    keep = np.ones((height, width), dtype=bool)
    band = max(1, int(band))
    keep[:band, :] = False
    keep[-band:, :] = False
    keep[:, :band] = False
    keep[:, -band:] = False
    return keep


def segment_curve_candidates(
    roi_bgr: np.ndarray,
    *,
    keep_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Build a soft score map and binary candidate mask in Lab space.

    Returns (score float HxW, binary_u8 HxW, mode).
    """
    height, width = roi_bgr.shape[:2]
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if keep_mask is None:
        keep_mask = np.ones((height, width), dtype=bool)

    colored = (chroma > 8.0) & (L < 250.0) & keep_mask
    colored_cols = int(np.sum(colored.any(axis=0)))
    if colored_cols >= max(20, int(width * 0.25)):
        # Prefer blue/cyan ink (XRD curves) over reddish-brown annotations.
        blue_boost = np.clip((130.0 - b) / 35.0, 0.15, 2.5)
        red_penalty = 1.0 + np.clip((a - 128.0) / 25.0, 0.0, 3.0)
        chroma_score = np.clip(chroma / 18.0, 0.0, 2.5) * blue_boost / red_penalty
        chroma_score = chroma_score * np.clip((252.0 - L) / 50.0, 0.0, 1.0)
        score = np.where(keep_mask, chroma_score, 0.0)
        binary = ((score > 0.12) & keep_mask).astype(np.uint8) * 255
        return score.astype(np.float32), binary, "colored"

    # Black / gray: adaptive darkness from the thin ink percentile.
    ink_hi = float(np.percentile(gray[keep_mask], 8)) if keep_mask.any() else 120.0
    thr = float(np.clip(max(ink_hi + 45.0, 95.0), 95.0, 210.0))
    darkness = np.clip((thr - gray) / max(thr, 1.0), 0.0, 1.0)
    dark = (gray < thr) & keep_mask
    dark_f = dark.astype(np.float32)
    left = np.zeros_like(dark_f)
    right = np.zeros_like(dark_f)
    left[:, 1:] = dark_f[:, :-1]
    right[:, :-1] = dark_f[:, 1:]
    continuity = 0.35 + 0.65 * (left + right) / 2.0
    score = darkness * continuity
    score = np.where(keep_mask, score, 0.0)
    binary = dark.astype(np.uint8) * 255
    return score.astype(np.float32), binary, "black"


def reject_text_components(
    binary_u8: np.ndarray,
    score: np.ndarray,
    *,
    min_width_fraction: float = MIN_CURVE_WIDTH_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reject floating compact annotations; keep curve ink (incl. peak fragments).

    Text sits above peaks and does not reach the baseline band. Curve ink either
    spans width or touches the lower plot region.
    """
    height, width = binary_u8.shape
    min_width = max(8, int(width * min_width_fraction))
    baseline_row = int(height * 0.80)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    closed = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, close_kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    kept = np.zeros_like(binary_u8)
    rejected = np.zeros_like(binary_u8)

    for label in range(1, num_labels):
        _x, y, w, h, area = stats[label]
        component = labels == label
        touches_baseline = (y + h) >= baseline_row
        spans_curve = w >= min_width
        floating = (y + h) < baseline_row
        compact = area < width * height * 0.03 and w < min_width
        if floating and compact and not spans_curve:
            rejected[component] = 255
            continue
        kept[component] = 255

    if not kept.any():
        kept = binary_u8.copy()
        rejected = np.zeros_like(binary_u8)

    kept = cv2.bitwise_and(kept, binary_u8)
    rejected = cv2.bitwise_and(cv2.bitwise_or(rejected, binary_u8), cv2.bitwise_not(kept))

    score_out = score.copy()
    score_out[rejected > 0] = 0.0
    score_out[kept > 0] = np.maximum(score_out[kept > 0], 0.2)
    return kept, rejected


def optional_ocr_text_mask(
    height: int,
    width: int,
    text_boxes: Sequence[tuple[int, int, int, int]] | None,
    *,
    pad: int = 2,
) -> np.ndarray:
    """Secondary text mask from OCR boxes (ROI-local). Never required."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not text_boxes:
        return mask
    for x0, y0, x1, y1 in text_boxes:
        xa = int(np.clip(min(x0, x1) - pad, 0, width))
        xb = int(np.clip(max(x0, x1) + pad, 0, width))
        ya = int(np.clip(min(y0, y1) - pad, 0, height))
        yb = int(np.clip(max(y0, y1) + pad, 0, height))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 255
    return mask


def extract_path_dp(
    score: np.ndarray,
    *,
    lambda1: float = DP_LAMBDA1,
    lambda2: float = DP_LAMBDA2,
    score_weight: float = DP_SCORE_WEIGHT,
    max_candidates: int = 64,
    max_jump: int | None = None,
    max_column_gap: int = DP_MAX_COLUMN_GAP,
) -> np.ndarray:
    """
    Choose one row per column via dynamic programming.

    Empty columns are skipped with a gap penalty so fragmented black curves
    still form a continuous left-to-right path.
    """
    height, width = score.shape
    if max_jump is None:
        max_jump = max(30, height // 2)

    candidates: list[np.ndarray] = []
    for col in range(width):
        col_scores = score[:, col]
        positive = np.where(col_scores > 1e-4)[0]
        if positive.size == 0:
            candidates.append(np.array([], dtype=int))
            continue
        if positive.size <= max_candidates:
            candidates.append(positive)
            continue
        top_idx = positive[np.argpartition(col_scores[positive], -max_candidates)[-max_candidates:]]
        candidates.append(np.sort(top_idx))

    # Only optimize over columns that have candidates.
    active = [c for c in range(width) if len(candidates[c]) > 0]
    if not active:
        return np.full(width, np.nan, dtype=np.float64)

    dp_cost: dict[int, np.ndarray] = {}
    dp_prev: dict[int, np.ndarray] = {}  # previous active column index in active[]
    dp_prev_k: dict[int, np.ndarray] = {}
    dp_prev_y: dict[int, np.ndarray] = {}

    first = active[0]
    n0 = len(candidates[first])
    dp_cost[first] = np.array(
        [-score_weight * float(score[y, first]) for y in candidates[first]],
        dtype=np.float64,
    )
    dp_prev[first] = np.full(n0, -1, dtype=np.int32)
    dp_prev_k[first] = np.full(n0, -1, dtype=np.int32)
    dp_prev_y[first] = np.full(n0, np.nan, dtype=np.float64)

    for ai, col in enumerate(active[1:], start=1):
        cands = candidates[col]
        n = len(cands)
        costs = np.full(n, np.inf, dtype=np.float64)
        prev_ai = np.full(n, -1, dtype=np.int32)
        prev_k = np.full(n, -1, dtype=np.int32)
        prev_y = np.full(n, np.nan, dtype=np.float64)

        # Consider several recent active columns (bridge short gaps).
        lookback_start = max(0, ai - max_column_gap)
        for k, y in enumerate(cands):
            unary = -score_weight * float(score[y, col])
            best = np.inf
            best_ai = -1
            best_j = -1
            best_py = np.nan
            for aj in range(lookback_start, ai):
                prev_col = active[aj]
                gap = col - prev_col
                if gap > max_column_gap:
                    continue
                prev_cands = candidates[prev_col]
                prev_costs = dp_cost[prev_col]
                prev_prev_ys = dp_prev_y[prev_col]
                gap_pen = 0.8 * max(0, gap - 1)
                for j, y_prev in enumerate(prev_cands):
                    jump = abs(int(y) - int(y_prev))
                    if jump > max_jump:
                        continue
                    first_ord = lambda1 * min(jump, 50) + 0.04 * max(0, jump - 50)
                    y_prev2 = prev_prev_ys[j]
                    if np.isfinite(y_prev2):
                        second = lambda2 * abs(int(y) - 2 * int(y_prev) + int(y_prev2))
                    else:
                        second = 0.0
                    total = prev_costs[j] + unary + first_ord + second + gap_pen
                    if total < best:
                        best = total
                        best_ai = aj
                        best_j = j
                        best_py = float(y_prev)
            if best_j < 0:
                costs[k] = unary + 12.0
            else:
                costs[k] = best
                prev_ai[k] = best_ai
                prev_k[k] = best_j
                prev_y[k] = best_py

        dp_cost[col] = costs
        dp_prev[col] = prev_ai
        dp_prev_k[col] = prev_k
        dp_prev_y[col] = prev_y

    # Backtrack.
    path = np.full(width, np.nan, dtype=np.float64)
    end_col = active[-1]
    k = int(np.nanargmin(dp_cost[end_col]))
    ai = len(active) - 1
    while ai >= 0 and k >= 0:
        col = active[ai]
        path[col] = float(candidates[col][k])
        paj = int(dp_prev[col][k])
        pj = int(dp_prev_k[col][k])
        if paj < 0 or pj < 0:
            break
        ai = paj
        k = pj

    return path


def interpolate_short_gaps(
    path: np.ndarray,
    *,
    max_gap: int = MAX_GAP_INTERPOLATE,
) -> np.ndarray:
    """Linearly interpolate only short NaN runs; leave longer holes as NaN."""
    out = path.astype(np.float64).copy()
    width = len(out)
    i = 0
    while i < width:
        if np.isfinite(out[i]):
            i += 1
            continue
        start = i
        while i < width and not np.isfinite(out[i]):
            i += 1
        end = i
        gap = end - start
        left = start - 1
        right = end
        if (
            gap <= max_gap
            and left >= 0
            and right < width
            and np.isfinite(out[left])
            and np.isfinite(out[right])
        ):
            for g in range(gap):
                t = (g + 1) / (gap + 1)
                out[start + g] = (1.0 - t) * out[left] + t * out[right]
    return out


def path_quality(path: np.ndarray, height: int) -> tuple[float, float]:
    """Return (coverage fraction, y-span / height)."""
    finite = np.isfinite(path)
    coverage = float(finite.mean()) if path.size else 0.0
    if not finite.any():
        return coverage, 0.0
    ys = path[finite]
    y_span = float(ys.max() - ys.min()) / max(height, 1)
    return coverage, y_span


def _column_median_path(score: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """
    Per-column path: max-score pixel among kept ink.

    No continuity prior here — XRD peaks require large vertical jumps; DP
    handles smoothness when selected. This fallback maximizes ink fidelity.
    """
    height, width = score.shape
    path = np.full(width, np.nan, dtype=np.float64)
    for col in range(width):
        mask = (kept[:, col] > 0) & (score[:, col] > 1e-4)
        if not mask.any():
            mask = score[:, col] > 1e-4
        rows = np.where(mask)[0]
        if rows.size == 0:
            continue
        weights = score[rows, col]
        path[col] = float(rows[int(np.argmax(weights))])

    finite = np.isfinite(path)
    if finite.sum() >= 5:
        vals = path.copy()
        for i in np.where(finite)[0]:
            lo = max(0, i - 1)
            hi = min(width, i + 2)
            window = vals[lo:hi]
            window = window[np.isfinite(window)]
            if window.size:
                med = float(np.median(window))
                if abs(vals[i] - med) <= max(3.0, height * 0.02):
                    path[i] = med
    return path


def _build_debug_images(
    source_bgr: np.ndarray,
    plot_bounds: tuple[int, int, int, int],
    roi_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    rejected_mask: np.ndarray,
    path: np.ndarray,
) -> ExtractDebugImages:
    left, top, right, bottom = plot_bounds
    plot_area = source_bgr.copy()
    cv2.rectangle(plot_area, (left, top), (right - 1, bottom - 1), (0, 180, 0), 2)

    overlay = roi_bgr.copy()
    red = overlay.copy()
    red[rejected_mask > 0] = (0, 0, 220)
    overlay = cv2.addWeighted(overlay, 0.65, red, 0.35, 0)
    for col, row in enumerate(path):
        if not np.isfinite(row):
            continue
        cv2.circle(overlay, (col, int(round(row))), 1, (255, 200, 0), -1)

    return ExtractDebugImages(
        plot_area_bgr=plot_area,
        candidates_u8=candidate_mask.copy(),
        rejected_text_u8=rejected_mask.copy(),
        path_overlay_bgr=overlay,
    )


def write_debug_outputs(
    debug: ExtractDebugImages,
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    """Write plot-area / candidates / rejected-text / path debug PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plot_area": output_dir / f"{stem}_plot_area.png",
        "candidates": output_dir / f"{stem}_candidates.png",
        "rejected_text": output_dir / f"{stem}_rejected_text.png",
        "path": output_dir / f"{stem}_path.png",
    }
    cv2.imwrite(str(paths["plot_area"]), debug.plot_area_bgr)
    cv2.imwrite(str(paths["candidates"]), debug.candidates_u8)
    cv2.imwrite(str(paths["rejected_text"]), debug.rejected_text_u8)
    cv2.imwrite(str(paths["path"]), debug.path_overlay_bgr)
    return paths


def map_path_to_calibrated_xy(
    path_rows: np.ndarray,
    *,
    calibration: AxisCalibrationResult,
    plot_bounds: tuple[int, int, int, int],
    y_scale: float | None = None,
) -> np.ndarray:
    """Map ROI-local path rows to (two_theta, intensity)."""
    left, top, right, bottom = plot_bounds
    width = right - left
    if width <= 0 or bottom - top <= 0:
        return np.zeros((0, 2), dtype=float)

    cols = np.arange(width, dtype=float)
    finite = np.isfinite(path_rows)
    if not finite.any():
        return np.zeros((0, 2), dtype=float)

    x_pixel = left + cols
    span_x = max(calibration.plot_right - calibration.plot_left, 1)
    two_theta = calibration.x_min + (x_pixel - calibration.plot_left) / span_x * (
        calibration.x_max - calibration.x_min
    )
    y_pixel = top + path_rows

    if calibration.has_y_calibration and len(calibration.y_tick_pairs) >= 2:
        ys = np.array([p[0] for p in calibration.y_tick_pairs], dtype=float)
        vals = np.array([p[1] for p in calibration.y_tick_pairs], dtype=float)
        coeffs = np.polyfit(ys, vals, 1)
        intensity = coeffs[0] * y_pixel + coeffs[1]
        y_lo = min(calibration.y_min or 0.0, calibration.y_max or 0.0)
        y_hi = max(calibration.y_min or 0.0, calibration.y_max or 0.0)
        intensity = np.clip(intensity, y_lo, y_hi)
    else:
        span_y = max(bottom - top, 1)
        frac = (bottom - y_pixel) / span_y
        scale = float(y_scale) if y_scale is not None else 1.0
        if calibration.y_max is not None and calibration.y_min is not None:
            scale = float(calibration.y_max - calibration.y_min)
        intensity = np.clip(frac, 0.0, None) * scale

    data = np.column_stack((two_theta[finite], intensity[finite]))
    return data[np.argsort(data[:, 0])]


def extract_curve_autonomous(
    image_bgr: np.ndarray,
    *,
    plot_bounds: tuple[int, int, int, int] | None = None,
    text_boxes_roi: Sequence[tuple[int, int, int, int]] | None = None,
    spine_inset: int = DEFAULT_SPINE_INSET,
    build_debug: bool = True,
) -> CurveExtractResult:
    """Full autonomous preprocess + DP curve extraction on one figure image."""
    warnings: list[str] = []
    bounds = detect_inner_plot_rectangle(
        image_bgr,
        suggested=plot_bounds,
        inset=spine_inset,
    )
    left, top, right, bottom = bounds
    roi = image_bgr[top:bottom, left:right].copy()
    if roi.size == 0:
        empty = np.zeros((0,), dtype=float)
        return CurveExtractResult(
            rows=empty,
            cols=empty.astype(int),
            candidate_mask=np.zeros((1, 1), dtype=np.uint8),
            rejected_text_mask=np.zeros((1, 1), dtype=np.uint8),
            score=np.zeros((1, 1), dtype=np.float32),
            plot_bounds=bounds,
            mode="black",
            coverage=0.0,
            y_span_fraction=0.0,
            warnings=["empty_plot_roi"],
        )

    h, w = roi.shape[:2]
    keep = mask_axes_and_ticks(h, w, band=max(2, spine_inset - 1))
    score, binary, mode = segment_curve_candidates(roi, keep_mask=keep)

    ocr_mask = optional_ocr_text_mask(h, w, text_boxes_roi)
    if ocr_mask.any():
        binary = binary.copy()
        binary[ocr_mask > 0] = 0
        score = score.copy()
        score[ocr_mask > 0] = 0.0
        warnings.append("applied_secondary_ocr_text_mask")

    kept, rejected = reject_text_components(binary, score)
    if ocr_mask.any():
        rejected = cv2.bitwise_or(rejected, ocr_mask)

    dp_score = score.copy()
    dp_score[rejected > 0] = 0.0
    if kept.any():
        dp_score[kept > 0] = np.maximum(dp_score[kept > 0], 0.25)

    # Compare DP and column-max; black curves are often fragmented so max usually wins.
    dp_path = extract_path_dp(dp_score)
    dp_path = interpolate_short_gaps(dp_path)
    max_path = _column_median_path(dp_score, kept)
    max_path = interpolate_short_gaps(max_path, max_gap=MAX_GAP_INTERPOLATE)

    def _path_score(p: np.ndarray) -> float:
        cov, span = path_quality(p, h)
        finite = p[np.isfinite(p)]
        if finite.size < 20:
            return -1.0
        y_unique = float(len(np.unique(np.round(finite, 0))))
        return cov * 100.0 + span * 80.0 + min(y_unique, 200) * 0.2

    if _path_score(max_path) >= _path_score(dp_path):
        path = max_path
        warnings.append(
            "used_column_max_black" if mode == "black" else "used_column_max_colored"
        )
    else:
        path = dp_path
        warnings.append("used_dp_path")
    coverage, y_span_fraction = path_quality(path, h)

    if coverage < MIN_PATH_COVERAGE:
        warnings.append(f"low_path_coverage={coverage:.2f}")
    if y_span_fraction < MIN_Y_SPAN_FRACTION:
        warnings.append(f"low_y_span={y_span_fraction:.3f}")

    debug = None
    if build_debug:
        debug = _build_debug_images(image_bgr, bounds, roi, kept, rejected, path)

    return CurveExtractResult(
        rows=path,
        cols=np.arange(w, dtype=int),
        candidate_mask=kept,
        rejected_text_mask=rejected,
        score=dp_score,
        plot_bounds=bounds,
        mode=mode,
        coverage=coverage,
        y_span_fraction=y_span_fraction,
        warnings=warnings,
        debug=debug,
    )
