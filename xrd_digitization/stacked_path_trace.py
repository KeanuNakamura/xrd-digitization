"""Continuous path tracing for stacked XRD curves.

Baselines are found via horizontal-persistence (coverage across x-windows),
not raw row ink density. Each curve is then extracted as a left-to-right
Viterbi path near its baseline, with asymmetric upward peak allowance and
penalties for compact annotation markers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from xrd_digitization.types import AxisCalibrationResult

LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineCandidate:
    y: int
    coverage: float
    confidence: float


@dataclass
class TraceQuality:
    coverage: float
    longest_gap_px: int
    jump_count: int
    median_baseline_dev_px: float
    confidence: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "longest_gap_px": self.longest_gap_px,
            "jump_count": self.jump_count,
            "median_baseline_dev_px": self.median_baseline_dev_px,
            "confidence": self.confidence,
            "flags": list(self.flags),
        }


@dataclass
class PathTraceResult:
    curve_id: int
    baseline_px: float
    search_top: int
    search_bottom: int
    pixel_path: list[tuple[int, int]]  # (x, y) in cropped image coords
    missing_mask: list[bool]
    quality: TraceQuality
    label: str | None = None


def build_ink_mask(
    image_bgr: np.ndarray,
    *,
    dark_threshold: int = 140,
    colored_only: bool = False,
    min_saturation: int = 40,
    light_gray: bool | None = None,
    light_threshold: int = 248,
) -> np.ndarray:
    """Boolean ink mask for dark/saturated curve strokes.

    When ``colored_only`` is True, ignore low-saturation dark ink (black text /
    axes) so strongly colored XRD traces can be followed on the original image.

    Anti-aliased black XRD lines are often lighter than ``dark_threshold``.
    When ``light_gray`` is True (or auto-detected), treat near-white deviations
    as ink so thin gray baselines remain visible.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    colored = (hsv[:, :, 1] > min_saturation) & (gray < 250)
    if colored_only:
        return colored
    dark = gray < dark_threshold
    ink = colored | dark
    if light_gray is None:
        light_gray = bool(ink.mean() < 0.03 and float(gray.mean()) > 230.0)
    if light_gray:
        light = gray < light_threshold
        ink = ink | light
    return ink


def mask_plot_frame(
    height: int,
    width: int,
    calibration: AxisCalibrationResult,
    *,
    inset: int | None = None,
) -> np.ndarray:
    """
    True where baseline/trace search is allowed (plot interior minus frame).

    Excludes a margin inside the axis box so the frame and tick stubs do not
    register as persistent horizontal baselines.
    """
    left = int(calibration.plot_left)
    right = int(calibration.plot_right)
    top = int(calibration.plot_top)
    bottom = int(calibration.plot_bottom)
    if inset is None:
        inset = max(4, int(0.012 * max(right - left, bottom - top)))
    allow = np.zeros((height, width), dtype=bool)
    y0 = min(height, max(0, top + inset))
    y1 = min(height, max(y0 + 1, bottom - inset))
    x0 = min(width, max(0, left + inset))
    x1 = min(width, max(x0 + 1, right - inset))
    allow[y0:y1, x0:x1] = True
    return allow


def horizontal_coverage_profile(
    ink: np.ndarray,
    allow: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    n_windows: int = 64,
    vertical_radius: int = 2,
) -> np.ndarray:
    """
    For each row y in [y0, y1), fraction of x-windows that contain ink near y.

    Tall peaks and compact annotations light up few windows; real baselines
    light up most windows.
    """
    height = y1 - y0
    if height <= 0 or x1 <= x0:
        return np.zeros(0, dtype=float)

    n_windows = max(8, min(n_windows, x1 - x0))
    edges = np.linspace(x0, x1, n_windows + 1, dtype=int)
    coverage = np.zeros(height, dtype=float)

    # Precompute, per window, the set of y rows (relative) with ink.
    window_rows: list[np.ndarray] = []
    for w in range(n_windows):
        xa, xb = int(edges[w]), int(edges[w + 1])
        if xb <= xa:
            xb = xa + 1
        region = ink[y0:y1, xa:xb] & allow[y0:y1, xa:xb]
        rows = np.flatnonzero(region.any(axis=1))
        window_rows.append(rows)

    radius = max(0, int(vertical_radius))
    for local_y in range(height):
        hits = 0
        lo = local_y - radius
        hi = local_y + radius
        for rows in window_rows:
            if rows.size == 0:
                continue
            # Any ink row in [lo, hi] within this window.
            if np.any((rows >= lo) & (rows <= hi)):
                hits += 1
        coverage[local_y] = hits / float(n_windows)
    return coverage


def detect_persistent_baselines(
    image_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    estimated_curve_count: int | None = None,
    n_windows: int = 64,
    min_coverage: float = 0.35,
    top_exclude_frac: float = 0.04,
    bottom_exclude_frac: float = 0.03,
) -> tuple[list[BaselineCandidate], np.ndarray, np.ndarray]:
    """
    Detect stacked XRD baselines via horizontal coverage.

    Returns (candidates top→bottom, coverage profile aligned to full image y,
    allow mask).
    """
    height, width = image_bgr.shape[:2]
    ink = build_ink_mask(image_bgr)
    allow = mask_plot_frame(height, width, calibration)

    plot_top = int(calibration.plot_top)
    plot_bottom = int(calibration.plot_bottom)
    plot_left = int(calibration.plot_left)
    plot_right = int(calibration.plot_right)
    plot_h = max(1, plot_bottom - plot_top)

    # Extra exclusion near top (legend / peak tips) and bottom (axis ticks).
    y0 = plot_top + max(6, int(plot_h * top_exclude_frac))
    y1 = plot_bottom - max(4, int(plot_h * bottom_exclude_frac))
    x0 = plot_left + max(4, int((plot_right - plot_left) * 0.02))
    x1 = plot_right - max(4, int((plot_right - plot_left) * 0.02))
    if y1 - y0 < 40:
        y0, y1 = plot_top + 4, plot_bottom - 4

    # Also blank a top-right legend strip inside allow (heuristic).
    legend = allow.copy()
    legend_x0 = plot_left + int(0.72 * (plot_right - plot_left))
    legend_y1 = plot_top + int(0.14 * plot_h)
    legend[plot_top:legend_y1, legend_x0:plot_right] = False
    allow = legend

    local = horizontal_coverage_profile(
        ink,
        allow,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        n_windows=n_windows,
        vertical_radius=2,
    )
    full_coverage = np.zeros(height, dtype=float)
    if local.size:
        full_coverage[y0 : y0 + local.size] = local

    if local.size == 0:
        return [], full_coverage, allow

    kernel = max(5, plot_h // 60)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = np.convolve(local, np.ones(kernel) / kernel, mode="same")

    try:
        from scipy.signal import find_peaks
    except ImportError:  # pragma: no cover
        find_peaks = None

    expected = estimated_curve_count if estimated_curve_count and estimated_curve_count >= 2 else None
    # Min spacing: if expecting K curves, keep them from collapsing.
    if expected is not None:
        min_dist = max(12, int(0.55 * plot_h / max(expected, 1)))
    else:
        min_dist = max(12, plot_h // 16)

    peaks: np.ndarray
    if find_peaks is not None:
        peaks, props = find_peaks(
            smoothed,
            height=max(min_coverage * 0.85, float(smoothed.max()) * 0.25),
            distance=min_dist,
            prominence=max(0.05, float(smoothed.max()) * 0.08),
        )
    else:
        peaks = np.array([], dtype=int)
        props = {}

    candidates: list[BaselineCandidate] = []
    for p in peaks:
        cov = float(smoothed[p])
        conf = float(np.clip(cov / max(0.2, float(smoothed.max())), 0.0, 1.0))
        candidates.append(
            BaselineCandidate(y=int(y0 + p), coverage=cov, confidence=conf)
        )

    # If we have more peaks than expected, keep the highest-coverage ones.
    if expected is not None and len(candidates) > expected:
        candidates = sorted(candidates, key=lambda c: c.coverage, reverse=True)[:expected]
        candidates.sort(key=lambda c: c.y)

    # If we have fewer than expected, relax threshold and retry once.
    if expected is not None and len(candidates) < expected and find_peaks is not None:
        peaks2, _ = find_peaks(
            smoothed,
            height=max(0.2, min_coverage * 0.55),
            distance=max(8, min_dist // 2),
            prominence=max(0.03, float(smoothed.max()) * 0.04),
        )
        extra: list[BaselineCandidate] = []
        existing_y = {c.y for c in candidates}
        for p in peaks2:
            y = int(y0 + p)
            if any(abs(y - ey) < min_dist // 2 for ey in existing_y):
                continue
            cov = float(smoothed[p])
            conf = float(np.clip(cov / max(0.2, float(smoothed.max())), 0.0, 1.0))
            extra.append(BaselineCandidate(y=y, coverage=cov, confidence=conf))
        pool = candidates + extra
        pool = sorted(pool, key=lambda c: c.coverage, reverse=True)[:expected]
        candidates = sorted(pool, key=lambda c: c.y)

    candidates.sort(key=lambda c: c.y)
    LOGGER.info(
        "Persistent baselines: %d (estimate=%s) ys=%s coverage=%s",
        len(candidates),
        expected,
        [c.y for c in candidates],
        [round(c.coverage, 3) for c in candidates],
    )
    return candidates, full_coverage, allow


def asymmetric_search_bounds(
    baselines: list[float],
    index: int,
    *,
    plot_top: int,
    plot_bottom: int,
    upper_frac: float = 0.85,
    lower_frac: float = 0.18,
    min_upper: int = 20,
    min_lower: int = 4,
) -> tuple[int, int]:
    """
    Allowed [top, bottom) search region for curve ``index`` (image y down).

    Large allowance ABOVE the baseline (toward previous curve / smaller y),
    small allowance BELOW (toward next curve / larger y).
    """
    b = float(baselines[index])
    if index == 0:
        # Leave headroom under the top frame/legend.
        usable_top = plot_top + max(8, int(0.06 * (b - plot_top)))
        gap_above = max(min_upper, b - usable_top)
    else:
        usable_top = plot_top
        gap_above = max(min_upper, b - float(baselines[index - 1]))
    if index == len(baselines) - 1:
        gap_below = max(min_lower, plot_bottom - b)
    else:
        gap_below = max(min_lower, float(baselines[index + 1]) - b)

    top = int(max(usable_top if index == 0 else plot_top, round(b - upper_frac * gap_above)))
    if index == 0:
        top = max(top, usable_top)
    bottom = int(min(plot_bottom, round(b + lower_frac * gap_below)))
    if bottom <= top + 2:
        bottom = min(plot_bottom, top + 3)
    return top, bottom


def detect_annotation_mask(
    image_bgr: np.ndarray,
    allow: np.ndarray,
    *,
    max_aspect: float = 3.2,
    max_width_frac: float = 0.055,
    max_height_frac: float = 0.10,
    min_area: int = 6,
    max_area_frac: float = 0.012,
) -> np.ndarray:
    """
    Mask compact isolated components (circles/squares/legend glyphs).

    Conservative: only small, roughly square blobs. Tall thin peaks connected
    to a long horizontal stroke are left alone (path continuity decides later).
    """
    height, width = image_bgr.shape[:2]
    ink = build_ink_mask(image_bgr)
    work = (ink & allow).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    mask = np.zeros((height, width), dtype=bool)
    max_w = max(8, int(width * max_width_frac))
    max_h = max(8, int(height * max_height_frac))
    max_area = max(40, int(height * width * max_area_frac))

    for label_id in range(1, num):
        x, y, w, h, area = stats[label_id]
        if area < min_area or area > max_area:
            continue
        if w > max_w or h > max_h:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > max_aspect:
            continue
        # Prefer components that are not extremely elongated horizontally
        # (those are more likely path fragments).
        if w >= max(20, int(width * 0.08)):
            continue
        # Compact "blobbiness": area close to bounding box (circles/squares).
        fill = area / max(1.0, float(w * h))
        if fill < 0.35 and aspect > 2.0:
            continue
        mask[labels == label_id] = True

    # Dilate slightly so peak-adjacent markers are fully covered.
    if mask.any():
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return mask


def _column_representatives(
    ink_col: np.ndarray,
    ann_col: np.ndarray,
    y_top: int,
    y_bottom: int,
    *,
    baseline: int,
    prev_y: int | None,
    max_candidates: int = 24,
) -> list[tuple[int, float]]:
    """
    One representative y per vertical ink run in the column.

    Prefer run centroids (stroke centerline). Always retain the run nearest the
    baseline and the highest run (peak tip) so sharp XRD peaks remain reachable.
    """
    ys = np.flatnonzero(ink_col[y_top:y_bottom]) + y_top
    if ys.size == 0:
        return []

    # Split into contiguous vertical runs.
    runs: list[np.ndarray] = []
    start = 0
    for i in range(1, len(ys)):
        if ys[i] != ys[i - 1] + 1:
            runs.append(ys[start:i])
            start = i
    runs.append(ys[start:])

    reps: list[tuple[int, float, str]] = []
    for run in runs:
        y_rep = int(np.median(run))
        ann = 1.0 if bool(ann_col[y_rep]) or bool(ann_col[run].mean() > 0.4) else 0.0
        # Tag role for prioritization when we must downsample.
        if abs(y_rep - baseline) <= 4:
            role = "base"
        elif int(run.min()) == int(ys.min()):
            role = "peak"
        else:
            role = "mid"
        reps.append((y_rep, ann, role))

    # Always keep baseline-nearest and peak-tip (highest) reps.
    keep: dict[int, tuple[int, float]] = {}
    if reps:
        nearest_base = min(reps, key=lambda r: abs(r[0] - baseline))
        highest = min(reps, key=lambda r: r[0])  # smaller y = higher on page
        keep[nearest_base[0]] = (nearest_base[0], nearest_base[1])
        keep[highest[0]] = (highest[0], highest[1])
        if prev_y is not None:
            nearest_prev = min(reps, key=lambda r: abs(r[0] - prev_y))
            keep[nearest_prev[0]] = (nearest_prev[0], nearest_prev[1])
        for y, ann, _role in reps:
            if len(keep) >= max_candidates:
                break
            keep[y] = (y, ann)
    return [keep[k] for k in sorted(keep)]


def viterbi_trace_curve(
    ink: np.ndarray,
    annotation: np.ndarray,
    *,
    baseline_px: float,
    search_top: int,
    search_bottom: int,
    x0: int,
    x1: int,
    continuity_weight: float = 0.85,
    upward_continuity_scale: float = 0.25,
    baseline_below_weight: float = 5.0,
    baseline_above_weight: float = 0.01,
    annotation_weight: float = 45.0,
    missing_weight: float = 5.0,
    max_gap: int = 28,
    jump_hard: int | None = None,
    frame_margin: int = 6,
) -> tuple[list[tuple[int, int]], list[bool], TraceQuality]:
    """
    Dynamic-programming path y(x) minimizing continuity + baseline + annotation costs.

    Upward moves (toward peaks, smaller image y) are cheaper than downward moves
    so sharp XRD peaks can be climbed without abandoning the baseline elsewhere.
    """
    height, width = ink.shape
    x0 = max(0, x0)
    x1 = min(width, x1)
    baseline = int(round(baseline_px))
    search_top = max(0, search_top)
    search_bottom = min(height, max(search_top + 2, search_bottom))
    # Keep path off the plot frame.
    search_top = max(search_top, frame_margin)
    if jump_hard is None:
        jump_hard = max(55, int(0.9 * (baseline - search_top)))

    xs = list(range(x0, x1))
    if not xs:
        empty_q = TraceQuality(0.0, 0, 0, 0.0, 0.0, ["empty_x_range"])
        return [], [], empty_q

    states: list[list[tuple[int | None, float]]] = []
    back: list[list[tuple[int, int]]] = []
    prev_costs: list[float] = []
    prev_y_hint: int | None = baseline

    for xi, x in enumerate(xs):
        cands = _column_representatives(
            ink[:, x],
            annotation[:, x],
            search_top,
            search_bottom,
            baseline=baseline,
            prev_y=prev_y_hint,
        )
        state_defs: list[tuple[int | None, float]] = [(y, ann) for y, ann in cands]
        # Missing state is always available but heavily penalized when ink exists.
        state_defs.append((None, 0.0))
        states.append(state_defs)
        has_ink = len(cands) > 0
        local_missing_weight = missing_weight * (3.5 if has_ink else 1.0)

        if xi == 0:
            costs = []
            for y, ann in state_defs:
                if y is None:
                    costs.append(local_missing_weight * 2.0)
                else:
                    below = max(0, y - baseline)
                    above = max(0, baseline - y)
                    costs.append(
                        baseline_below_weight * below
                        + baseline_above_weight * above
                        + annotation_weight * ann
                    )
            back.append([(-1, -1 if y is None else int(y)) for y, _ in state_defs])
            prev_costs = costs
            # Seed toward baseline.
            finite = [(i, c) for i, (st, c) in enumerate(zip(state_defs, costs)) if st[0] is not None]
            if finite:
                prev_y_hint = int(state_defs[min(finite, key=lambda t: t[1])[0]][0])  # type: ignore[index]
            continue

        prev_defs = states[xi - 1]
        new_costs: list[float] = []
        new_back: list[tuple[int, int]] = []
        for y, ann in state_defs:
            best = 1e18
            best_j = 0
            chosen_y = baseline
            for j, (py, _pann) in enumerate(prev_defs):
                prev_y = py if py is not None else (
                    back[xi - 1][j][1] if back[xi - 1][j][1] >= 0 else baseline
                )
                if y is None:
                    # Missing observation: drift toward baseline instead of
                    # floating at a previous peak tip across empty columns.
                    if abs(prev_y - baseline) <= 6:
                        cand_y = int(prev_y)
                        step = local_missing_weight
                    else:
                        drift = min(18, abs(prev_y - baseline))
                        cand_y = int(prev_y + (1 if baseline > prev_y else -1) * drift)
                        step = local_missing_weight + 0.35 * drift
                else:
                    dy = y - prev_y
                    ady = abs(dy)
                    # Climbing peaks (up) is cheap; returning toward baseline is
                    # also relatively cheap; lateral/downward away from baseline
                    # is expensive.
                    moving_toward_baseline = abs(y - baseline) < abs(prev_y - baseline)
                    if dy < 0:
                        cont = continuity_weight * upward_continuity_scale * ady
                    elif moving_toward_baseline:
                        cont = continuity_weight * 0.35 * ady
                    else:
                        cont = continuity_weight * ady
                    if ady > jump_hard:
                        cont += continuity_weight * 2.5 * (ady - jump_hard)
                    below = max(0, y - baseline)
                    above = max(0, baseline - y)
                    # Mild bonus for leaving baseline when climbing into ink peak.
                    peak_bonus = 0.0
                    if dy < -3 and above > 5:
                        peak_bonus = -0.15 * min(above, jump_hard)
                    # Heavy penalty near the top frame (legend / border spikes).
                    frame_pen = 0.0
                    if y <= search_top + 4:
                        frame_pen = 40.0
                    step = (
                        cont
                        + baseline_below_weight * below
                        + baseline_above_weight * above
                        + annotation_weight * ann
                        + peak_bonus
                        + frame_pen
                    )
                    cand_y = int(y)
                total = prev_costs[j] + step
                if total < best:
                    best = total
                    best_j = j
                    chosen_y = cand_y
            new_costs.append(best)
            new_back.append((best_j, chosen_y))
        back.append(new_back)
        prev_costs = new_costs
        # Update hint from best non-miss state.
        best_i = int(np.argmin(new_costs))
        if states[xi][best_i][0] is not None:
            prev_y_hint = int(states[xi][best_i][0])  # type: ignore[arg-type]
        else:
            prev_y_hint = new_back[best_i][1]

    last_costs = prev_costs
    end = int(np.argmin(last_costs))
    path_rev: list[tuple[int, int]] = []
    miss_rev: list[bool] = []
    state_idx = end
    for xi in range(len(xs) - 1, -1, -1):
        x = xs[xi]
        y_opt = states[xi][state_idx][0]
        chosen = back[xi][state_idx][1]
        is_miss = y_opt is None
        if chosen < 0:
            chosen = baseline
        path_rev.append((x, int(chosen)))
        miss_rev.append(bool(is_miss))
        if xi == 0:
            break
        state_idx = back[xi][state_idx][0]

    path = list(reversed(path_rev))
    missing = list(reversed(miss_rev))

    cleaned: list[tuple[int, int]] = []
    cleaned_miss: list[bool] = []
    gap = 0
    jump_count = 0
    prev_kept_y: int | None = None
    for (x, y), is_miss in zip(path, missing):
        if is_miss:
            gap += 1
            if gap > max_gap:
                continue
        else:
            gap = 0
        if prev_kept_y is not None and abs(y - prev_kept_y) > jump_hard:
            jump_count += 1
        cleaned.append((x, y))
        cleaned_miss.append(is_miss)
        prev_kept_y = y

    path = cleaned
    missing = cleaned_miss

    observed = sum(1 for m in missing if not m)
    span = max(1, x1 - x0)
    coverage = observed / float(span)

    longest = 0
    cur = 0
    for m in missing:
        if m:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    near = [
        abs(y - baseline)
        for (x, y), m in zip(path, missing)
        if not m and abs(y - baseline) < 12
    ]
    med_dev = (
        float(np.median(near))
        if near
        else float(np.median([abs(y - baseline) for (_, y) in path]) if path else 0.0)
    )

    flags: list[str] = []
    if coverage < 0.55:
        flags.append("low_coverage")
    if longest > max_gap:
        flags.append("long_gap")
    if jump_count > max(3, span // 80):
        flags.append("many_jumps")

    # Reward following tall structure when present.
    peak_follow = 0.0
    if path:
        amps = [max(0, baseline - y) for _, y in path]
        if max(amps) > 15:
            peak_follow = min(1.0, float(np.percentile(amps, 90)) / 80.0)

    confidence = float(
        np.clip(
            0.40 * coverage
            + 0.20 * (1.0 - min(1.0, longest / max(1.0, 2 * max_gap)))
            + 0.15 * (1.0 - min(1.0, jump_count / 20.0))
            + 0.10 * (1.0 - min(1.0, med_dev / 20.0))
            + 0.15 * peak_follow,
            0.0,
            1.0,
        )
    )
    if flags:
        confidence *= 0.75

    quality = TraceQuality(
        coverage=float(coverage),
        longest_gap_px=int(longest),
        jump_count=int(jump_count),
        median_baseline_dev_px=float(med_dev),
        confidence=float(confidence),
        flags=flags,
    )
    return path, missing, quality


def pixel_path_to_calibrated_xy(
    path: list[tuple[int, int]],
    calibration: AxisCalibrationResult,
    *,
    baseline_px: float,
    missing: list[bool] | None = None,
    max_gap_px: int = 28,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Convert pixel path to calibrated XY using shared x-axis only.

    Returns (raw_pixel_xy as [[x_px,y_px],...], data_xy [[theta, intensity],...],
    baseline_corrected same as data_xy here since intensity is already baseline-relative).

    When ``missing`` is provided, short-gap placeholders are skipped and long x-gaps
    emit NaN intensity samples so consumers do not invent long diagonals.
    """
    x_min = float(calibration.x_min)
    x_max = float(calibration.x_max)
    p0 = float(calibration.plot_left)
    p1 = float(calibration.plot_right)
    span = max(1e-9, p1 - p0)
    raw: list[list[float]] = []
    data: list[list[float]] = []
    prev_x: int | None = None
    for index, (x_px, y_px) in enumerate(path):
        is_miss = bool(missing[index]) if missing is not None and index < len(missing) else False
        if prev_x is not None and (x_px - prev_x) > max_gap_px:
            gap_theta = x_min + (float(prev_x + 1) - p0) / span * (x_max - x_min)
            raw.append([float(prev_x + 1), float("nan")])
            data.append([float(gap_theta), float("nan")])
        if is_miss:
            prev_x = int(x_px)
            continue
        theta = x_min + (float(x_px) - p0) / span * (x_max - x_min)
        intensity = float(baseline_px) - float(y_px)
        raw.append([float(x_px), float(y_px)])
        data.append([float(theta), float(intensity)])
        prev_x = int(x_px)
    return raw, data, list(data)


def high_confidence_trace(quality: TraceQuality) -> bool:
    return (
        quality.confidence >= 0.50
        and quality.coverage >= 0.55
        and "long_gap" not in quality.flags
    )


def column_trace_curve(
    image_bgr: np.ndarray,
    *,
    baseline_px: float,
    search_top: int,
    search_bottom: int,
    x0: int,
    x1: int,
    annotation_mask: np.ndarray | None = None,
    max_gap: int = 28,
    jump_hard: int = 45,
    blank_margin: int = 6,
    ink_threshold: int = 245,
    baseline_attach_px: int = 28,
) -> tuple[list[tuple[int, int]], list[bool], TraceQuality]:
    """
    Trace a curve by picking the peak tip of the ink run attached to baseline.

    Works for anti-aliased black XRD lines where Viterbi ink coverage is too
    sparse for continuous path DP. Floating Miller-index glyphs are ignored
    unless they connect down toward the baseline.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    top = max(0, int(search_top))
    bottom = min(height, int(search_bottom))
    left = max(0, int(x0))
    right = min(width, int(x1))
    baseline = int(np.clip(round(baseline_px), top, max(top, bottom - 1)))
    if bottom - top < 4 or right - left < 4:
        empty_q = TraceQuality(
            coverage=0.0,
            longest_gap_px=0,
            jump_count=0,
            median_baseline_dev_px=0.0,
            confidence=0.0,
            flags=["empty_search_region"],
        )
        return [], [], empty_q

    path: list[tuple[int, int]] = []
    missing: list[bool] = []
    prev_y: int | None = None
    jump_count = 0
    longest = 0
    gap = 0
    observed = 0

    for x in range(left, right):
        col = gray[top:bottom, x].astype(np.float32)
        if annotation_mask is not None and annotation_mask.shape[:2] == gray.shape:
            ann = annotation_mask[top:bottom, x]
            if ann.any():
                col = col.copy()
                col[ann] = np.minimum(255.0, col[ann] + 45.0)

        dark = col < float(ink_threshold)
        if not dark.any() or float(255.0 - float(col.min())) < blank_margin:
            gap += 1
            longest = max(longest, gap)
            if gap <= max_gap and prev_y is not None:
                path.append((x, int(prev_y)))
                missing.append(True)
            continue

        base_local = int(np.clip(baseline - top, 0, len(col) - 1))
        tip_local = base_local
        white_gap = 0
        seen_ink = False
        max_white_gap = 12
        for loc in range(base_local, -1, -1):
            if dark[loc]:
                seen_ink = True
                tip_local = loc
                white_gap = 0
            else:
                if not seen_ink:
                    if base_local - loc > baseline_attach_px:
                        break
                    continue
                white_gap += 1
                if white_gap > max_white_gap:
                    break
        if not seen_ink:
            lo = max(0, base_local - max(30, (base_local + 1) // 2))
            seg = col[lo : base_local + 1]
            if seg.size and float(255.0 - float(np.min(seg))) >= blank_margin:
                tip_local = lo + int(np.argmin(seg))
                seen_ink = True
            else:
                gap += 1
                longest = max(longest, gap)
                if gap <= max_gap and prev_y is not None:
                    path.append((x, int(prev_y)))
                    missing.append(True)
                continue
        y = top + int(tip_local)

        if prev_y is not None and abs(y - prev_y) > jump_hard:
            y0 = max(top, prev_y - jump_hard)
            y1 = min(bottom, max(prev_y + 4, baseline + 2))
            seg = col[y0 - top : y1 - top]
            seg_dark = dark[y0 - top : y1 - top]
            if seg_dark.any():
                y = y0 + int(np.flatnonzero(seg_dark)[0])
                jump_count += 1
            elif seg.size and float(255.0 - float(np.min(seg))) >= blank_margin:
                y = y0 + int(np.argmin(seg))
                jump_count += 1

        gap = 0
        path.append((x, int(y)))
        missing.append(False)
        observed += 1
        if prev_y is not None and abs(y - prev_y) > jump_hard:
            jump_count += 1
        prev_y = int(y)

    cleaned: list[tuple[int, int]] = []
    cleaned_miss: list[bool] = []
    run = 0
    for (x, y), is_miss in zip(path, missing):
        if is_miss:
            run += 1
            if run > max_gap:
                continue
        else:
            run = 0
        cleaned.append((x, y))
        cleaned_miss.append(is_miss)

    span = max(1, right - left)
    coverage = observed / float(span)
    flags: list[str] = []
    if coverage < 0.35:
        flags.append("low_coverage")
    if longest > max_gap:
        flags.append("long_gap")
    if jump_count > max(5, span // 60):
        flags.append("many_jumps")

    peak_follow = 0.0
    amps = [
        max(0.0, float(baseline_px) - float(y))
        for (x, y), m in zip(cleaned, cleaned_miss)
        if not m
    ]
    if amps and max(amps) > 15:
        peak_follow = min(1.0, float(np.percentile(amps, 90)) / 100.0)

    confidence = float(
        np.clip(
            0.45 * coverage
            + 0.20 * (1.0 - min(1.0, longest / max(1.0, 2 * max_gap)))
            + 0.15 * (1.0 - min(1.0, jump_count / 25.0))
            + 0.20 * peak_follow,
            0.0,
            1.0,
        )
    )
    if flags:
        confidence *= 0.8

    quality = TraceQuality(
        coverage=float(coverage),
        longest_gap_px=int(longest),
        jump_count=int(jump_count),
        median_baseline_dev_px=float(np.median(amps) if amps else 0.0),
        confidence=float(confidence),
        flags=flags,
    )
    return cleaned, cleaned_miss, quality


def usable_column_trace(
    quality: TraceQuality,
    *,
    n_points: int,
) -> bool:
    """Accept sparse anti-aliased black XRD traces that Viterbi under-covers."""
    if n_points < 35:
        return False
    if quality.coverage >= 0.22 and quality.confidence >= 0.20:
        return True
    if quality.coverage >= 0.12 and n_points >= 80 and "empty_search_region" not in quality.flags:
        return True
    if n_points >= 100 and quality.coverage >= 0.10:
        return True
    return False
