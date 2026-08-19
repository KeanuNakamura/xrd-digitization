"""Global continuous XRD path tracing (DP / shortest-path).

Replaces independent per-column selection with one y-value per x-column
minimizing a cost that includes darkness, jump, curvature, text-mask, axis,
and optional agent-prior terms.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from xrd_digitization.types import AxisCalibrationResult, CurveData

LOGGER = logging.getLogger(__name__)

CURVE_INK_THRESH = 170
MAX_JUMP_PX = 120
JUMP_COST = 0.18
CURVATURE_COST = 0.04
TEXT_PENALTY = 14.0
SOFT_BOX_PENALTY = 1.2
AXIS_PENALTY = 50.0
PRIOR_PENALTY = 0.045
BASELINE_BAND_HALF = 36
PEAK_BAND_HALF = 400
PEAK_WINDOW_DEG = 1.15
PEAK_JUMP_DISCOUNT = 0.22
PEAK_SEARCH_RADIUS = 420
PEAK_APEX_WINDOW_PX = 14


def build_vertical_text_penalty(
    gray: np.ndarray,
    text_mask: np.ndarray | None,
    *,
    min_run: int = 8,
) -> np.ndarray:
    """
    Hard penalty only on tall vertical dark runs inside annotation boxes
    (glyph stems), not on thin curve strokes crossing the label.
    """
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    if text_mask is None or not np.any(text_mask):
        return out
    dark = gray < CURVE_INK_THRESH

    def _horiz_curve_linked(x: int, y: int) -> bool:
        for dx in (-3, -2, -1, 1, 2, 3):
            xx = x + dx
            if not (0 <= xx < w):
                continue
            y0 = max(0, y - 2)
            y1 = min(h, y + 3)
            if np.any(dark[y0:y1, xx]):
                return True
        return False

    ys, xs = np.where(text_mask > 0)
    if len(xs) == 0:
        return out
    # Process unique x columns that intersect the text mask.
    for x in np.unique(xs):
        col_ys = ys[xs == x]
        y = int(col_ys.min())
        y_end = int(col_ys.max()) + 1
        while y < y_end:
            if not dark[y, x] or text_mask[y, x] == 0:
                y += 1
                continue
            yb = y
            while yb < y_end and dark[yb, x] and text_mask[yb, x] > 0:
                yb += 1
            run = yb - y
            if run >= min_run:
                for yy in range(y, yb):
                    if not _horiz_curve_linked(x, yy):
                        out[yy, x] = 255
            y = yb
    return out


def detect_axis_mask(
    gray: np.ndarray,
    *,
    plot_left: int | None = None,
    plot_right: int | None = None,
    plot_top: int | None = None,
    plot_bottom: int | None = None,
    border_inset: int = 4,
) -> np.ndarray:
    """
    Detect and return a mask of axis lines, ticks, and plot borders.

    Suppresses bottom x-axis, left y-axis, tick marks, and top/right borders
    so the path tracer cannot stay on the frame.
    """
    h, w = gray.shape
    x0 = 0 if plot_left is None else max(0, int(plot_left))
    x1 = w if plot_right is None else min(w, int(plot_right))
    y0 = 0 if plot_top is None else max(0, int(plot_top))
    y1 = h if plot_bottom is None else min(h, int(plot_bottom))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.zeros((h, w), dtype=np.uint8)

    dark = (gray < CURVE_INK_THRESH).astype(np.uint8) * 255
    axis = np.zeros((h, w), dtype=np.uint8)

    # Long morphological axes.
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, (y1 - y0) // 8)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, (x1 - x0) // 8), 1))
    v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
    h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
    axis = cv2.bitwise_or(axis, cv2.bitwise_or(v_lines, h_lines))

    # Explicit border bands inside the plot rectangle.
    inset = max(2, int(border_inset))
    axis[y0 : min(y1, y0 + inset), x0:x1] = 255
    axis[max(y0, y1 - inset) : y1, x0:x1] = 255
    axis[y0:y1, x0 : min(x1, x0 + inset)] = 255
    axis[y0:y1, max(x0, x1 - inset) : x1] = 255

    # Tick marks: short inward dark runs from borders.
    tick_len = max(4, min(14, (y1 - y0) // 40))
    # Bottom ticks
    for x in range(x0, x1):
        col = dark[max(y0, y1 - tick_len - 2) : y1, x]
        if col.size and np.count_nonzero(col) >= 2:
            axis[max(y0, y1 - tick_len - 2) : y1, x] = 255
    # Top ticks
    for x in range(x0, x1):
        col = dark[y0 : min(y1, y0 + tick_len + 2), x]
        if col.size and np.count_nonzero(col) >= 2:
            axis[y0 : min(y1, y0 + tick_len + 2), x] = 255
    # Left / right ticks
    for y in range(y0, y1):
        row_l = dark[y, x0 : min(x1, x0 + tick_len + 2)]
        if row_l.size and np.count_nonzero(row_l) >= 2:
            axis[y, x0 : min(x1, x0 + tick_len + 2)] = 255
        row_r = dark[y, max(x0, x1 - tick_len - 2) : x1]
        if row_r.size and np.count_nonzero(row_r) >= 2:
            axis[y, max(x0, x1 - tick_len - 2) : x1] = 255

    return axis


def estimate_baseline_y(
    gray: np.ndarray,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    axis_mask: np.ndarray | None = None,
    text_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate a smooth baseline / background envelope (image y, top=0).

    Uses per-column lower-envelope dark pixels (near bottom) with heavy
    smoothing so sharp peaks do not dominate the baseline.
    """
    h, w = gray.shape
    x0, x1 = max(0, plot_left), min(w, plot_right)
    y0, y1 = max(0, plot_top), min(h, plot_bottom)
    baseline = np.full(w, np.nan, dtype=np.float32)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return baseline

    axis = axis_mask if axis_mask is not None else np.zeros_like(gray)
    text = text_mask if text_mask is not None else np.zeros_like(gray)

    samples: list[tuple[int, float]] = []
    for x in range(x0, x1):
        col = gray[y0:y1, x]
        usable = (col < CURVE_INK_THRESH) & (axis[y0:y1, x] == 0)
        if text is not None:
            usable = usable & (text[y0:y1, x] == 0)
        ys = np.where(usable)[0]
        if len(ys) == 0:
            continue
        # Lower envelope in image coords = largest y (closest to bottom).
        # Exclude the very bottom border band.
        ys = ys[ys < (y1 - y0 - 6)]
        if len(ys) == 0:
            continue
        # Robust lower envelope: high percentile of dark y positions.
        y_env = float(np.percentile(ys, 90)) + y0
        samples.append((x, y_env))

    if len(samples) < 8:
        # Fallback: constant near bottom.
        fallback = float(y1 - max(12, (y1 - y0) * 0.08))
        baseline[x0:x1] = fallback
        return baseline

    xs = np.array([s[0] for s in samples], dtype=float)
    ys = np.array([s[1] for s in samples], dtype=float)
    # Rolling median then light poly trend.
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    win = max(9, len(ys) // 40)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp = np.pad(ys, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    ys_s = np.convolve(yp, kernel, mode="valid")
    grid = np.arange(x0, x1, dtype=float)
    baseline[x0:x1] = np.interp(grid, xs, ys_s).astype(np.float32)
    return baseline


def _agent_prior_y_pixels(
    agent_prior_intensity: np.ndarray | None,
    agent_prior_two_theta: np.ndarray | None,
    calibration: AxisCalibrationResult,
    width: int,
) -> np.ndarray | None:
    if (
        agent_prior_intensity is None
        or agent_prior_two_theta is None
        or len(agent_prior_intensity) < 2
    ):
        return None
    span_x = max(calibration.x_max - calibration.x_min, 1e-9)
    xs = np.arange(width, dtype=float)
    # Map pixel x → two_theta using calibration (crop-local).
    tt = calibration.x_min + (xs - calibration.plot_left) / max(
        calibration.plot_right - calibration.plot_left, 1
    ) * span_x
    inten = np.interp(
        tt,
        np.asarray(agent_prior_two_theta, dtype=float),
        np.asarray(agent_prior_intensity, dtype=float),
        left=np.nan,
        right=np.nan,
    )
    top, bottom = calibration.plot_top, calibration.plot_bottom
    span = max(bottom - top, 1)
    if calibration.has_y_calibration and calibration.y_min is not None and calibration.y_max is not None:
        y_span = max(calibration.y_max - calibration.y_min, 1e-9)
        frac = (inten - calibration.y_min) / y_span
    else:
        max_i = float(np.nanmax(inten)) if np.any(np.isfinite(inten)) else 1.0
        max_i = max(max_i, 1e-9)
        frac = inten / max_i
    frac = np.clip(frac, 0.0, 1.0)
    y = bottom - frac * span
    y[~np.isfinite(inten)] = np.nan
    return y.astype(np.float32)


def build_search_band(
    baseline_y: np.ndarray,
    *,
    plot_top: int,
    plot_bottom: int,
    half_width: int = BASELINE_BAND_HALF,
    peak_half_width: int = PEAK_BAND_HALF,
    peak_columns: np.ndarray | None = None,
    prev_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (y_lo, y_hi) search limits around baseline / previous path."""
    w = baseline_y.shape[0]
    y_lo = np.full(w, plot_top, dtype=np.int32)
    y_hi = np.full(w, plot_bottom, dtype=np.int32)
    for x in range(w):
        center = None
        if prev_path is not None and np.isfinite(prev_path[x]):
            center = float(prev_path[x])
        elif np.isfinite(baseline_y[x]):
            center = float(baseline_y[x])
        if center is None:
            continue
        half = peak_half_width if (peak_columns is not None and peak_columns[x]) else half_width
        # Allow upward (smaller y) travel for peaks more than downward.
        lo = int(max(plot_top, center - half))
        hi = int(min(plot_bottom, center + max(8, half // 3)))
        y_lo[x] = lo
        y_hi[x] = max(lo + 1, hi)
    return y_lo, y_hi


def trace_curve_dp(
    gray: np.ndarray,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    text_mask: np.ndarray | None = None,
    axis_mask: np.ndarray | None = None,
    prior_y: np.ndarray | None = None,
    baseline_y: np.ndarray | None = None,
    peak_two_thetas: Sequence[float] | None = None,
    calibration: AxisCalibrationResult | None = None,
    max_jump: int = MAX_JUMP_PX,
) -> np.ndarray:
    """
    Global DP path: one continuous y per x-column.

    Cost = pixel darkness + vertical jump + curvature + text + axis
    + optional distance-to-agent-prior.
    """
    h, w = gray.shape
    x0, x1 = max(0, int(plot_left)), min(w, int(plot_right))
    y0, y1 = max(0, int(plot_top)), min(h, int(plot_bottom))
    curve = np.full(w, np.nan, dtype=np.float32)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return curve

    if axis_mask is None:
        axis_mask = detect_axis_mask(
            gray,
            plot_left=x0,
            plot_right=x1,
            plot_top=y0,
            plot_bottom=y1,
        )
    if baseline_y is None:
        baseline_y = estimate_baseline_y(
            gray,
            plot_left=x0,
            plot_right=x1,
            plot_top=y0,
            plot_bottom=y1,
            axis_mask=axis_mask,
            text_mask=text_mask,
        )

    peak_cols = np.zeros(w, dtype=bool)
    peak_centers: list[int] = []
    if peak_two_thetas and calibration is not None:
        span = max(calibration.plot_right - calibration.plot_left, 1)
        deg_span = max(calibration.x_max - calibration.x_min, 1e-9)
        px_per_deg = span / deg_span
        half_px = max(4, int(round(PEAK_WINDOW_DEG * px_per_deg)))
        for tt in peak_two_thetas:
            frac = (float(tt) - calibration.x_min) / deg_span
            px = int(round(calibration.plot_left + frac * span))
            peak_centers.append(px)
            for dx in range(-half_px, half_px + 1):
                xx = px + dx
                if 0 <= xx < w:
                    peak_cols[xx] = True

    # Near probable peaks, open the band from plot top so tall XRD spikes are reachable.
    y_lo_arr, y_hi_arr = build_search_band(
        baseline_y,
        plot_top=y0,
        plot_bottom=y1,
        half_width=BASELINE_BAND_HALF,
        peak_half_width=PEAK_SEARCH_RADIUS,
        peak_columns=peak_cols,
    )
    for x in range(w):
        if peak_cols[x]:
            y_lo_arr[x] = y0
            if np.isfinite(baseline_y[x]):
                y_hi_arr[x] = int(min(y1, float(baseline_y[x]) + 12))
            else:
                y_hi_arr[x] = y1

    # First pass: full-band DP with axis/text penalties (band restricts unary).
    roi_h = y1 - y0
    rw = x1 - x0
    unary = np.full((roi_h, rw), 1e3, dtype=np.float32)
    darkness = np.clip((220.0 - gray[y0:y1, x0:x1].astype(np.float32)) / 220.0, 0.0, 1.0)

    for xi in range(rw):
        x = x0 + xi
        lo = max(0, int(y_lo_arr[x]) - y0)
        hi = min(roi_h, int(y_hi_arr[x]) - y0)
        if hi <= lo:
            lo, hi = 0, roi_h
        # Wider band near peaks already encoded in y_lo/y_hi.
        for yi in range(lo, hi):
            unary[yi, xi] = 6.0 * (1.0 - darkness[yi, xi])
            # Prefer near baseline when not dark (keeps path off empty sky).
            if darkness[yi, xi] < 0.15 and np.isfinite(baseline_y[x]):
                unary[yi, xi] += 0.8 * abs((y0 + yi) - baseline_y[x]) / max(roi_h, 1)

    # Soft penalty on whole text boxes; hard penalty only on vertical glyph stems.
    # Blanket hard penalties prevent the path from climbing peak tips under labels.
    if text_mask is not None:
        soft = text_mask[y0:y1, x0:x1] > 0
        unary = unary + SOFT_BOX_PENALTY * soft.astype(np.float32)
        stem = build_vertical_text_penalty(gray, text_mask)
        unary = unary + TEXT_PENALTY * (stem[y0:y1, x0:x1] > 0).astype(np.float32)
    ap = axis_mask[y0:y1, x0:x1] > 0
    unary = unary + AXIS_PENALTY * ap.astype(np.float32)

    # Strong border avoidance (never ride the bottom axis).
    inset = max(3, min(10, roi_h // 30))
    unary[:inset, :] += AXIS_PENALTY
    unary[-inset:, :] += AXIS_PENALTY * 1.5
    unary[:, : max(1, inset // 2)] += AXIS_PENALTY
    unary[:, -max(1, inset // 2) :] += AXIS_PENALTY

    if prior_y is not None:
        for xi in range(rw):
            py = prior_y[x0 + xi]
            if not np.isfinite(py):
                continue
            for yi in range(roi_h):
                unary[yi, xi] += PRIOR_PENALTY * abs((y0 + yi) - py)

    # Soft attraction to darkest pixel in each column (non-axis).
    # Near agent peak columns, prefer the uppermost strong dark tip.
    stem_mask = (
        build_vertical_text_penalty(gray, text_mask)
        if text_mask is not None
        else np.zeros_like(gray)
    )
    for xi in range(rw):
        x = x0 + xi
        col = darkness[:, xi].copy()
        col[:inset] = 0
        col[-inset:] = 0
        col = col * (1.0 - 0.95 * (axis_mask[y0:y1, x] > 0).astype(np.float32))
        col = col * (1.0 - 0.9 * (stem_mask[y0:y1, x] > 0).astype(np.float32))
        if float(col.max()) < 0.08:
            continue
        if peak_cols[x]:
            # Uppermost strong local maximum (peak tip), not baseline ink.
            thresh = max(0.4, float(col.max()) * 0.55)
            candidates = np.where(col >= thresh)[0]
            peak_y = int(candidates.min()) if candidates.size else int(np.argmax(col))
            attract = 3.2
            radius = 14
        else:
            peak_y = int(np.argmax(col))
            attract = 1.8
            radius = 8
        for yi in range(roi_h):
            dist = abs(yi - peak_y)
            if dist <= radius:
                unary[yi, xi] -= attract * max(0.0, 1.0 - dist / radius) * col[peak_y]

    jump = max(1, int(max_jump))
    dp = np.full((roi_h, rw), np.inf, dtype=np.float32)
    back = np.full((roi_h, rw), -1, dtype=np.int32)
    # Seed only within the first-column search band.
    lo0 = max(0, int(y_lo_arr[x0]) - y0)
    hi0 = min(roi_h, int(y_hi_arr[x0]) - y0)
    dp[lo0:hi0, 0] = unary[lo0:hi0, 0]

    for xi in range(1, rw):
        prev = dp[:, xi - 1]
        x = x0 + xi
        near_peak = bool(peak_cols[x] or (x > 0 and peak_cols[x - 1]) or (x + 1 < w and peak_cols[x + 1]))
        local_jump = int(jump * 2.5) if near_peak else jump
        # Allow rapid upward/downward movement at narrow peaks.
        jump_pen = JUMP_COST * PEAK_JUMP_DISCOUNT if near_peak else JUMP_COST
        curv_pen = CURVATURE_COST * (0.25 if near_peak else 1.0)
        lo = max(0, int(y_lo_arr[x]) - y0)
        hi = min(roi_h, int(y_hi_arr[x]) - y0)
        if hi <= lo:
            lo, hi = 0, roi_h
        for yi in range(lo, hi):
            y_a = max(0, yi - local_jump)
            y_b = min(roi_h, yi + local_jump + 1)
            # Vectorized transition over the jump window.
            pys = np.arange(y_a, y_b)
            prev_vals = prev[pys]
            valid = np.isfinite(prev_vals)
            if not np.any(valid):
                continue
            pys = pys[valid]
            prev_vals = prev_vals[valid]
            dy = np.abs(pys - yi).astype(np.float32)
            # Upward (smaller image-y / higher intensity) is cheaper near peaks.
            up_scale = 0.35 if near_peak else 0.55
            down_scale = 0.55 if near_peak else 1.35
            jump_term = np.where(yi < pys, jump_pen * up_scale * dy, jump_pen * down_scale * dy)
            curv = np.where(dy >= 2, curv_pen * (dy - 1), 0.0)
            costs = prev_vals + jump_term + curv
            best_i = int(np.argmin(costs))
            dp[yi, xi] = float(costs[best_i]) + unary[yi, xi]
            back[yi, xi] = int(pys[best_i])

    # Prefer ending near baseline, not on bottom border.
    end_costs = dp[:, -1].copy()
    end_costs[-inset:] += 100.0
    y = int(np.argmin(end_costs))
    for xi in range(rw - 1, -1, -1):
        curve[x0 + xi] = float(y0 + y)
        if xi == 0:
            break
        y = int(back[y, xi])
        if y < 0:
            break

    # Second pass: tighten band around first path (continuity), but keep
    # peak columns fully open so apex height is not clipped.
    y_lo2, y_hi2 = build_search_band(
        baseline_y,
        plot_top=y0,
        plot_bottom=y1,
        half_width=max(12, BASELINE_BAND_HALF // 2),
        peak_half_width=PEAK_SEARCH_RADIUS,
        peak_columns=peak_cols,
        prev_path=curve,
    )
    for x in range(w):
        if peak_cols[x]:
            y_lo2[x] = y0
            if np.isfinite(baseline_y[x]):
                y_hi2[x] = int(min(y1, float(baseline_y[x]) + 12))

    unary2 = np.full((roi_h, rw), 1e3, dtype=np.float32)
    for xi in range(rw):
        x = x0 + xi
        lo = max(0, int(y_lo2[x]) - y0)
        hi = min(roi_h, int(y_hi2[x]) - y0)
        if hi <= lo:
            continue
        unary2[lo:hi, xi] = unary[lo:hi, xi]
        # Extra continuity bonus near previous path (weaker near peaks).
        if np.isfinite(curve[x]):
            cy = int(round(curve[x])) - y0
            cont = 0.04 if peak_cols[x] else 0.15
            for yi in range(lo, hi):
                unary2[yi, xi] += cont * abs(yi - cy)

    dp2 = np.full((roi_h, rw), np.inf, dtype=np.float32)
    back2 = np.full((roi_h, rw), -1, dtype=np.int32)
    dp2[:, 0] = unary2[:, 0]
    for xi in range(1, rw):
        prev = dp2[:, xi - 1]
        x = x0 + xi
        near_peak = bool(peak_cols[x])
        local_jump = int(jump * 2.5) if near_peak else jump
        jump_pen = JUMP_COST * PEAK_JUMP_DISCOUNT if near_peak else JUMP_COST
        curv_pen = CURVATURE_COST * (0.25 if near_peak else 1.0)
        for yi in range(roi_h):
            if unary2[yi, xi] >= 500:
                continue
            y_a = max(0, yi - local_jump)
            y_b = min(roi_h, yi + local_jump + 1)
            best = np.inf
            best_y = yi
            for py in range(y_a, y_b):
                if not np.isfinite(prev[py]):
                    continue
                dy = abs(py - yi)
                up = yi < py
                scale = (0.35 if near_peak else 0.55) if up else (0.55 if near_peak else 1.0)
                cost = prev[py] + jump_pen * scale * dy
                if dy >= 2:
                    cost += curv_pen * (dy - 1)
                if cost < best:
                    best = cost
                    best_y = py
            if np.isfinite(best):
                dp2[yi, xi] = best + unary2[yi, xi]
                back2[yi, xi] = best_y

    if np.any(np.isfinite(dp2[:, -1])):
        end_costs = dp2[:, -1].copy()
        end_costs[~np.isfinite(end_costs)] = np.inf
        end_costs[-inset:] += 100.0
        y = int(np.argmin(end_costs))
        if np.isfinite(end_costs[y]):
            for xi in range(rw - 1, -1, -1):
                curve[x0 + xi] = float(y0 + y)
                if xi == 0:
                    break
                y = int(back2[y, xi])
                if y < 0:
                    break

    # Peak-apex recovery: do not smooth/clip before preserving full peak height.
    if peak_centers:
        curve = recover_peak_apexes(
            gray,
            curve,
            peak_centers,
            plot_left=x0,
            plot_right=x1,
            plot_top=y0,
            plot_bottom=y1,
            text_mask=text_mask,
            axis_mask=axis_mask,
            window_px=PEAK_APEX_WINDOW_PX,
        )

    return curve


def recover_peak_apexes(
    gray: np.ndarray,
    curve_y: np.ndarray,
    peak_centers_x: Sequence[int],
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    text_mask: np.ndarray | None = None,
    axis_mask: np.ndarray | None = None,
    window_px: int = PEAK_APEX_WINDOW_PX,
) -> np.ndarray:
    """
    After an initial DP path, recover tall narrow peaks that were flattened.

    For each expected peak center, inspect a local x-window, find the highest
    connected curve ink, and replace the local path with the ascending edge,
    apex, and descending edge when the image clearly contains a taller peak.
    """
    h, w = gray.shape
    out = curve_y.astype(np.float32).copy()
    stem = (
        build_vertical_text_penalty(gray, text_mask)
        if text_mask is not None
        else np.zeros((h, w), dtype=np.uint8)
    )

    for cx in peak_centers_x:
        x0 = max(plot_left, int(cx) - int(window_px))
        x1 = min(plot_right, int(cx) + int(window_px) + 1)
        if x1 - x0 < 3:
            continue

        tip_ys = np.full(x1 - x0, np.nan, dtype=np.float32)
        tip_scores = np.zeros(x1 - x0, dtype=np.float32)
        for i, x in enumerate(range(x0, x1)):
            path_y = out[x] if np.isfinite(out[x]) else float(plot_bottom - 4)
            top = plot_top + 2
            bot = min(plot_bottom - 2, int(path_y) + 14)
            if bot - top < 8:
                continue
            col = gray[top:bot, x].astype(np.float32)
            score = np.clip((200.0 - col) / 200.0, 0.0, 1.0)
            score = score * (1.0 - 0.95 * (stem[top:bot, x] > 0).astype(np.float32))
            if axis_mask is not None:
                score = score * (
                    1.0 - 0.95 * (axis_mask[top:bot, x] > 0).astype(np.float32)
                )
            if text_mask is not None:
                score = score * (
                    1.0 - 0.45 * (text_mask[top:bot, x] > 0).astype(np.float32)
                )
            if float(score.max()) < 0.35:
                continue
            # Uppermost strong tip in this column.
            thresh = max(0.4, float(score.max()) * 0.55)
            candidates = np.where(score >= thresh)[0]
            if candidates.size == 0:
                continue
            tip = int(candidates.min())
            # Require a short connected dark run (curve stroke, not noise).
            run = 1
            yy = tip + 1
            while yy < score.size and score[yy] > 0.25:
                run += 1
                yy += 1
            if run < 1:
                continue
            tip_ys[i] = float(top + tip)
            tip_scores[i] = float(score[tip])

        valid = np.isfinite(tip_ys)
        if int(valid.sum()) < 2:
            continue

        # Apex = highest tip (smallest image y) among strong columns.
        strong = valid & (tip_scores >= 0.45)
        if not np.any(strong):
            strong = valid
        apex_i = int(np.argmin(np.where(strong, tip_ys, np.inf)))
        apex_y = float(tip_ys[apex_i])
        apex_x = x0 + apex_i

        path_y = float(out[apex_x]) if np.isfinite(out[apex_x]) else float(plot_bottom)
        # Only replace when the image apex is clearly taller than the path.
        if path_y - apex_y < 12:
            continue

        # Build ascending / apex / descending edges from connected tips.
        # Walk left from apex while tips climb or stay near the ridge.
        left = apex_i
        while left > 0 and valid[left - 1]:
            if tip_ys[left - 1] > tip_ys[left] + 18:
                break
            left -= 1
        right = apex_i
        while right + 1 < len(tip_ys) and valid[right + 1]:
            if tip_ys[right + 1] > tip_ys[right] + 18:
                break
            right += 1

        # Replace flattened local path with the tip envelope (no smoothing).
        for i in range(left, right + 1):
            if not valid[i]:
                continue
            x = x0 + i
            tip = float(tip_ys[i])
            if np.isfinite(out[x]) and tip < out[x] - 2:
                out[x] = tip
            elif not np.isfinite(out[x]):
                out[x] = tip

        # Ensure the apex column keeps the absolute tip.
        out[apex_x] = min(float(out[apex_x]) if np.isfinite(out[apex_x]) else apex_y, apex_y)

        # Linearly connect immediate neighbors if missing so the peak is sharp,
        # not a plateau — pull them toward the apex without widening.
        for dx in (-1, 1):
            xx = apex_x + dx
            if plot_left <= xx < plot_right and np.isfinite(out[xx]):
                # Keep neighbor between apex and its previous value (sharp sides).
                out[xx] = 0.25 * out[xx] + 0.75 * apex_y

    return out


def refine_curve_y_with_column_peaks(
    gray: np.ndarray,
    curve_y: np.ndarray,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    text_mask: np.ndarray | None = None,
    peak_columns: np.ndarray | None = None,
    axis_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Snap the DP path onto strong non-text dark peak tips in each column."""
    h, w = gray.shape
    out = curve_y.astype(np.float32).copy()
    stem = (
        build_vertical_text_penalty(gray, text_mask)
        if text_mask is not None
        else np.zeros((h, w), dtype=np.uint8)
    )
    candidates: list[tuple[int, float, float]] = []  # x, y_peak, score
    for x in range(max(0, plot_left), min(w, plot_right)):
        y = out[x]
        if not np.isfinite(y):
            continue
        yi = int(round(float(y)))
        top = plot_top + 2
        bot = min(plot_bottom - 2, yi + 10)
        if bot - top < 8:
            continue
        col = gray[top:bot, x].astype(np.float32)
        score = np.clip((200.0 - col) / 200.0, 0.0, 1.0)
        score = score * (1.0 - 0.95 * (stem[top:bot, x] > 0).astype(np.float32))
        if axis_mask is not None:
            score = score * (1.0 - 0.95 * (axis_mask[top:bot, x] > 0).astype(np.float32))
        if text_mask is not None:
            # Softly discount text-box interiors so glyph ink is not a "tip".
            score = score * (1.0 - 0.5 * (text_mask[top:bot, x] > 0).astype(np.float32))
        if float(score.max()) < 0.35:
            continue
        peak = int(np.argmax(score))
        for _ in range(6):
            above = score[: max(1, peak - 2)]
            if above.size and float(above.max()) >= max(0.45, float(score[peak]) * 0.75):
                peak = int(np.argmax(above))
            else:
                break
        y_peak = float(top + peak)
        run_lo, run_hi = peak, peak
        while run_lo > 0 and score[run_lo - 1] > 0.3:
            run_lo -= 1
        while run_hi + 1 < score.size and score[run_hi + 1] > 0.3:
            run_hi += 1
        if (run_hi - run_lo + 1) > 12 and not (peak_columns is not None and peak_columns[x]):
            continue
        if y_peak < y - 4 and score[peak] >= 0.4:
            candidates.append((x, y_peak, float(score[peak])))
        elif peak_columns is not None and peak_columns[x] and y_peak < y - 2:
            candidates.append((x, y_peak, float(score[peak])))

    if not candidates:
        return out

    # Keep only local maxima along x so a horizontal shelf cannot flatten a peak.
    cand_x = {c[0]: c for c in candidates}
    accepted: list[tuple[int, float, float]] = []
    for x, y_peak, sc in candidates:
        left = cand_x.get(x - 1)
        right = cand_x.get(x + 1)
        is_ridge = True
        if left is not None and left[1] < y_peak - 1:
            is_ridge = False
        if right is not None and right[1] < y_peak - 1:
            is_ridge = False
        near_prior = peak_columns is not None and bool(
            np.any(peak_columns[max(0, x - 2) : x + 3])
        )
        if is_ridge or (near_prior and sc >= 0.55):
            accepted.append((x, y_peak, sc))

    # Per local cluster, keep only the single sharpest tip (±4 px).
    accepted.sort(key=lambda t: t[0])
    i = 0
    while i < len(accepted):
        j = i + 1
        while j < len(accepted) and accepted[j][0] - accepted[i][0] <= 4:
            j += 1
        cluster = accepted[i:j]
        best = min(cluster, key=lambda t: (t[1], -t[2]))  # highest tip, then score
        bx, by, _ = best
        out[bx] = by
        # Softly pull immediate neighbors toward the tip (1 px each side).
        for dx in (-1, 1):
            xx = bx + dx
            if 0 <= xx < w and np.isfinite(out[xx]):
                out[xx] = 0.35 * out[xx] + 0.65 * by
        i = j
    return out


def curve_y_to_curve_data(
    curve_y: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
    source: str = "dp_trace",
    curve_id: str = "curve_1",
) -> CurveData:
    """Convert a pixel path ``curve_y[x]`` into calibrated ``CurveData``."""
    w = curve_y.shape[0]
    x0 = max(0, calibration.plot_left)
    x1 = min(w, calibration.plot_right)
    xs = np.arange(x0, x1, dtype=float)
    ys = curve_y[x0:x1]
    valid = np.isfinite(ys)
    if valid.sum() < 5:
        return CurveData(two_theta=[], intensity=[], warnings=["dp_trace_empty"], source=source)

    xs_v = xs[valid]
    ys_v = ys[valid]

    span_x = max(calibration.plot_right - calibration.plot_left, 1)
    two_theta = calibration.x_min + (xs_v - calibration.plot_left) / span_x * (
        calibration.x_max - calibration.x_min
    )

    top, bottom = calibration.plot_top, calibration.plot_bottom
    span_y = max(bottom - top, 1)
    if calibration.has_y_calibration and calibration.y_min is not None and calibration.y_max is not None:
        frac = (bottom - ys_v) / span_y
        intensity = calibration.y_min + frac * (calibration.y_max - calibration.y_min)
    else:
        frac = (bottom - ys_v) / span_y
        intensity = np.clip(frac, 0.0, None)
        peak = float(np.percentile(intensity, 99)) if intensity.size else 1.0
        peak = max(peak, 1e-9)
        intensity = intensity / peak * 100.0

    # Resample uniform in 2θ without smoothing (preserve noise).
    order = np.argsort(two_theta)
    x = two_theta[order]
    y = intensity[order]
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    if len(x_u) < 2:
        return CurveData(
            two_theta=x_u.tolist(),
            intensity=y_u.tolist(),
            curve_id=curve_id,
            source=source,
            warnings=["dp_trace"],
        )
    grid = np.linspace(float(x_u.min()), float(x_u.max()), num_points)
    y_grid = np.interp(grid, x_u, y_u)
    return CurveData(
        two_theta=grid.tolist(),
        intensity=y_grid.tolist(),
        curve_id=curve_id,
        source=source,
        warnings=["dp_trace", "black_curve_extraction"],
    )


def digitize_via_global_trace(
    image_bgr: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
    text_mask: np.ndarray | None = None,
    peak_two_thetas: Sequence[float] | None = None,
    prior_curve: CurveData | None = None,
    source: str = "dp_trace",
) -> tuple[CurveData, np.ndarray, np.ndarray]:
    """
    Full digitize: suppress axes, estimate baseline, DP-trace, convert.

    Returns ``(curve, curve_y, axis_mask)``.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    axis_mask = detect_axis_mask(
        gray,
        plot_left=calibration.plot_left,
        plot_right=calibration.plot_right,
        plot_top=calibration.plot_top,
        plot_bottom=calibration.plot_bottom,
    )
    baseline_y = estimate_baseline_y(
        gray,
        plot_left=calibration.plot_left,
        plot_right=calibration.plot_right,
        plot_top=calibration.plot_top,
        plot_bottom=calibration.plot_bottom,
        axis_mask=axis_mask,
        text_mask=text_mask,
    )
    prior_y = None
    if prior_curve is not None and prior_curve.two_theta:
        prior_y = _agent_prior_y_pixels(
            np.asarray(prior_curve.intensity, dtype=float),
            np.asarray(prior_curve.two_theta, dtype=float),
            calibration,
            gray.shape[1],
        )

    curve_y = trace_curve_dp(
        gray,
        plot_left=calibration.plot_left,
        plot_right=calibration.plot_right,
        plot_top=calibration.plot_top,
        plot_bottom=calibration.plot_bottom,
        text_mask=text_mask,
        axis_mask=axis_mask,
        prior_y=prior_y,
        baseline_y=baseline_y,
        peak_two_thetas=peak_two_thetas,
        calibration=calibration,
    )
    peak_cols = np.zeros(gray.shape[1], dtype=bool)
    peak_centers: list[int] = []
    if peak_two_thetas and calibration is not None:
        span = max(calibration.plot_right - calibration.plot_left, 1)
        deg_span = max(calibration.x_max - calibration.x_min, 1e-9)
        px_per_deg = span / deg_span
        half_px = max(4, int(round(PEAK_WINDOW_DEG * px_per_deg)))
        for tt in peak_two_thetas:
            frac = (float(tt) - calibration.x_min) / deg_span
            px = int(round(calibration.plot_left + frac * span))
            peak_centers.append(px)
            for dx in range(-half_px, half_px + 1):
                xx = px + dx
                if 0 <= xx < gray.shape[1]:
                    peak_cols[xx] = True
    curve_y = refine_curve_y_with_column_peaks(
        gray,
        curve_y,
        plot_left=calibration.plot_left,
        plot_right=calibration.plot_right,
        plot_top=calibration.plot_top,
        plot_bottom=calibration.plot_bottom,
        text_mask=text_mask,
        peak_columns=peak_cols,
        axis_mask=axis_mask,
    )
    # Second apex recovery after column refine (no smoothing / clipping).
    if peak_centers:
        curve_y = recover_peak_apexes(
            gray,
            curve_y,
            peak_centers,
            plot_left=calibration.plot_left,
            plot_right=calibration.plot_right,
            plot_top=calibration.plot_top,
            plot_bottom=calibration.plot_bottom,
            text_mask=text_mask,
            axis_mask=axis_mask,
            window_px=PEAK_APEX_WINDOW_PX,
        )
    # Light continuity repair: suppress one-column spikes that are not near
    # agent peak priors (keeps real XRD peaks, removes isolation artifacts).
    x0 = max(0, calibration.plot_left)
    x1 = min(gray.shape[1], calibration.plot_right)
    for x in range(x0 + 1, x1 - 1):
        if not (
            np.isfinite(curve_y[x - 1])
            and np.isfinite(curve_y[x])
            and np.isfinite(curve_y[x + 1])
        ):
            continue
        left, mid, right = float(curve_y[x - 1]), float(curve_y[x]), float(curve_y[x + 1])
        neigh = 0.5 * (left + right)
        if abs(mid - neigh) > 25 and not peak_cols[x]:
            # Isolated jump away from neighbors — pull back.
            if abs(left - right) < 20:
                curve_y[x] = neigh
    curve = curve_y_to_curve_data(
        curve_y, calibration, num_points=num_points, source=source
    )
    return curve, curve_y, axis_mask
