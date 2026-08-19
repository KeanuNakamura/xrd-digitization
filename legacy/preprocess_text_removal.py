"""
Apply removable in-plot text masks, then reconstruct the XRD curve.

Pipeline:
  1. Estimate a single-valued curve centerline ``curve_y[x]`` (DP trace).
  2. Build a complete glyph-removal mask (including antialiased edges).
  3. Replace glyph pixels with local background.
  4. Reconstruct the curve stroke through annotation regions from the
     precomputed centerline (plus surviving dark pixels / local width).

``plot_only.png`` remains an untouched crop. Cleaned images are
``*_preprocessed*.png``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

RemovalMethod = Literal["white", "local_background", "inpaint", "mask_only"]
DEFAULT_REMOVAL_METHOD: RemovalMethod = "local_background"

# Glyph removal (aggressive — reconstruction restores the curve).
BG_MARGIN = 12.0
GLYPH_DILATE_PX = 3
MEMBER_PAD_PX = 3
PALE_WIPE_THRESH = 253

# Curve tracing / reconstruction.
CURVE_INK_THRESH = 170
MAX_JUMP_PX = 48
JUMP_COST = 0.28
CURVATURE_COST = 0.08
TEXT_PENALTY = 12.0
AXIS_PENALTY = 40.0
DEFAULT_STROKE_RADIUS = 1
SURVIVING_SNAP_PX = 3
DP_HERMITE_MAX_DEV = 14.0

# Validation.
RESIDUAL_FRAC_245 = 0.05
RESIDUAL_FRAC_250 = 0.08
RESIDUAL_FRAC_253 = 0.12
CURVE_GAP_COLS = 2
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
REMOVABLE_PREFIX = "removable_"


@dataclass
class PreprocessTextRemovalResult:
    preprocessed_bgr: np.ndarray
    glyph_bgr: np.ndarray
    region_bgr: np.ndarray
    overlay_bgr: np.ndarray
    residual_debug_bgr: np.ndarray | None = None
    curve_damage_debug_bgr: np.ndarray | None = None
    removal_mask_used: np.ndarray | None = None
    protected_curve_mask: np.ndarray | None = None
    preserved_axis_mask: np.ndarray | None = None
    expanded_glyph_mask: np.ndarray | None = None
    curve_y: np.ndarray | None = None
    removal_method: str = DEFAULT_REMOVAL_METHOD
    status: str = STATUS_PARTIAL
    removed_pixel_count: int = 0
    protected_curve_pixel_count: int = 0
    residual_text_groups: int = 0
    curve_damage_groups: int = 0
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _as_u8_mask(mask: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    arr = mask
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape[:2] != (h, w):
        raise ValueError(f"Mask shape {arr.shape[:2]} does not match {(h, w)}")
    return (arr > 0).astype(np.uint8) * 255


def _clamp_box(box: Sequence[float], width: int, height: int, *, pad: int = 0) -> list[int]:
    x0 = max(0, int(box[0]) - pad)
    y0 = max(0, int(box[1]) - pad)
    x1 = min(width, int(box[2]) + pad)
    y1 = min(height, int(box[3]) + pad)
    return [x0, y0, x1, y1]


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1].copy()


def _local_background_value(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    ring: int = 6,
) -> float:
    h, w = gray.shape
    rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
    rx1, ry1 = min(w, x1 + ring), min(h, y1 + ring)
    ring_mask = np.zeros((h, w), dtype=bool)
    ring_mask[ry0:ry1, rx0:rx1] = True
    ring_mask[y0:y1, x0:x1] = False
    vals = gray[ring_mask]
    if vals.size == 0:
        vals = gray[ry0:ry1, rx0:rx1].ravel()
    if vals.size == 0:
        return 255.0
    return float(max(np.median(vals), 230.0))


# ---------------------------------------------------------------------------
# Global curve centerline estimation (DP)
# ---------------------------------------------------------------------------


def build_vertical_text_penalty(
    gray: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    min_run: int = 8,
) -> np.ndarray:
    """
    Penalty mask for DP tracing: tall vertical dark runs inside annotation
    boxes (glyph stems), not thin curve strokes.

    A column run of dark ink longer than ``min_run`` is treated as text,
    except pixels that are horizontally continuous with neighboring dark
    ink (likely the XRD stroke crossing behind the label).
    """
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
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

    for det in detections:
        if not str(det.get("role") or "").startswith(REMOVABLE_PREFIX):
            continue
        box = det.get("bbox")
        if not box:
            continue
        x0, y0, x1, y1 = _clamp_box(box, w, h, pad=2)
        for x in range(x0, x1):
            y = y0
            while y < y1:
                if not dark[y, x]:
                    y += 1
                    continue
                yb = y
                while yb < y1 and dark[yb, x]:
                    yb += 1
                run = yb - y
                if run >= min_run:
                    for yy in range(y, yb):
                        if not _horiz_curve_linked(x, yy):
                            out[yy, x] = 255
                y = yb
    return out


def estimate_curve_y_per_column(
    gray: np.ndarray,
    *,
    inner_plot: tuple[int, int, int, int] | None = None,
    text_penalty_mask: np.ndarray | None = None,
    axis_penalty_mask: np.ndarray | None = None,
    soft_box_penalty_mask: np.ndarray | None = None,
    max_jump: int = MAX_JUMP_PX,
) -> np.ndarray:
    """
    Estimate one y-position per x-column via dynamic programming.

    Cost encourages dark ink continuity and penalizes text / axis regions and
    large vertical jumps. Returns float array of length ``width`` (NaN outside
    the plot span).
    """
    h, w = gray.shape
    if inner_plot is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = inner_plot
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.full(w, np.nan, dtype=np.float32)

    # Unary cost: darker => lower. Normalize to [0, 1] then scale.
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    rh, rw = roi.shape
    # Soft darkness score (0 = white, 1 = black).
    darkness = np.clip((220.0 - roi) / 220.0, 0.0, 1.0)
    # Amplify unary so following dark ink outweighs small vertical jumps.
    unary = 6.0 * (1.0 - darkness)

    if text_penalty_mask is not None:
        tp = text_penalty_mask[y0:y1, x0:x1] > 0
        unary = unary + TEXT_PENALTY * tp.astype(np.float32)
    if soft_box_penalty_mask is not None:
        sb = soft_box_penalty_mask[y0:y1, x0:x1] > 0
        unary = unary + 1.5 * sb.astype(np.float32)
    if axis_penalty_mask is not None:
        ap = axis_penalty_mask[y0:y1, x0:x1] > 0
        unary = unary + AXIS_PENALTY * ap.astype(np.float32)

    # Strongly avoid plot-frame borders (long continuous axis ink).
    inset = max(3, min(10, rh // 30, rw // 40))
    unary[:inset, :] += AXIS_PENALTY
    unary[-inset:, :] += AXIS_PENALTY
    unary[:, :inset] += AXIS_PENALTY
    unary[:, -inset:] += AXIS_PENALTY

    # Prefer sparse dark columns (curve-like) over dense vertical text.
    text_roi = None
    if text_penalty_mask is not None:
        text_roi = text_penalty_mask[y0:y1, x0:x1] > 0
    for col in range(rw):
        col_d = darkness[:, col]
        dense = float((col_d > 0.25).mean())
        if dense > 0.25:
            unary[:, col] = unary[:, col] + 3.0 * dense
        if col_d.max() > 0.2:
            search = col_d.copy()
            search[:inset] = 0
            search[-inset:] = 0
            # Do not let vertical glyph ink define the column peak.
            if text_roi is not None:
                search = search * (1.0 - 0.9 * text_roi[:, col].astype(np.float32))
            if float(search.max()) <= 0.05:
                continue
            peak_y = int(np.argmax(search))
            for yy in range(rh):
                dist = abs(yy - peak_y)
                if dist <= 6:
                    unary[yy, col] -= 1.5 * max(0.0, 1.0 - dist / 6.0) * search[peak_y]
    jump = max(1, int(max_jump))
    dp = np.full((rh, rw), np.inf, dtype=np.float32)
    back = np.full((rh, rw), -1, dtype=np.int32)
    dp[:, 0] = unary[:, 0]

    # Forward DP with limited jump transitions.
    for x in range(1, rw):
        prev = dp[:, x - 1]
        # Rolling minimum trick for |y-y'| cost within jump window is heavier;
        # use a compact loop — plot widths are O(1e3).
        for y in range(rh):
            y_lo = max(0, y - jump)
            y_hi = min(rh, y + jump + 1)
            best = np.inf
            best_y = y
            for py in range(y_lo, y_hi):
                dy = abs(py - y)
                cost = prev[py] + JUMP_COST * dy
                if dy >= 2:
                    cost += CURVATURE_COST * (dy - 1)
                if cost < best:
                    best = cost
                    best_y = py
            dp[y, x] = best + unary[y, x]
            back[y, x] = best_y

    # Backtrack.
    curve = np.full(w, np.nan, dtype=np.float32)
    y = int(np.argmin(dp[:, -1]))
    for x in range(rw - 1, -1, -1):
        curve[x0 + x] = float(y0 + y)
        if x == 0:
            break
        y = int(back[y, x])
        if y < 0:
            break

    # Light smoothing (does not flatten sharp peaks much).
    valid = np.isfinite(curve)
    if valid.sum() >= 8:
        ys = curve.copy()
        idx = np.where(valid)[0]
        for _ in range(2):
            for i in range(1, len(idx) - 1):
                a, b, c = idx[i - 1], idx[i], idx[i + 1]
                # Only smooth mild wiggles; keep large peak excursions.
                mid = 0.25 * ys[a] + 0.5 * ys[b] + 0.25 * ys[c]
                if abs(ys[b] - mid) <= 4.0:
                    ys[b] = mid
        curve = ys
    return curve


def refine_curve_y_with_column_peaks(
    gray: np.ndarray,
    curve_y: np.ndarray,
    *,
    text_mask: np.ndarray | None = None,
    inner_plot: tuple[int, int, int, int] | None = None,
    search_above: int = 0,
) -> np.ndarray:
    """
    Snap the DP path onto strong non-text dark peaks in each column.

    When a vertical label box covers a peak tip, DP may hug the baseline.
    If a clear dark ridge exists above the path (and is continuous with
    neighbors), pull ``curve_y`` up to that ridge.

    ``search_above`` of 0 means search from the inner-plot top down to just
    below the current path (needed for tall XRD peaks).
    """
    h, w = gray.shape
    if inner_plot is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = inner_plot
    out = curve_y.astype(np.float32).copy()
    cand = np.full(w, np.nan, dtype=np.float32)

    for x in range(max(0, x0), min(w, x1)):
        y = out[x]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        if search_above > 0:
            top = max(y0 + 2, yi - search_above)
        else:
            top = y0 + 2
        bot = min(y1 - 2, yi + 8)
        if bot - top < 8:
            continue
        col = gray[top:bot, x].astype(np.float32)
        score = np.clip((200.0 - col) / 200.0, 0.0, 1.0)
        if text_mask is not None:
            tm = text_mask[top:bot, x] > 0
            if tm.any():
                soft = tm.astype(np.float32)
                score = score * (1.0 - 0.75 * soft)
        if float(score.max()) < 0.35:
            continue
        # Prefer uppermost strong local maximum (peak tip).
        peak = int(np.argmax(score))
        for _ in range(5):
            above = score[: max(1, peak - 3)]
            if above.size and float(above.max()) >= max(0.45, float(score[peak]) * 0.8):
                peak = int(np.argmax(above))
            else:
                break
        y_peak = float(top + peak)
        run_lo, run_hi = peak, peak
        while run_lo > 0 and score[run_lo - 1] > 0.3:
            run_lo -= 1
        while run_hi + 1 < score.size and score[run_hi + 1] > 0.3:
            run_hi += 1
        # Reject tall vertical text stems.
        if (run_hi - run_lo + 1) > 10:
            continue
        if y_peak < y - 6 and score[peak] >= 0.45:
            cand[x] = y_peak
        elif abs(y_peak - y) <= 5 and score[peak] >= 0.35:
            cand[x] = y_peak

    # Accept candidate only when neighboring columns agree (continuity).
    for x in range(max(0, x0), min(w, x1)):
        if not np.isfinite(cand[x]):
            # Bridge from neighbors even when local score failed (peak tip in glyph).
            ys_nb = []
            for dx in (-2, -1, 1, 2):
                xx = x + dx
                if 0 <= xx < w and np.isfinite(cand[xx]):
                    ys_nb.append(cand[xx])
            if len(ys_nb) >= 2 and float(np.std(ys_nb)) <= 12:
                y_guess = float(np.median(ys_nb))
                yi = int(round(y_guess))
                y_lo = max(y0, yi - 4)
                y_hi = min(y1, yi + 5)
                seg = gray[y_lo:y_hi, x]
                if seg.size and int(seg.min()) < CURVE_INK_THRESH:
                    cand[x] = float(y_lo + int(np.argmin(seg)))
                else:
                    cand[x] = y_guess
            else:
                continue
        ys = [cand[x]]
        for dx in (-2, -1, 1, 2):
            xx = x + dx
            if 0 <= xx < w and np.isfinite(cand[xx]):
                ys.append(cand[xx])
        if len(ys) < 2 and not (
            np.isfinite(cand[x]) and abs(cand[x] - out[x]) <= 4
        ):
            continue
        if len(ys) >= 2 and float(np.std(ys)) > 18:
            continue
        out[x] = float(np.median(ys))

    # Fill short holes where snap failed inside a peak run.
    x = max(0, x0)
    while x < min(w, x1):
        if np.isfinite(cand[x]):
            x += 1
            continue
        left = x - 1
        while left >= x0 and not np.isfinite(cand[left]):
            left -= 1
        right = x
        while right < x1 and not np.isfinite(cand[right]):
            right += 1
        if left >= x0 and right < x1 and 0 < (right - left) <= 12:
            for xx in range(left + 1, right):
                t = (xx - left) / max(1, right - left)
                y_lin = (1 - t) * cand[left] + t * cand[right]
                y_dp = out[xx] if np.isfinite(out[xx]) else y_lin
                out[xx] = float(min(y_lin, y_dp))
        x = right if right > x else x + 1

    # Suppress sudden peak→baseline drops: continue a peak plateau while the
    # original column still has dark ink near the previous peak y.
    for x in range(max(0, x0) + 1, min(w, x1)):
        if not (np.isfinite(out[x]) and np.isfinite(out[x - 1])):
            continue
        if out[x] <= out[x - 1] + 20:
            continue
        y_prev = float(out[x - 1])
        yi = int(round(y_prev))
        y_lo = max(y0, yi - 5)
        y_hi = min(y1, yi + 6)
        seg = gray[y_lo:y_hi, x]
        if seg.size and int(seg.min()) < CURVE_INK_THRESH:
            out[x] = float(y_lo + int(np.argmin(seg)))
        elif x + 1 < x1 and np.isfinite(out[x + 1]) and out[x + 1] <= y_prev + 15:
            out[x] = 0.5 * y_prev + 0.5 * float(out[x + 1])

    return out


def estimate_local_stroke_radius(
    gray: np.ndarray,
    curve_y: np.ndarray,
    *,
    sample_stride: int = 8,
) -> int:
    """Estimate curve stroke half-width from original dark ink around centerline."""
    h, w = gray.shape
    widths: list[int] = []
    for x in range(0, w, sample_stride):
        y = curve_y[x]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        if not (0 <= yi < h):
            continue
        # Expand while dark.
        lo = yi
        while lo > 0 and gray[lo - 1, x] < CURVE_INK_THRESH:
            lo -= 1
        hi = yi
        while hi + 1 < h and gray[hi + 1, x] < CURVE_INK_THRESH:
            hi += 1
        widths.append(max(1, (hi - lo + 1) // 2))
    if not widths:
        return DEFAULT_STROKE_RADIUS
    return int(np.clip(np.median(widths), 1, 4))


def build_centerline_mask(
    curve_y: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    radius: int = DEFAULT_STROKE_RADIUS,
) -> np.ndarray:
    """Thin protected band = dilate(centerline, radius)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        y = curve_y[x]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        y0 = max(0, yi - radius)
        y1 = min(h, yi + radius + 1)
        mask[y0:y1, x] = 255
    return mask


# ---------------------------------------------------------------------------
# Glyph removal masks (complete glyphs, including AA)
# ---------------------------------------------------------------------------


def _threshold_text_in_box(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return mask
    roi = gray[y0:y1, x0:x1]
    bg = _local_background_value(gray, x0, y0, x1, y1)
    local = (roi < (bg - BG_MARGIN)).astype(np.uint8) * 255
    if float(roi.std()) > 6.0 and roi.size >= 16:
        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        _t, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if 0.01 < float((otsu > 0).mean()) < 0.9:
            local = cv2.bitwise_or(local, otsu)
    # Catch pale AA / ghost outlines near stronger ink.
    pale = (roi < min(253.0, bg - 2.0)).astype(np.uint8) * 255
    near = cv2.dilate(local, np.ones((5, 5), np.uint8), 1)
    local = cv2.bitwise_or(local, cv2.bitwise_and(pale, near))
    # Extremely pale but structured: dilate once more for AA fringe.
    if np.any(local):
        fringe = cv2.dilate(local, np.ones((3, 3), np.uint8), 1)
        very_pale = (roi < 254).astype(np.uint8) * 255
        local = cv2.bitwise_or(local, cv2.bitwise_and(very_pale, fringe))
    mask[y0:y1, x0:x1] = local
    return mask


def build_complete_glyph_removal_mask(
    image_bgr: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    preserved_axis_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Aggressive glyph mask covering full characters + antialiased edges.

    Does not attempt to spare curve pixels inside the box — reconstruction
    restores the trace afterward. For in-plot annotations, every non-near-white
    pixel inside the padded detection box is included so overlapping glyph
    strokes cannot survive as dark fragments on the curve.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    preserved = _as_u8_mask(preserved_axis_mask, (h, w))
    out = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        if not str(det.get("role") or "").startswith(REMOVABLE_PREFIX):
            continue
        box = det.get("bbox")
        if not box:
            continue
        members = det.get("member_boxes") or [box]
        orientation = str(det.get("orientation") or det.get("alignment") or "")
        group = np.zeros((h, w), dtype=np.uint8)
        for mb in members:
            mx0, my0, mx1, my1 = _clamp_box(mb, w, h, pad=MEMBER_PAD_PX)
            group = cv2.bitwise_or(group, _threshold_text_in_box(gray, mx0, my0, mx1, my1))

        gx0, gy0, gx1, gy1 = _clamp_box(box, w, h, pad=MEMBER_PAD_PX)
        if orientation == "vertical" or (gy1 - gy0) >= 1.4 * max(1, gx1 - gx0):
            col = np.zeros((h, w), dtype=np.uint8)
            for mb in members:
                mx0, my0, mx1, my1 = _clamp_box(mb, w, h, pad=3)
                col[gy0:gy1, mx0:mx1] = 255
            full = _threshold_text_in_box(gray, gx0, gy0, gx1, gy1)
            group = cv2.bitwise_or(group, cv2.bitwise_and(full, col))
        else:
            group = cv2.bitwise_or(group, _threshold_text_in_box(gray, gx0, gy0, gx1, gy1))

        # Full-box pale wipe: any non-near-white pixel in the annotation box.
        # This catches glyph stems that touch the curve and would otherwise be
        # mistaken for curve ink by connected-component logic.
        bg = _local_background_value(gray, gx0, gy0, gx1, gy1)
        box_pale = (gray[gy0:gy1, gx0:gx1] < min(253.0, bg - 1.0)).astype(np.uint8) * 255
        group[gy0:gy1, gx0:gx1] = cv2.bitwise_or(group[gy0:gy1, gx0:gx1], box_pale)

        if np.any(group):
            group = cv2.dilate(
                group,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=GLYPH_DILATE_PX,
            )
        if np.any(preserved):
            group = cv2.bitwise_and(group, cv2.bitwise_not(preserved))
        out = cv2.bitwise_or(out, group)
    return out


# ---------------------------------------------------------------------------
# Removal + reconstruction
# ---------------------------------------------------------------------------


def _ring_background_bgr(image_bgr: np.ndarray, removal_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    unmasked = removal_mask == 0
    global_bg = float(np.median(gray[unmasked])) if np.any(unmasked) else 255.0
    global_bg = max(global_bg, 245.0)
    if global_bg >= 245:
        out = np.empty_like(image_bgr)
        out[:] = (255, 255, 255)
        return out
    filled = image_bgr.copy()
    filled[removal_mask > 0] = (int(global_bg), int(global_bg), int(global_bg))
    return cv2.medianBlur(filled, 17)


def apply_removal_method(
    image_bgr: np.ndarray,
    removal_mask: np.ndarray,
    *,
    method: RemovalMethod = DEFAULT_REMOVAL_METHOD,
    protected_curve_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Replace removal pixels. When reconstructing afterward, pass
    ``protected_curve_mask=None`` so glyphs are fully cleared.
    """
    out = image_bgr.copy()
    rem = removal_mask > 0
    if protected_curve_mask is not None:
        rem = rem & (protected_curve_mask == 0)
    applied = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    applied[rem] = 255
    removed = int(np.count_nonzero(rem))
    if method == "mask_only" or removed == 0:
        return out, applied, 0 if method == "mask_only" else removed
    if method == "white":
        out[rem] = (255, 255, 255)
        return out, applied, removed
    if method == "inpaint":
        return (
            cv2.inpaint(image_bgr, applied, inpaintRadius=2, flags=cv2.INPAINT_TELEA),
            applied,
            removed,
        )
    if method == "local_background":
        bg = _ring_background_bgr(image_bgr, applied)
        out[rem] = bg[rem]
        return out, applied, removed
    raise ValueError(f"Unsupported removal_method: {method!r}")


def _local_ink_intensity(gray: np.ndarray, curve_y: np.ndarray, x: int, radius: int = 12) -> int:
    h, w = gray.shape
    vals = []
    for xx in range(max(0, x - radius), min(w, x + radius + 1)):
        y = curve_y[xx]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        if 0 <= yi < h and gray[yi, xx] < CURVE_INK_THRESH:
            vals.append(int(gray[yi, xx]))
    if not vals:
        return 40
    return int(np.median(vals))


def wipe_non_curve_ink_in_annotations(
    cleaned_bgr: np.ndarray,
    curve_y: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    stroke_radius: int = DEFAULT_STROKE_RADIUS,
    preserved_axis_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    After reconstruction, delete leftover glyph ink inside annotation boxes
    that is not on the thin centerline band.

    This is the operational form of centerline-only protection: a CC that
    touches the curve is not preserved as a whole.
    """
    out = cleaned_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    band = build_centerline_mask(
        curve_y, (h, w), radius=max(1, stroke_radius) + 1
    )
    wipe = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        if not str(det.get("role") or "").startswith(REMOVABLE_PREFIX):
            continue
        box = det.get("bbox")
        if not box:
            continue
        x0, y0, x1, y1 = _clamp_box(box, w, h, pad=2)
        roi_dark = gray[y0:y1, x0:x1] < PALE_WIPE_THRESH
        roi_band = band[y0:y1, x0:x1] > 0
        wipe[y0:y1, x0:x1][roi_dark & ~roi_band] = 255
    if preserved_axis_mask is not None:
        wipe = cv2.bitwise_and(wipe, cv2.bitwise_not(preserved_axis_mask))
    if np.any(wipe):
        out[wipe > 0] = (255, 255, 255)
        # Re-paint the centerline so wipe cannot nibble reconstructed stroke AA.
        for x in range(w):
            y = curve_y[x]
            if not np.isfinite(y):
                continue
            yi = int(round(y))
            if not (0 <= yi < h):
                continue
            if wipe[
                max(0, yi - stroke_radius - 2) : min(h, yi + stroke_radius + 3), x
            ].any():
                rad = max(1, stroke_radius)
                intensity = 30
                out[max(0, yi - rad) : min(h, yi + rad + 1), x] = (
                    intensity,
                    intensity,
                    intensity,
                )
    return out


def wipe_pale_residuals(
    cleaned_bgr: np.ndarray,
    glyph_mask: np.ndarray,
    *,
    preserved_axis_mask: np.ndarray | None = None,
    thresh: int = PALE_WIPE_THRESH,
) -> np.ndarray:
    """
    Second-pass wipe: any non-white pixel inside the glyph mask becomes
    background. Curve ink inside the mask is intentionally cleared; the
    stroke is redrawn afterward.
    """
    out = cleaned_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    rem = glyph_mask > 0
    if preserved_axis_mask is not None:
        rem = rem & (preserved_axis_mask == 0)
    pale = rem & (gray < thresh)
    if not np.any(pale):
        return out
    # Local background from cleaned pixels just outside the pale set.
    dil = cv2.dilate(pale.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), 1) > 0
    ring = dil & ~pale
    if np.any(ring):
        bg = int(np.median(gray[ring]))
    else:
        bg = 255
    bg = max(bg, 250)
    out[pale] = (bg, bg, bg)
    return out


def _hermite_y(t: float, yL: float, yR: float, mL: float, mR: float, span: float) -> float:
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return float(h00 * yL + h10 * mL * span + h01 * yR + h11 * mR * span)


def _anchor_y(curve_y: np.ndarray, x: int, x0: int, x1: int, side: str) -> tuple[int, float] | None:
    """Find nearest finite curve sample just outside [x0, x1)."""
    if side == "left":
        xx = x0 - 1
        while xx >= 0:
            if np.isfinite(curve_y[xx]):
                return xx, float(curve_y[xx])
            xx -= 1
    else:
        xx = x1
        while xx < len(curve_y):
            if np.isfinite(curve_y[xx]):
                return xx, float(curve_y[xx])
            xx += 1
    return None


def reconstruct_curve_stroke(
    cleaned_bgr: np.ndarray,
    original_bgr: np.ndarray,
    curve_y: np.ndarray,
    *,
    removal_mask: np.ndarray,
    detections: Sequence[dict[str, Any]],
    stroke_radius: int = DEFAULT_STROKE_RADIUS,
    inner_plot: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw the reconstructed XRD stroke through annotation regions.

    Primary path is the precomputed ``curve_y`` (peak-preserving). Hermite
    interpolation bridges spans where the DP path is missing or clearly
    contaminated by vertical text. Surviving original dark pixels may snap
    the path only when within ``SURVIVING_SNAP_PX`` of that estimate.

    Returns ``(reconstructed_bgr, final_curve_y)``.
    """
    out = cleaned_bgr.copy()
    gray_o = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray_o.shape
    if inner_plot is None:
        px0, py0, px1, py1 = 0, 0, w, h
    else:
        px0, py0, px1, py1 = inner_plot

    final_y = curve_y.astype(np.float32).copy()
    rem = removal_mask > 0

    # Merge removable x-spans (pad so AA wipe columns are included).
    spans: list[list[int]] = []
    for det in detections:
        if not str(det.get("role") or "").startswith(REMOVABLE_PREFIX):
            continue
        box = det.get("bbox")
        if not box:
            continue
        x0, _y0, x1, _y1 = _clamp_box(box, w, h, pad=2)
        if x1 - x0 < 1:
            continue
        spans.append([x0, x1])
    spans.sort()
    merged: list[list[int]] = []
    for s in spans:
        if not merged or s[0] > merged[-1][1] + 1:
            merged.append(s)
        else:
            merged[-1][1] = max(merged[-1][1], s[1])

    for x0, x1 in merged:
        left = _anchor_y(final_y, x0, x0, x1, "left")
        right = _anchor_y(final_y, x1, x0, x1, "right")
        if left is None or right is None:
            # Fall back to any finite samples inside / near the span.
            xs = [x for x in range(max(px0, x0 - 5), min(px1, x1 + 5)) if np.isfinite(final_y[x])]
            if len(xs) < 2:
                continue
            left = (xs[0], float(final_y[xs[0]]))
            right = (xs[-1], float(final_y[xs[-1]]))
        xL, yL = left
        xR, yR = right
        span = max(1, xR - xL)
        # Endpoint slopes from outside samples.
        yLm = float(final_y[max(px0, xL - 4)]) if np.isfinite(final_y[max(px0, xL - 4)]) else yL
        yRp = float(final_y[min(px1 - 1, xR + 4)]) if np.isfinite(final_y[min(px1 - 1, xR + 4)]) else yR
        mL = (yL - yLm) / max(1.0, xL - max(px0, xL - 4))
        mR = (yRp - yR) / max(1.0, min(px1 - 1, xR + 4) - xR)

        for x in range(x0, x1):
            t = (x - xL) / span
            t = float(np.clip(t, 0.0, 1.0))
            y_herm = _hermite_y(t, yL, yR, mL, mR, float(span))
            y_dp = float(final_y[x]) if np.isfinite(final_y[x]) else np.nan

            # Prefer DP when it stays near the bridge (keeps peaks). Reject
            # DP samples that jumped onto vertical glyph ink far from the bridge.
            if np.isfinite(y_dp) and abs(y_dp - y_herm) <= DP_HERMITE_MAX_DEV:
                y_use = y_dp
            elif np.isfinite(y_dp) and y_dp < min(yL, yR) - 2.0:
                # Peak tip above both anchors — keep DP (image y decreases upward).
                y_use = y_dp
            else:
                y_use = y_herm

            # Surviving original dark pixels near the estimate only.
            yi = int(round(y_use))
            y_lo = max(py0, yi - SURVIVING_SNAP_PX)
            y_hi = min(py1, yi + SURVIVING_SNAP_PX + 1)
            col = gray_o[y_lo:y_hi, x]
            dark = np.where(col < CURVE_INK_THRESH)[0]
            if dark.size:
                # Prefer the darkest candidate (true curve ink over pale AA).
                scores = col[dark].astype(np.float32)
                pick = int(dark[int(np.argmin(scores))])
                y_snap = float(y_lo + pick)
                if abs(y_snap - y_use) <= SURVIVING_SNAP_PX:
                    y_use = 0.65 * y_use + 0.35 * y_snap

            final_y[x] = float(y_use)

    # Paint columns: every annotation span + any removal intersecting centerline.
    paint_xs: set[int] = set()
    for x0, x1 in merged:
        paint_xs.update(range(x0, x1))
    for x in range(max(0, px0), min(w, px1)):
        y = final_y[x]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        band = rem[max(0, yi - stroke_radius - 2) : min(h, yi + stroke_radius + 3), x]
        if band.size and band.any():
            paint_xs.add(x)

    # Fill NaNs on paint columns from neighbors.
    for x in sorted(paint_xs):
        if np.isfinite(final_y[x]):
            continue
        left = x - 1
        right = x + 1
        while left >= 0 and not np.isfinite(final_y[left]):
            left -= 1
        while right < w and not np.isfinite(final_y[right]):
            right += 1
        if left >= 0 and right < w:
            t = (x - left) / max(1, right - left)
            final_y[x] = (1 - t) * final_y[left] + t * final_y[right]
        elif left >= 0:
            final_y[x] = final_y[left]
        elif right < w:
            final_y[x] = final_y[right]

    for x in sorted(paint_xs):
        y = final_y[x]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        if not (0 <= yi < h):
            continue
        if inner_plot is not None and not (py0 <= yi < py1):
            continue
        intensity = _local_ink_intensity(gray_o, final_y, x)
        # Avoid inheriting white from wiped columns: clamp to dark ink.
        intensity = int(np.clip(intensity, 0, 70))
        color = (intensity, intensity, intensity)
        rad = max(1, stroke_radius)
        y0b = max(0, yi - rad)
        y1b = min(h, yi + rad + 1)
        out[y0b:y1b, x] = color
        # Soft AA edge only on empty background.
        if y0b - 1 >= 0 and int(out[y0b - 1, x, 0]) > 245:
            edge = int(0.5 * 255 + 0.5 * intensity)
            out[y0b - 1, x] = (edge, edge, edge)
        if y1b < h and int(out[y1b, x, 0]) > 245:
            edge = int(0.5 * 255 + 0.5 * intensity)
            out[y1b, x] = (edge, edge, edge)

    return out, final_y


# ---------------------------------------------------------------------------
# Validation / overlays
# ---------------------------------------------------------------------------


def validate_residual_text(
    original_bgr: np.ndarray,
    cleaned_bgr: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    glyph_mask: np.ndarray,
    curve_y: np.ndarray | None = None,
    stroke_radius: int = DEFAULT_STROKE_RADIUS,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Detect pale ghost outlines using multi-threshold residual metrics."""
    h, w = original_bgr.shape[:2]
    gray_c = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    gray_o = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    debug = cleaned_bgr.copy()
    residuals: list[dict[str, Any]] = []

    # Dilate glyph template slightly so AA fringe is included in support.
    support_mask = cv2.dilate(
        (glyph_mask > 0).astype(np.uint8) * 255,
        np.ones((3, 3), np.uint8),
        1,
    )
    # Reconstructed stroke must not count as residual text.
    curve_band = np.zeros((h, w), dtype=bool)
    if curve_y is not None:
        rad = max(1, stroke_radius) + 1
        for x in range(w):
            y = curve_y[x]
            if not np.isfinite(y):
                continue
            yi = int(round(y))
            y0b = max(0, yi - rad)
            y1b = min(h, yi + rad + 1)
            curve_band[y0b:y1b, x] = True

    for det in detections:
        if not str(det.get("role") or "").startswith(REMOVABLE_PREFIX):
            continue
        box = det.get("bbox")
        if not box:
            continue
        x0, y0, x1, y1 = _clamp_box(box, w, h, pad=1)
        if x1 - x0 < 4 or y1 - y0 < 6:
            continue
        support = support_mask[y0:y1, x0:x1] > 0
        if curve_band[y0:y1, x0:x1].any():
            support = support & ~curve_band[y0:y1, x0:x1]
        if not np.any(support):
            bg = _local_background_value(gray_o, x0, y0, x1, y1)
            support = gray_o[y0:y1, x0:x1] < bg - 2.0
            if curve_band[y0:y1, x0:x1].any():
                support = support & ~curve_band[y0:y1, x0:x1]
        # Also count pale leftovers anywhere in the box (minus curve band),
        # so AA ghosts outside the thresholded glyph support are still flagged.
        box_support = np.ones((y1 - y0, x1 - x0), dtype=bool)
        if curve_band[y0:y1, x0:x1].any():
            box_support = box_support & ~curve_band[y0:y1, x0:x1]
        support_n = int(support.sum())
        box_n = int(box_support.sum())
        if support_n < 8 and box_n < 20:
            continue
        region = gray_c[y0:y1, x0:x1]
        denom = max(support_n, 1)
        f245 = float(((region < 245) & support).sum()) / denom
        f250 = float(((region < 250) & support).sum()) / denom
        f253 = float(((region < 253) & support).sum()) / denom
        if box_n >= 20:
            f245 = max(f245, float(((region < 245) & box_support).sum()) / box_n)
            f250 = max(f250, float(((region < 250) & box_support).sum()) / box_n)
            f253 = max(f253, float(((region < 253) & box_support).sum()) / box_n)
        orig = gray_o[y0:y1, x0:x1].astype(np.float32)
        clean = region.astype(np.float32)
        residual_map = np.clip(clean - 255.0, -255, 0)
        template = np.clip(255.0 - orig, 0, 255)
        if support.any():
            a = residual_map[support].ravel()
            b = template[support].ravel()
            if a.size >= 8 and float(np.std(a)) > 1e-6 and float(np.std(b)) > 1e-6:
                corr = float(np.corrcoef(a, b)[0, 1])
            else:
                corr = 0.0
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        incomplete = (
            f245 >= RESIDUAL_FRAC_245
            or f250 >= RESIDUAL_FRAC_250
            or f253 >= RESIDUAL_FRAC_253
            or (corr >= 0.25 and f253 >= 0.05)
            or (corr >= 0.4 and f250 >= 0.03)
        )
        if incomplete:
            residuals.append(
                {
                    "bbox": [float(v) for v in box],
                    "residual_fraction_lt_245": f245,
                    "residual_fraction_lt_250": f250,
                    "residual_fraction_lt_253": f253,
                    "residual_template_corr": corr,
                }
            )
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 0, 255), 2)
            ghost = (region < 253) & support
            ys, xs = np.where(ghost)
            debug[y0 + ys, x0 + xs] = (0, 128, 255)

    cv2.putText(
        debug,
        f"residual groups={len(residuals)} (pale ghosts included)",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return residuals, debug


def validate_curve_damage(
    cleaned_bgr: np.ndarray,
    curve_y: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    stroke_radius: int,
    inner_plot: tuple[int, int, int, int] | None = None,
    original_bgr: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Flag annotation regions where the cleaned trace is discontinuous vs curve_y.

    ``curve_y`` should be the reconstructed centerline. Every accepted removable
    annotation is checked; damaged columns are marked red.
    """
    del inner_plot  # reserved for future ROI gating
    h, w = cleaned_bgr.shape[:2]
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    gray_o = (
        cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        if original_bgr is not None
        else None
    )
    debug = cleaned_bgr.copy()
    damaged: list[dict[str, Any]] = []

    # Expected centerline in cyan; reconstructed presence checked below.
    for x in range(w):
        y = curve_y[x]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        if 0 <= yi < h:
            debug[yi, x] = (255, 255, 0)

    for det in detections:
        if not str(det.get("role") or "").startswith(REMOVABLE_PREFIX):
            continue
        box = det.get("bbox")
        if not box:
            continue
        x0, y0, x1, y1 = _clamp_box(box, w, h, pad=0)
        if x1 - x0 < 3:
            continue
        missing_cols = 0
        max_run = 0
        run = 0
        jump = 0.0
        prev_y = None
        thin_cols = 0
        truncated_peak = 0
        for x in range(x0, x1):
            y = curve_y[x]
            if not np.isfinite(y):
                missing_cols += 1
                run += 1
                max_run = max(max_run, run)
                continue
            yi = int(round(y))
            y_lo = max(0, yi - stroke_radius - 1)
            y_hi = min(h, yi + stroke_radius + 2)
            band = gray[y_lo:y_hi, x]
            present = bool(np.any(band < CURVE_INK_THRESH))
            if present:
                width = int(np.count_nonzero(band < CURVE_INK_THRESH))
                if width < max(1, stroke_radius):
                    thin_cols += 1
            if not present:
                missing_cols += 1
                run += 1
                max_run = max(max_run, run)
                debug[y_lo:y_hi, x] = (0, 0, 255)
            else:
                run = 0
                debug[yi, x] = (0, 255, 0)
            if prev_y is not None and np.isfinite(y):
                jump = max(jump, abs(float(y) - float(prev_y)))
            prev_y = y

            # Peak truncation vs original: a short dark ridge well above the
            # reconstructed y (curve-like, not a tall glyph stem).
            if gray_o is not None:
                # Search from the annotation top (or above reconstructed y).
                top = max(0, min(y0, yi - 120) if yi < h else y0)
                bot = max(top + 1, min(h, yi - 12) if np.isfinite(y) else y1)
                if bot > top + 5:
                    seg = gray_o[top:bot, x]
                    if seg.size and int(seg.min()) < 40:
                        peak_y = top + int(np.argmin(seg))
                        lo, hi = peak_y, peak_y
                        while lo > top and gray_o[lo - 1, x] < CURVE_INK_THRESH:
                            lo -= 1
                        while hi + 1 < bot and gray_o[hi + 1, x] < CURVE_INK_THRESH:
                            hi += 1
                        if (hi - lo + 1) <= 6 and (yi - peak_y) >= 30:
                            truncated_peak += 1
                            debug[peak_y, x] = (0, 0, 255)

        span = max(1, x1 - x0)
        frac_missing = missing_cols / span
        broken = (
            max_run > CURVE_GAP_COLS
            or frac_missing >= 0.2
            or (jump >= 40 and frac_missing >= 0.1)
            or thin_cols >= max(3, span // 4)
            or truncated_peak >= max(3, span // 5)
        )
        if broken:
            damaged.append(
                {
                    "bbox": [float(v) for v in box],
                    "missing_columns": missing_cols,
                    "max_gap_run": max_run,
                    "missing_fraction": float(frac_missing),
                    "max_y_jump": float(jump),
                    "thin_columns": thin_cols,
                    "truncated_peak_columns": truncated_peak,
                }
            )
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 0, 255), 2)

    cv2.putText(
        debug,
        f"curve damage={len(damaged)}  cyan=expected  green=ok  red=missing",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return damaged, debug


def build_preprocessed_overlay(
    image_bgr: np.ndarray,
    *,
    removed_mask: np.ndarray,
    preserved_axis_mask: np.ndarray | None = None,
    protected_curve_mask: np.ndarray | None = None,
) -> np.ndarray:
    overlay = image_bgr.copy()
    if preserved_axis_mask is not None:
        sel = preserved_axis_mask > 0
        overlay[sel] = (0.55 * overlay[sel] + 0.45 * np.array([0, 140, 255])).astype(
            np.uint8
        )
    if protected_curve_mask is not None:
        sel = protected_curve_mask > 0
        overlay[sel] = (0.45 * overlay[sel] + 0.55 * np.array([255, 220, 0])).astype(
            np.uint8
        )
    if removed_mask is not None:
        sel = removed_mask > 0
        overlay[sel] = (0.35 * overlay[sel] + 0.65 * np.array([0, 0, 255])).astype(
            np.uint8
        )
    cv2.putText(
        overlay,
        "red=removed  orange=preserved_axis  cyan=centerline_band",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return overlay


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def preprocess_removable_text(
    image_bgr: np.ndarray,
    *,
    glyph_mask: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
    preserved_axis_mask: np.ndarray | None = None,
    detections: Sequence[dict[str, Any]] | None = None,
    plot_bbox: tuple[int, int, int, int] | None = None,
    inner_plot_bbox: tuple[int, int, int, int] | None = None,
    removal_method: RemovalMethod = DEFAULT_REMOVAL_METHOD,
) -> PreprocessTextRemovalResult:
    """
    Remove in-plot annotations and reconstruct the XRD curve.

    Status is ``success`` only when residual ghosting and curve gaps are both
    below threshold; otherwise ``partial`` / ``failed``.
    """
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    preserved = _as_u8_mask(preserved_axis_mask, (height, width))
    region = _as_u8_mask(region_mask, (height, width))
    region = cv2.bitwise_and(region, cv2.bitwise_not(preserved))
    dets = list(detections or [])
    removable = [d for d in dets if str(d.get("role") or "").startswith(REMOVABLE_PREFIX)]
    inner = inner_plot_bbox or plot_bbox

    # 1) Complete glyph mask for erasure (aggressive; may include curve pixels).
    if removable:
        expanded = build_complete_glyph_removal_mask(
            image_bgr, removable, preserved_axis_mask=preserved
        )
    else:
        expanded = _as_u8_mask(glyph_mask, (height, width))
        expanded = cv2.bitwise_and(expanded, cv2.bitwise_not(preserved))
        if np.any(expanded):
            expanded = cv2.dilate(
                expanded,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=GLYPH_DILATE_PX,
            )

    # Tracing penalty: tall vertical glyph runs only (not thin curve strokes).
    text_penalty = build_vertical_text_penalty(gray, removable)
    soft_box = np.zeros((height, width), dtype=np.uint8)
    for det in removable:
        box = det.get("bbox")
        if not box:
            continue
        x0, y0, x1, y1 = _clamp_box(box, width, height, pad=1)
        soft_box[y0:y1, x0:x1] = 255
    soft_box = cv2.bitwise_and(soft_box, cv2.bitwise_not(text_penalty))

    # 2) Global curve centerline BEFORE deleting text.
    curve_y = estimate_curve_y_per_column(
        gray,
        inner_plot=inner,
        text_penalty_mask=text_penalty,
        axis_penalty_mask=preserved,
        soft_box_penalty_mask=soft_box,
    )
    curve_y = refine_curve_y_with_column_peaks(
        gray,
        curve_y,
        text_mask=text_penalty,
        inner_plot=inner,
    )
    stroke_radius = estimate_local_stroke_radius(gray, curve_y)

    # 3) Remove glyphs (no curve protection during erase), then wipe pale ghosts.
    erased, applied, removed = apply_removal_method(
        image_bgr,
        expanded,
        method=removal_method,
        protected_curve_mask=None,
    )
    if removal_method != "mask_only":
        erased = wipe_pale_residuals(
            erased, expanded, preserved_axis_mask=preserved, thresh=PALE_WIPE_THRESH
        )

    # 4) Reconstruct curve stroke through annotation regions.
    if removal_method != "mask_only":
        glyph_clean, final_y = reconstruct_curve_stroke(
            erased,
            image_bgr,
            curve_y,
            removal_mask=applied,
            detections=removable,
            stroke_radius=stroke_radius,
            inner_plot=inner,
        )
        # Drop leftover glyph stems that are not on the thin centerline band.
        glyph_clean = wipe_non_curve_ink_in_annotations(
            glyph_clean,
            final_y,
            removable,
            stroke_radius=stroke_radius,
            preserved_axis_mask=preserved,
        )
    else:
        glyph_clean = erased
        final_y = curve_y

    # Thin protected band from the reconstructed centerline (not full CCs).
    protected = build_centerline_mask(
        final_y, (height, width), radius=max(1, stroke_radius)
    )

    # Region diagnostic path remains destructive (no reconstruction claim).
    region_method: RemovalMethod = (
        "white" if removal_method == "local_background" else removal_method
    )
    if removal_method == "mask_only":
        region_method = "mask_only"
    region_clean, _ra, region_removed = apply_removal_method(
        image_bgr,
        region,
        method=region_method,
        protected_curve_mask=None,
    )

    residuals, residual_dbg = validate_residual_text(
        image_bgr,
        glyph_clean,
        removable,
        glyph_mask=expanded,
        curve_y=final_y,
        stroke_radius=stroke_radius,
    )
    damages, damage_dbg = validate_curve_damage(
        glyph_clean,
        final_y,
        removable,
        stroke_radius=stroke_radius,
        inner_plot=inner,
        original_bgr=image_bgr,
    )

    # Success only when residual ghosting and curve gaps are both clean.
    max_gap = max((d.get("max_gap_run", 0) for d in damages), default=0)
    if removed == 0 and removable:
        status = STATUS_FAILED
    elif residuals or damages or max_gap > CURVE_GAP_COLS:
        status = STATUS_PARTIAL
    else:
        status = STATUS_SUCCESS

    overlay = build_preprocessed_overlay(
        image_bgr,
        removed_mask=applied,
        preserved_axis_mask=preserved,
        protected_curve_mask=protected,
    )

    notes = [
        "plot_only.png remains the untouched crop; preprocessed images are separate.",
        "Pipeline: DP curve_y[x] -> full glyph erase -> pale wipe -> stroke reconstruction.",
        "protected_curve_mask is a thin centerline band (not full CCs).",
        f"status={status}; residual={len(residuals)}; damage={len(damages)}; "
        f"stroke_radius={stroke_radius}",
    ]

    return PreprocessTextRemovalResult(
        preprocessed_bgr=_crop(glyph_clean, plot_bbox),
        glyph_bgr=_crop(glyph_clean, plot_bbox),
        region_bgr=_crop(region_clean, plot_bbox),
        overlay_bgr=_crop(overlay, plot_bbox),
        residual_debug_bgr=_crop(residual_dbg, plot_bbox),
        curve_damage_debug_bgr=_crop(damage_dbg, plot_bbox),
        removal_mask_used=_crop(applied, plot_bbox),
        protected_curve_mask=_crop(protected, plot_bbox),
        preserved_axis_mask=_crop(preserved, plot_bbox),
        expanded_glyph_mask=_crop(expanded, plot_bbox),
        curve_y=final_y,
        removal_method=removal_method,
        status=status,
        removed_pixel_count=removed,
        protected_curve_pixel_count=int(np.count_nonzero(protected)),
        residual_text_groups=len(residuals),
        curve_damage_groups=len(damages),
        notes=notes,
        meta={
            "region_removal_method": region_method,
            "region_removed_pixel_count": region_removed,
            "region_note": "Destructive diagnostic fill without reconstruction.",
            "plot_bbox": list(plot_bbox) if plot_bbox else None,
            "stroke_radius": stroke_radius,
            "residual_groups": residuals,
            "curve_damage_groups_detail": damages,
            "max_curve_gap_run": int(max_gap),
            "reconstruction": "dp_centerline_hermite_stroke",
        },
    )


# Back-compat helpers used by older tests.
def detect_protected_curve_pixels(
    image_bgr: np.ndarray,
    removal_mask: np.ndarray,
    *,
    inner_plot: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rem = _as_u8_mask(removal_mask, gray.shape)
    curve_y = estimate_curve_y_per_column(
        gray, inner_plot=inner_plot, text_penalty_mask=rem
    )
    radius = estimate_local_stroke_radius(gray, curve_y)
    return build_centerline_mask(curve_y, gray.shape, radius=radius)


def protect_curve_path_through_box(
    gray: np.ndarray,
    box: Sequence[float],
    *,
    image_shape: tuple[int, int],
    orientation: str | None = None,
) -> np.ndarray:
    del orientation
    h, w = image_shape
    curve_y = estimate_curve_y_per_column(gray, inner_plot=(0, 0, w, h))
    x0, _y0, x1, _y1 = _clamp_box(box, w, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(x0, x1):
        y = curve_y[x]
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        mask[max(0, yi - 1) : min(h, yi + 2), x] = 255
    return mask


def build_expanded_glyph_mask_from_detections(
    image_bgr: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    preserved_axis_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Compatibility wrapper: returns (glyph_mask, centerline_band, info)."""
    glyph = build_complete_glyph_removal_mask(
        image_bgr, detections, preserved_axis_mask=preserved_axis_mask
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    curve_y = estimate_curve_y_per_column(gray, text_penalty_mask=glyph)
    protected = build_centerline_mask(curve_y, gray.shape, radius=1)
    info = [
        {
            "bbox": d.get("bbox"),
            "role": d.get("role"),
            "orientation": d.get("orientation"),
            "n_components": d.get("n_components"),
            "glyph_pixels": int(np.count_nonzero(glyph)),
            "protected_pixels": int(np.count_nonzero(protected)),
        }
        for d in detections
        if str(d.get("role") or "").startswith(REMOVABLE_PREFIX)
    ]
    return glyph, protected, info
