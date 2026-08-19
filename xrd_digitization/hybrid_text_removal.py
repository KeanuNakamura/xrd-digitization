"""Curve-preserving text removal guided by agent text regions.

Inside each agent text box:
  - identify text strokes,
  - preserve only pixels that belong to a continuous curve entering and leaving
    the box,
  - remove remaining text pixels,
  - reconstruct missing curve pixels by interpolating between left/right
    boundaries.

For vertical labels above peaks, removal is limited to the label region above
the detected peak apex whenever possible.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from xrd_digitization.agent_guidance import AgentTextRegion

LOGGER = logging.getLogger(__name__)

BG_MARGIN = 12.0
CURVE_INK_THRESH = 170
CURVE_BAND_PX = 2


def create_text_mask(
    shape_hw: tuple[int, int],
    text_regions: Sequence[AgentTextRegion | dict],
    *,
    pad: int = 2,
) -> np.ndarray:
    """Build a white-on-black union mask from crop-local text bounding boxes."""
    height, width = shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in text_regions:
        if isinstance(region, AgentTextRegion):
            box = region.bbox
        else:
            box = region.get("bbox") if isinstance(region, dict) else None
        if not box or len(box) < 4:
            continue
        x0 = max(0, int(box[0]) - pad)
        y0 = max(0, int(box[1]) - pad)
        x1 = min(width, int(box[2]) + pad)
        y1 = min(height, int(box[3]) + pad)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _region_bbox(region: AgentTextRegion | dict) -> list[float] | None:
    if isinstance(region, AgentTextRegion):
        return list(region.bbox)
    if isinstance(region, dict):
        box = region.get("bbox")
        return list(box) if box and len(box) >= 4 else None
    return None


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


def _detect_peak_apex_in_box(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> int | None:
    """
    Find the peak apex (image-y of a thin dark spike) inside/near box.

    Prefers tips near the *bottom* of the label box that continue as a curve
    outside the box — not the top of vertical text glyphs.
    """
    h, w = gray.shape
    # Search slightly below the box — peak tip may sit just under the label.
    sy0 = max(0, y0)
    sy1 = min(h, y1 + max(16, (y1 - y0) // 2))
    sx0, sx1 = max(0, x0), min(w, x1)
    if sx1 - sx0 < 2 or sy1 - sy0 < 4:
        return None

    best: tuple[float, int] | None = None
    for x in range(sx0, sx1):
        col = gray[sy0:sy1, x]
        dark = col < CURVE_INK_THRESH
        y = 0
        while y < len(dark):
            if not dark[y]:
                y += 1
                continue
            yb = y
            while yb < len(dark) and dark[yb]:
                yb += 1
            run = yb - y
            tip = sy0 + y
            # Curve tip heuristic: short/medium run near lower half of search,
            # with dark ink continuing below the box (peak stem / baseline).
            below = gray[min(h - 1, y1) : min(h, y1 + 25), x]
            continues = bool(below.size and np.any(below < CURVE_INK_THRESH))
            if 2 <= run <= 20 and continues and tip >= y0 + (y1 - y0) * 0.25:
                # Prefer lower tips (closer to curve body) over text tops.
                score = tip + min(run, 12) * 0.5
                if best is None or score > best[0]:
                    best = (score, tip)
            y = yb
    return None if best is None else best[1]


def _column_peak_tip(
    gray: np.ndarray,
    x: int,
    *,
    y_top: int,
    y_bottom: int,
    max_run: int = 12,
) -> float | None:
    """Uppermost short dark run tip in a column (peak apex), or None."""
    h = gray.shape[0]
    y_a = max(0, y_top)
    y_b = min(h, y_bottom)
    if y_b - y_a < 4 or not (0 <= x < gray.shape[1]):
        return None
    col = gray[y_a:y_b, x]
    dark = col < CURVE_INK_THRESH
    y = 0
    best: tuple[float, float] | None = None
    while y < len(dark):
        if not dark[y]:
            y += 1
            continue
        yb = y
        while yb < len(dark) and dark[yb]:
            yb += 1
        run = yb - y
        if 2 <= run <= max_run:
            tip = float(y_a + y)
            score = -tip + run * 0.05
            if best is None or score > best[0]:
                best = (score, tip)
        y = yb
    return None if best is None else best[1]


def _curve_y_crossing_box(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    curve_hint_y: np.ndarray | None = None,
    peak_apex: int | None = None,
) -> np.ndarray:
    """Estimate curve y(x) through a box from outside anchors + local ink.

    When a peak apex sits in/near the box, prefer the apex (not a flat
    linear bridge that creates rectangular plateaus).
    """
    h, w = gray.shape
    xs = np.arange(max(0, x0), min(w, x1), dtype=int)
    out = np.full(w, np.nan, dtype=np.float32)

    def _anchor(side: str) -> tuple[int, float] | None:
        if side == "left":
            for x in range(x0 - 1, max(-1, x0 - 40), -1):
                if x < 0:
                    break
                if curve_hint_y is not None and np.isfinite(curve_hint_y[x]):
                    return x, float(curve_hint_y[x])
                tip = _column_peak_tip(gray, x, y_top=max(0, y0 - 40), y_bottom=min(h, y1 + 40))
                if tip is not None:
                    return x, tip
                col = gray[:, x]
                ys = np.where(col < CURVE_INK_THRESH)[0]
                ys = ys[(ys >= max(0, y0 - 30)) & (ys <= min(h - 1, y1 + 30))]
                if len(ys):
                    return x, float(np.median(ys))
        else:
            for x in range(x1, min(w, x1 + 40)):
                if curve_hint_y is not None and np.isfinite(curve_hint_y[x]):
                    return x, float(curve_hint_y[x])
                tip = _column_peak_tip(gray, x, y_top=max(0, y0 - 40), y_bottom=min(h, y1 + 40))
                if tip is not None:
                    return x, tip
                col = gray[:, x]
                ys = np.where(col < CURVE_INK_THRESH)[0]
                ys = ys[(ys >= max(0, y0 - 30)) & (ys <= min(h - 1, y1 + 30))]
                if len(ys):
                    return x, float(np.median(ys))
        return None

    left = _anchor("left")
    right = _anchor("right")
    if left is None and right is None:
        return out
    if left is None:
        left = right
    if right is None:
        right = left
    assert left is not None and right is not None
    xL, yL = left
    xR, yR = right
    span = max(1, xR - xL)

    # Search for a peak tip inside the box (above the side anchors).
    search_bottom = min(h, (peak_apex + 8) if peak_apex is not None else y1 + 20)
    tip_xs: list[tuple[int, float]] = []
    for x in xs:
        tip = _column_peak_tip(
            gray,
            x,
            y_top=max(0, min(y0, int(min(yL, yR)) - 80)),
            y_bottom=search_bottom,
            max_run=14,
        )
        if tip is not None and tip < min(yL, yR) - 3:
            tip_xs.append((x, tip))
    peak_x = None
    peak_y = None
    if tip_xs:
        # Sharpest (highest) tip near box center.
        tip_xs.sort(key=lambda t: (t[1], abs(t[0] - 0.5 * (x0 + x1))))
        peak_x, peak_y = tip_xs[0]
    elif peak_apex is not None:
        peak_x = (x0 + x1) // 2
        peak_y = float(peak_apex)

    for x in xs:
        t = (x - xL) / span
        y_lin = yL + t * (yR - yL)
        # Piecewise bridge through peak apex when present (no flat plateau).
        if peak_x is not None and peak_y is not None and xL < peak_x < xR:
            if x <= peak_x:
                t2 = (x - xL) / max(1, peak_x - xL)
                y_bridge = yL + t2 * (peak_y - yL)
            else:
                t2 = (x - peak_x) / max(1, xR - peak_x)
                y_bridge = peak_y + t2 * (yR - peak_y)
        else:
            y_bridge = y_lin

        tip = _column_peak_tip(
            gray,
            x,
            y_top=max(0, int(min(y_bridge, y_lin)) - 30),
            y_bottom=min(h, int(max(y_bridge, y_lin)) + 20),
            max_run=14,
        )
        if tip is not None and tip <= y_bridge + 2:
            y_use = tip
        else:
            yi = int(round(y_bridge))
            y_a = max(0, yi - 5)
            y_b = min(h, yi + 6)
            col = gray[y_a:y_b, x]
            dark = np.where(col < CURVE_INK_THRESH)[0]
            if dark.size:
                pick = int(dark[np.argmin(np.abs(dark - (yi - y_a)))])
                cand = float(y_a + pick)
                # Reject tall vertical glyph stems.
                run_mask = gray[:, x] < CURVE_INK_THRESH
                yy = int(cand)
                ya, yb = yy, yy
                while ya > 0 and run_mask[ya - 1]:
                    ya -= 1
                while yb + 1 < h and run_mask[yb + 1]:
                    yb += 1
                y_use = y_bridge if (yb - ya + 1) > 16 else cand
            else:
                y_use = y_bridge
        out[x] = float(y_use)
    return out


def _fill_with_local_background(
    image_bgr: np.ndarray,
    removal_mask: np.ndarray,
) -> np.ndarray:
    cleaned = image_bgr.copy()
    if not np.any(removal_mask):
        return cleaned
    # Fast path: inpaint for large masks; pixel medians for small.
    n = int(np.count_nonzero(removal_mask))
    if n > 800:
        return cv2.inpaint(image_bgr, removal_mask, 2, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ys, xs = np.where(removal_mask > 0)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 4), min(h, y + 5)
        x0, x1 = max(0, x - 4), min(w, x + 5)
        patch = image_bgr[y0:y1, x0:x1]
        keep = removal_mask[y0:y1, x0:x1] == 0
        if np.any(keep):
            cleaned[y, x] = np.median(patch[keep], axis=0).astype(np.uint8)
        else:
            cleaned[y, x] = (255, 255, 255)
    return cleaned


def _paint_curve_stroke(
    image_bgr: np.ndarray,
    curve_y: np.ndarray,
    xs: Sequence[int],
    *,
    radius: int = CURVE_BAND_PX,
    sample_gray: np.ndarray | None = None,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for x in xs:
        if x < 0 or x >= w:
            continue
        y = curve_y[x] if x < len(curve_y) else np.nan
        if not np.isfinite(y):
            continue
        yi = int(round(float(y)))
        if not (0 <= yi < h):
            continue
        intensity = 30
        if sample_gray is not None:
            ya, yb = max(0, yi - 2), min(h, yi + 3)
            vals = sample_gray[ya:yb, x]
            dark = vals[vals < CURVE_INK_THRESH]
            if dark.size:
                intensity = int(np.median(dark))
        out[max(0, yi - radius) : min(h, yi + radius + 1), x] = (
            intensity,
            intensity,
            intensity,
        )
    return out


def remove_text_preserve_curve(
    image_bgr: np.ndarray,
    text_mask: np.ndarray,
    *,
    text_regions: Sequence[AgentTextRegion | dict] | None = None,
    curve_hint_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove text ink inside agent boxes while preserving the continuous curve.

    Returns ``(cleaned_bgr, removal_mask)``.
    """
    if text_mask is None or not np.any(text_mask):
        return image_bgr.copy(), np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    if text_mask.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(
            f"text_mask shape {text_mask.shape[:2]} != image {image_bgr.shape[:2]}"
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    removal = np.zeros((height, width), dtype=np.uint8)
    reconstruct_y = (
        curve_hint_y.astype(np.float32).copy()
        if curve_hint_y is not None
        else np.full(width, np.nan, dtype=np.float32)
    )

    # Build per-box work list from regions or connected components of text_mask.
    boxes: list[tuple[int, int, int, int]] = []
    if text_regions:
        for region in text_regions:
            box = _region_bbox(region)
            if not box:
                continue
            x0 = max(0, int(np.floor(box[0])))
            y0 = max(0, int(np.floor(box[1])))
            x1 = min(width, int(np.ceil(box[2])))
            y1 = min(height, int(np.ceil(box[3])))
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1, y1))
    if not boxes:
        num, _labels, stats, _ = cv2.connectedComponentsWithStats(
            (text_mask > 0).astype(np.uint8), connectivity=8
        )
        for label in range(1, num):
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            boxes.append((x0, y0, x0 + bw, y0 + bh))

    paint_xs: set[int] = set()

    for x0, y0, x1, y1 in boxes:
        # Limit vertical labels to above peak apex when possible.
        apex = _detect_peak_apex_in_box(gray, x0, y0, x1, y1)
        box_y1 = y1
        label_above_peak = False
        if apex is not None and apex > y0 + 4:
            # Keep removal above apex; do not erase the peak itself.
            box_y1 = min(y1, max(y0 + 4, apex - 2))
            label_above_peak = box_y1 <= apex

        if box_y1 <= y0:
            continue

        preserve = np.zeros((box_y1 - y0, x1 - x0), dtype=np.uint8)
        if label_above_peak:
            # Vertical Miller labels sit entirely above the peak tip: wipe the
            # whole box. Do not paint a horizontal stroke across the box width
            # (that creates rectangular plateaus); DP will recover the tip.
            pass
        else:
            curve_y = _curve_y_crossing_box(
                gray,
                x0,
                y0,
                x1,
                box_y1,
                curve_hint_y=reconstruct_y,
                peak_apex=apex,
            )
            for x in range(x0, x1):
                y = curve_y[x]
                if not np.isfinite(y):
                    continue
                yi = int(round(float(y)))
                if y0 <= yi < box_y1:
                    local_y = yi - y0
                    a = max(0, local_y - CURVE_BAND_PX)
                    b = min(box_y1 - y0, local_y + CURVE_BAND_PX + 1)
                    preserve[a:b, x - x0] = 255
                    reconstruct_y[x] = float(y)
                    paint_xs.add(x)

        roi = gray[y0:box_y1, x0:x1]
        # Wipe any non-white pixel in the label box that is not on the curve band.
        rem_local = ((roi < 250).astype(np.uint8) * 255)
        rem_local = cv2.bitwise_and(rem_local, cv2.bitwise_not(preserve))
        removal[y0:box_y1, x0:x1] = cv2.bitwise_or(
            removal[y0:box_y1, x0:x1], rem_local
        )

    # Restrict removal to the provided text_mask union (safety).
    removal = cv2.bitwise_and(removal, (text_mask > 0).astype(np.uint8) * 255)

    LOGGER.debug(
        "Text removal: removing %d pixels across %d box(es)",
        int(np.count_nonzero(removal)),
        len(boxes),
    )
    cleaned = _fill_with_local_background(image_bgr, removal)
    if paint_xs:
        cleaned = _paint_curve_stroke(
            cleaned,
            reconstruct_y,
            sorted(paint_xs),
            sample_gray=gray,
        )
    return cleaned, removal
