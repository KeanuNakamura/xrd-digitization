"""Guide-aware curve tracing using OpenAI bands and sparse keypoints."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from autodigitizer.curves.curve_tracer import resample_points
from autodigitizer.models import CurveGuide, PlotArea, PlotExtractionGuide

logger = logging.getLogger(__name__)


def plot_norm_to_pixel(x_n: float, y_n: float, plot_area: PlotArea) -> tuple[float, float]:
    x = plot_area.left + x_n * plot_area.width
    y = plot_area.top + y_n * plot_area.height
    return float(x), float(y)


def interpolate_guide_y(key_points_normalized: list[list[float]], x_n: float) -> Optional[float]:
    if len(key_points_normalized) < 2:
        return None
    pts = sorted(
        ((float(p[0]), float(p[1])) for p in key_points_normalized if len(p) >= 2),
        key=lambda t: t[0],
    )
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if x_n < xs.min() - 0.02 or x_n > xs.max() + 0.02:
        return None
    return float(np.interp(x_n, xs, ys))


def band_mask(
    plot_area: PlotArea, y_band: list[float], shape: tuple[int, ...], pad: float = 0.01
) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    y0 = max(0.0, min(y_band) - pad)
    y1 = min(1.0, max(y_band) + pad)
    py0 = int(plot_area.top + y0 * plot_area.height)
    py1 = int(plot_area.top + y1 * plot_area.height)
    mask[py0:py1, plot_area.left : plot_area.right] = True
    return mask


def color_proximity_mask(
    image: np.ndarray,
    target_rgb: list[int] | tuple[int, int, int],
    ink: np.ndarray,
    max_dist: float = 55.0,
) -> np.ndarray:
    # Near-black / gray targets: color distance is unreliable; keep ink as-is
    r, g, b = [int(v) for v in target_rgb[:3]]
    if max(r, g, b) < 40 or (abs(r - g) < 15 and abs(g - b) < 15 and max(r, g, b) < 80):
        return ink
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = np.uint8([[list(target_rgb[:3])]])
    tlab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.sqrt(((lab - tlab) ** 2).sum(axis=2))
    return ink & (dist <= max_dist)


def largest_horizontal_component(ink: np.ndarray, plot_area: PlotArea) -> np.ndarray:
    """Keep the dominant horizontally spanning CC — removes isolated text glyphs."""
    roi = ink[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right].astype(np.uint8)
    if roi.sum() == 0:
        return ink
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    dil = cv2.dilate(roi, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    if n <= 1:
        return ink
    best = None
    best_score = -1.0
    for i in range(1, n):
        _x, _y, bw, _bh, area = stats[i]
        span = bw / max(1, roi.shape[1])
        score = span * 2.0 + area / max(1, roi.size)
        if span < 0.25:
            score *= 0.2
        if score > best_score:
            best_score = score
            best = i
    keep = labels == best
    out = np.zeros_like(ink)
    out[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right] = (
        ink[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right] & keep
    )
    return out


def _column_runs(col_idxs: np.ndarray) -> list[tuple[int, int]]:
    if len(col_idxs) == 0:
        return []
    runs = []
    start = int(col_idxs[0])
    prev = start
    for v in col_idxs[1:]:
        v = int(v)
        if v - prev > 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    return runs


def guided_column_trace(
    ink: np.ndarray,
    plot_area: PlotArea,
    guide: CurveGuide,
    *,
    max_jump: float = 28.0,
    image: np.ndarray | None = None,
    snap_radius: float = 12.0,
) -> list[tuple[float, float]]:
    """Trace one curve constrained to a y-band, biased toward OpenAI keypoints.

    When keypoints are dense enough, walk the guide polyline and snap each sample
    to nearby ink inside the band (reduces text contamination).
    """
    band = band_mask(plot_area, guide.y_band_normalized, ink.shape)
    local = ink & band

    # Dense keypoint path: snap guide samples to local ink
    if len(guide.key_points_normalized) >= 8:
        xs = np.linspace(0.0, 1.0, max(plot_area.width, 64))
        points: list[tuple[float, float]] = []
        prev_y = None
        for x_n in xs:
            g_n = interpolate_guide_y(guide.key_points_normalized, float(x_n))
            if g_n is None:
                continue
            x = plot_area.left + float(x_n) * plot_area.width
            gy = plot_area.top + g_n * plot_area.height
            xi = int(round(x))
            if xi < plot_area.left or xi >= plot_area.right:
                continue
            col = np.where(local[plot_area.top : plot_area.bottom, xi])[0]
            if len(col):
                ys = plot_area.top + col.astype(float)
                near = ys[np.abs(ys - gy) <= snap_radius]
                if len(near):
                    y = float(near[np.argmin(np.abs(near - gy))])
                else:
                    y = float(gy)
            else:
                y = float(gy)
            if prev_y is not None and abs(y - prev_y) > max_jump * 2:
                y = float(gy)
            points.append((float(x), y))
            prev_y = y
        if len(points) >= 10:
            return points

    if local.sum() < 10 and guide.key_points_normalized:
        return [
            plot_norm_to_pixel(p[0], p[1], plot_area)
            for p in sorted(guide.key_points_normalized, key=lambda q: q[0])
            if len(p) >= 2
        ]

    gray = None
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    w = plot_area.width
    points = []
    prev_y = None

    for xi in range(w):
        x = plot_area.left + xi
        x_n = xi / max(1, w - 1)
        col = np.where(local[plot_area.top : plot_area.bottom, x])[0]
        guide_y_n = interpolate_guide_y(guide.key_points_normalized, x_n)
        guide_y = (
            plot_area.top + guide_y_n * plot_area.height if guide_y_n is not None else None
        )

        if len(col) == 0:
            if guide_y is not None and (prev_y is None or abs(guide_y - prev_y) <= max_jump * 2):
                points.append((float(x), float(guide_y)))
                prev_y = float(guide_y)
            continue

        ys = plot_area.top + col.astype(float)
        runs = _column_runs(col)
        run_ys = [plot_area.top + 0.5 * (a + b) for a, b in runs]

        def score_y(y: float) -> float:
            s = 0.0
            if guide_y is not None:
                s -= abs(y - guide_y)
            if prev_y is not None:
                s -= 0.5 * abs(y - prev_y)
            if gray is not None:
                yy = int(np.clip(round(y), 0, gray.shape[0] - 1))
                s -= 0.05 * float(gray[yy, x])
            return s

        if guide_y is not None:
            y = (
                float(max(run_ys, key=score_y))
                if run_ys
                else float(ys[np.argmin(np.abs(ys - guide_y))])
            )
            if abs(y - guide_y) > snap_radius:
                # Prefer the guide over far ink (often text)
                y = float(guide_y)
        elif prev_y is not None:
            y = float(max(run_ys, key=lambda yy: -abs(yy - prev_y))) if run_ys else float(np.median(ys))
            if abs(y - prev_y) > max_jump:
                near = [yy for yy in run_ys if abs(yy - prev_y) <= max_jump]
                y = float(np.median(near)) if near else float(np.median(ys))
        else:
            if gray is not None and run_ys:
                y = float(max(run_ys, key=lambda yy: -float(gray[int(round(yy)), x])))
            else:
                y = float(np.median(ys))

        if prev_y is not None and abs(y - prev_y) > max_jump * 1.5:
            if guide_y is not None:
                y = float(guide_y)
            else:
                continue

        points.append((float(x), float(y)))
        prev_y = y

    if len(points) < 5 and guide.key_points_normalized:
        return [
            plot_norm_to_pixel(p[0], p[1], plot_area)
            for p in sorted(guide.key_points_normalized, key=lambda q: q[0])
            if len(p) >= 2
        ]
    return points


def merge_guide_polyline(
    traced: list[tuple[float, float]],
    guide: CurveGuide,
    plot_area: PlotArea,
    samples: int,
) -> list[tuple[float, float]]:
    """Resample the CV trace. Ignore unreliable keypoints."""
    if traced:
        return resample_points(traced, samples)
    guide_px = [
        plot_norm_to_pixel(p[0], p[1], plot_area)
        for p in guide.key_points_normalized
        if len(p) >= 2
    ]
    return resample_points(guide_px, samples) if guide_px else []



def upper_envelope_trace(
    ink: np.ndarray,
    plot_area: PlotArea,
    *,
    band: np.ndarray | None = None,
    gap_fill: int = 6,
) -> list[tuple[float, float]]:
    """Per-column topmost ink pixel — correct for thin line plots / XRD peaks."""
    local = ink if band is None else (ink & band)
    roi = local[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right]
    h, w = roi.shape
    xs: list[int] = []
    ys: list[float] = []
    for x in range(w):
        col = np.where(roi[:, x])[0]
        if len(col) == 0:
            continue
        xs.append(x)
        ys.append(float(col.min()))  # topmost in image coords
    if len(xs) < 2:
        return []
    filled_x: list[float] = []
    filled_y: list[float] = []
    for i in range(len(xs)):
        filled_x.append(float(xs[i]))
        filled_y.append(ys[i])
        if i + 1 < len(xs):
            gap = xs[i + 1] - xs[i]
            if 1 < gap <= gap_fill:
                for g in range(1, gap):
                    alpha = g / gap
                    filled_x.append(xs[i] + g)
                    filled_y.append(ys[i] * (1 - alpha) + ys[i + 1] * alpha)
    return [
        (plot_area.left + fx, plot_area.top + fy) for fx, fy in zip(filled_x, filled_y)
    ]


def keypoint_ink_hit_rate(
    guide: CurveGuide,
    ink: np.ndarray,
    plot_area: PlotArea,
    tol: int = 4,
) -> float:
    if not guide.key_points_normalized:
        return 0.0
    hits = 0
    for p in guide.key_points_normalized:
        if len(p) < 2:
            continue
        x, y = plot_norm_to_pixel(p[0], p[1], plot_area)
        xi, yi = int(round(x)), int(round(y))
        y0, y1 = max(0, yi - tol), min(ink.shape[0], yi + tol + 1)
        x0, x1 = max(0, xi - tol), min(ink.shape[1], xi + tol + 1)
        if ink[y0:y1, x0:x1].any():
            hits += 1
    return hits / max(1, len(guide.key_points_normalized))


def extract_guided_curves(
    image: np.ndarray,
    ink: np.ndarray,
    plot_area: PlotArea,
    guide: PlotExtractionGuide,
) -> list[tuple[list[tuple[float, float]], CurveGuide]]:
    """Extract curves using OpenAI bands/text; CV does the stroke tracing.

    Keypoints are used only when they actually land on ink; otherwise we fall back
    to upper-envelope / band-local tracing (critical for annotated XRD peaks).
    """
    out = []
    multi = len(guide.curves) > 1
    for i, cg in enumerate(guide.curves):
        local_ink = ink
        if cg.color_rgb is not None and len(cg.color_rgb) >= 3:
            prox = color_proximity_mask(image, cg.color_rgb, ink, max_dist=70.0)
            if prox.sum() > 30:
                local_ink = prox

        band = band_mask(plot_area, cg.y_band_normalized, ink.shape, pad=0.02)
        hit = keypoint_ink_hit_rate(cg, local_ink, plot_area)

        if not multi or (max(cg.y_band_normalized) - min(cg.y_band_normalized) > 0.55):
            # Single / wide-band: keep dominant CC then upper envelope
            cleaned = largest_horizontal_component(local_ink & band, plot_area)
            pts = upper_envelope_trace(cleaned, plot_area, band=band)
            if len(pts) < 10:
                pts = upper_envelope_trace(local_ink, plot_area, band=band)
        else:
            # Narrow stacked band
            pts = upper_envelope_trace(local_ink, plot_area, band=band)
            if len(pts) < 10:
                pts = guided_column_trace(local_ink, plot_area, cg, image=image)

        # Optionally refine with validated keypoints — never shrink x-span vs CV envelope
        if hit >= 0.55 and cg.key_points_normalized:
            guide_pts = guided_column_trace(local_ink, plot_area, cg, image=image)
            if len(guide_pts) >= 20 and pts:
                cv_span = max(p[0] for p in pts) - min(p[0] for p in pts)
                g_span = max(p[0] for p in guide_pts) - min(p[0] for p in guide_pts)
                if g_span >= 0.85 * cv_span:
                    pts = guide_pts

        logger.debug(
            "guided curve %d label=%s hit=%.2f pts=%d band=%s",
            i,
            cg.label,
            hit,
            len(pts),
            cg.y_band_normalized,
        )
        out.append((pts, cg))
    return out
