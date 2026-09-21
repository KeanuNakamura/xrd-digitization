"""Detect tick mark positions along axes using local CV."""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from autodigitizer.models import AxisScale, PlotArea, TickMark, SubplotAnalysis
from autodigitizer.processing.preprocessing import save_image

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(
    r"[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?|\d+\.?\d*\s*[×xX]\s*10\^?[+\-]?\d+"
)


def _parse_number(text: str) -> Optional[float]:
    text = text.strip().replace(",", "")
    text = text.replace("−", "-").replace("–", "-")
    # scientific unicode
    text = text.replace("×", "x").replace("·", "")
    m = _NUM_RE.search(text)
    if not m:
        return None
    token = m.group(0).replace(" ", "")
    if "x10" in token.lower() or "x10" in token:
        token = re.sub(r"[xX]10\^?", "e", token)
    try:
        return float(token)
    except ValueError:
        return None


def _tick_peaks(profile: np.ndarray, min_distance: int = 8) -> list[int]:
    """Simple peak picker on a 1D projection."""
    if profile.size == 0:
        return []
    thr = max(0.08, float(np.percentile(profile, 75)))
    peaks = []
    for i in range(1, len(profile) - 1):
        if profile[i] >= thr and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
            elif profile[i] > profile[peaks[-1]]:
                peaks[-1] = i
    return peaks


def detect_tick_positions(image: np.ndarray, plot_area: PlotArea) -> tuple[list[TickMark], list[TickMark]]:
    """Detect x (bottom) and y (left) tick stub positions in pixel coords of `image`."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    # Bottom strip under plot
    y0 = min(h - 1, plot_area.bottom)
    y1 = min(h, plot_area.bottom + max(8, int(0.08 * h)))
    bottom = gray[y0:y1, plot_area.left : plot_area.right]
    # Left strip
    x0 = max(0, plot_area.left - max(8, int(0.12 * w)))
    x1 = max(x0 + 1, plot_area.left)
    left = gray[plot_area.top : plot_area.bottom, x0:x1]

    x_ticks: list[TickMark] = []
    y_ticks: list[TickMark] = []

    if bottom.size:
        # Tick stubs: short dark vertical segments -> column projection of dark
        dark = (bottom < 100).astype(np.float32)
        col = dark.mean(axis=0)
        for p in _tick_peaks(col, min_distance=max(6, bottom.shape[1] // 25)):
            x_ticks.append(TickMark(pixel=float(plot_area.left + p), axis="x", confidence=0.5))

    if left.size:
        dark = (left < 100).astype(np.float32)
        row = dark.mean(axis=1)
        for p in _tick_peaks(row, min_distance=max(6, left.shape[0] // 25)):
            y_ticks.append(TickMark(pixel=float(plot_area.top + p), axis="y", confidence=0.5))

    # Always include plot corners as candidate anchors
    if not any(abs(t.pixel - plot_area.left) < 3 for t in x_ticks):
        x_ticks.insert(0, TickMark(pixel=float(plot_area.left), axis="x", confidence=0.3))
    if not any(abs(t.pixel - plot_area.right) < 3 for t in x_ticks):
        x_ticks.append(TickMark(pixel=float(plot_area.right), axis="x", confidence=0.3))
    if not any(abs(t.pixel - plot_area.top) < 3 for t in y_ticks):
        y_ticks.insert(0, TickMark(pixel=float(plot_area.top), axis="y", confidence=0.3))
    if not any(abs(t.pixel - plot_area.bottom) < 3 for t in y_ticks):
        y_ticks.append(TickMark(pixel=float(plot_area.bottom), axis="y", confidence=0.3))

    return x_ticks, y_ticks


def assign_tick_values_from_analysis(
    x_ticks: list[TickMark],
    y_ticks: list[TickMark],
    analysis: SubplotAnalysis,
    plot_area: PlotArea,
) -> tuple[list[TickMark], list[TickMark]]:
    """Map LLM-provided axis range / tick hints onto detected tick positions."""
    x_vals = list(analysis.x_axis.tick_values_hint or [])
    y_vals = list(analysis.y_axis.tick_values_hint or [])

    if analysis.x_axis.visible_min is not None and analysis.x_axis.visible_max is not None:
        if not x_vals:
            x_vals = [analysis.x_axis.visible_min, analysis.x_axis.visible_max]
        # Prefer endpoints for left/right
        x_ticks = _match_range_to_edges(
            x_ticks,
            analysis.x_axis.visible_min,
            analysis.x_axis.visible_max,
            edge_low=float(plot_area.left),
            edge_high=float(plot_area.right),
            hints=x_vals,
            scale=analysis.x_axis.scale,
        )
    elif x_vals and len(x_vals) >= 2:
        x_ticks = _distribute_values(x_ticks, sorted(x_vals))

    if analysis.y_axis.visible_min is not None and analysis.y_axis.visible_max is not None:
        if not y_vals:
            y_vals = [analysis.y_axis.visible_min, analysis.y_axis.visible_max]
        # Image y increases downward: top pixel -> visible_max for typical plots
        y_ticks = _match_range_to_edges(
            y_ticks,
            analysis.y_axis.visible_max,  # top
            analysis.y_axis.visible_min,  # bottom
            edge_low=float(plot_area.top),
            edge_high=float(plot_area.bottom),
            hints=y_vals,
            scale=analysis.y_axis.scale,
            y_axis=True,
        )
    elif y_vals and len(y_vals) >= 2:
        # Sort so larger values map toward top
        y_ticks = _distribute_values(y_ticks, sorted(y_vals, reverse=True))

    return x_ticks, y_ticks


def _match_range_to_edges(
    ticks: list[TickMark],
    val_at_low_edge: float,
    val_at_high_edge: float,
    edge_low: float,
    edge_high: float,
    hints: list[float],
    scale: AxisScale,
    y_axis: bool = False,
) -> list[TickMark]:
    """Assign values using edge range; interpolate intermediate ticks if possible."""
    out: list[TickMark] = []
    # Ensure edge ticks exist
    ticks_sorted = sorted(ticks, key=lambda t: t.pixel)
    if not ticks_sorted:
        ticks_sorted = [
            TickMark(pixel=edge_low, axis="y" if y_axis else "x"),
            TickMark(pixel=edge_high, axis="y" if y_axis else "x"),
        ]

    # Force first/last near edges
    ticks_sorted[0] = TickMark(
        pixel=edge_low,
        value=float(val_at_low_edge),
        axis=ticks_sorted[0].axis,
        confidence=0.8,
    )
    ticks_sorted[-1] = TickMark(
        pixel=edge_high,
        value=float(val_at_high_edge),
        axis=ticks_sorted[-1].axis,
        confidence=0.8,
    )

    # Interpolate intermediates in data space (or log space)
    for i, t in enumerate(ticks_sorted):
        if i == 0 or i == len(ticks_sorted) - 1:
            out.append(ticks_sorted[i])
            continue
        alpha = (t.pixel - edge_low) / max(1e-9, edge_high - edge_low)
        if scale == AxisScale.LOG:
            import math

            if val_at_low_edge > 0 and val_at_high_edge > 0:
                lv = math.log10(val_at_low_edge) * (1 - alpha) + math.log10(val_at_high_edge) * alpha
                val = 10**lv
            else:
                val = val_at_low_edge * (1 - alpha) + val_at_high_edge * alpha
        else:
            val = val_at_low_edge * (1 - alpha) + val_at_high_edge * alpha
        # Snap to nearest hint if close
        if hints:
            nearest = min(hints, key=lambda h: abs(h - val))
            if abs(nearest - val) / max(1e-9, abs(val_at_high_edge - val_at_low_edge)) < 0.08:
                val = nearest
        out.append(TickMark(pixel=t.pixel, value=float(val), axis=t.axis, confidence=0.55))
    return out


def _distribute_values(ticks: list[TickMark], values: list[float]) -> list[TickMark]:
    ticks_sorted = sorted(ticks, key=lambda t: t.pixel)
    if len(ticks_sorted) < 2 or len(values) < 2:
        return ticks
    # Map evenly by rank
    n = min(len(ticks_sorted), len(values))
    # Pick n ticks spread across
    idxs = np.linspace(0, len(ticks_sorted) - 1, n).astype(int)
    vidxs = np.linspace(0, len(values) - 1, n).astype(int)
    out = list(ticks_sorted)
    for i, ti in enumerate(idxs):
        out[ti] = TickMark(
            pixel=ticks_sorted[ti].pixel,
            value=float(values[vidxs[i]]),
            axis=ticks_sorted[ti].axis,
            confidence=0.6,
        )
    return out


def render_ticks(image: np.ndarray, plot_area: PlotArea, x_ticks: list[TickMark], y_ticks: list[TickMark], path) -> None:
    vis = image.copy()
    cv2.rectangle(vis, (plot_area.left, plot_area.top), (plot_area.right, plot_area.bottom), (0, 120, 255), 1)
    for t in x_ticks:
        x = int(t.pixel)
        cv2.line(vis, (x, plot_area.bottom), (x, plot_area.bottom + 8), (0, 200, 0), 1)
        if t.value is not None:
            cv2.putText(vis, f"{t.value:g}", (x - 10, plot_area.bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 0), 1)
    for t in y_ticks:
        y = int(t.pixel)
        cv2.line(vis, (plot_area.left - 8, y), (plot_area.left, y), (200, 0, 0), 1)
        if t.value is not None:
            cv2.putText(vis, f"{t.value:g}", (max(0, plot_area.left - 50), y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 0, 0), 1)
    save_image(path, vis)
