"""Shared pytest fixtures and digitization helpers for tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodigitizer.axes.axis_calibrator import fit_axis_calibration
from autodigitizer.axes.tick_detector import assign_tick_values_from_analysis, detect_tick_positions
from autodigitizer.curves.curve_detector import extract_curves
from autodigitizer.models import (
    AxisInfo,
    AxisScale,
    CurveSeparation,
    FigureAnalysis,
    SubplotAnalysis,
    SubplotSummary,
)
from autodigitizer.processing.preprocessing import load_image
from autodigitizer.segmentation.plot_area_detector import detect_plot_area
from autodigitizer.segmentation.subplot_detector import detect_subplots, split_equal_grid
from tests.synthetic import FigureTruth, PanelTruth


def analysis_from_panel(panel: PanelTruth, *, stacked: bool = False) -> SubplotAnalysis:
    n = len(panel.curves)
    colors = {c.color for c in panel.curves}
    if stacked:
        sep = CurveSeparation.STACKED
    elif n == 1:
        sep = CurveSeparation.SINGLE
    elif len(colors) > 1:
        sep = CurveSeparation.COLOR
    else:
        sep = CurveSeparation.SPATIAL
    return SubplotAnalysis(
        subplot_id=panel.id,
        digitizable=True,
        confidence=0.9,
        plot_type="multi_line" if n > 1 else "line",
        x_axis=AxisInfo(
            label=panel.xlabel or "x",
            scale=AxisScale.LOG if panel.xscale == "log" else AxisScale.LINEAR,
            visible_min=panel.xlim[0],
            visible_max=panel.xlim[1],
            confidence=0.95,
        ),
        y_axis=AxisInfo(
            label=panel.ylabel or "y",
            scale=AxisScale.LOG if panel.yscale == "log" else AxisScale.LINEAR,
            visible_min=panel.ylim[0],
            visible_max=panel.ylim[1],
            confidence=0.95,
        ),
        estimated_curve_count=n,
        curve_separation=sep,
        stacked=stacked,
        vertically_offset=stacked,
        legend_present=panel.legend,
        annotations_present=panel.annotate is not None,
    )


def digitize_panel_image(
    image: np.ndarray,
    panel: PanelTruth,
    *,
    samples: int = 400,
    stacked: bool = False,
    strategy: str | None = None,
):
    analysis = analysis_from_panel(panel, stacked=stacked)
    plot_area = detect_plot_area(image)
    x_ticks, y_ticks = detect_tick_positions(image, plot_area)
    x_ticks, y_ticks = assign_tick_values_from_analysis(x_ticks, y_ticks, analysis, plot_area)
    x_cal = fit_axis_calibration(x_ticks, scale=analysis.x_axis.scale, axis_name="x")
    y_cal = fit_axis_calibration(y_ticks, scale=analysis.y_axis.scale, axis_name="y")
    assert x_cal is not None and y_cal is not None
    strat = strategy
    if strat is None:
        if stacked:
            strat = "stacked"
        elif analysis.curve_separation == CurveSeparation.COLOR:
            strat = "color"
        elif analysis.estimated_curve_count == 1:
            strat = "single"
        else:
            strat = "continuity"
    curves = extract_curves(
        image,
        plot_area,
        analysis,
        x_cal,
        y_cal,
        samples=samples,
        strategy=strat,
        color_clusters_override=analysis.estimated_curve_count if strat == "color" else None,
        subplot_id=panel.id,
    )
    return curves, x_cal, y_cal, plot_area


def rmse_against_truth(extracted_xy: list[tuple[float, float]], tx: np.ndarray, ty: np.ndarray) -> float:
    if not extracted_xy:
        return float("inf")
    ex = np.array([p[0] for p in extracted_xy])
    ey = np.array([p[1] for p in extracted_xy])
    # Interpolate truth onto extracted x (clipped to truth domain)
    mask = (ex >= tx.min()) & (ex <= tx.max())
    if mask.sum() < 10:
        return float("inf")
    ex, ey = ex[mask], ey[mask]
    order = np.argsort(tx)
    ty_i = np.interp(ex, tx[order], ty[order])
    return float(np.sqrt(np.mean((ey - ty_i) ** 2)))


def best_rmse_matching(
    curves, truths: list, y_span: float
) -> tuple[float, int]:
    """Greedy match extracted curves to truth; return mean relative RMSE and matched count."""
    remaining = list(enumerate(truths))
    scores = []
    for curve in curves:
        best_i = None
        best = float("inf")
        for idx, (ti, t) in enumerate(remaining):
            err = rmse_against_truth(curve.data, t.x, t.y)
            if err < best:
                best = err
                best_i = idx
        if best_i is not None and best < float("inf"):
            scores.append(best / max(y_span, 1e-9))
            remaining.pop(best_i)
    if not scores:
        return float("inf"), 0
    return float(np.mean(scores)), len(scores)


@pytest.fixture
def tmp_fig_dir(tmp_path):
    d = tmp_path / "figs"
    d.mkdir()
    return d
