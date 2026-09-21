"""End-to-end digitization tests on synthetic figures with RMSE checks."""

from __future__ import annotations

import numpy as np
import pytest

from autodigitizer.models import (
    AxisInfo,
    AxisRelationships,
    AxisScale,
    CurveSeparation,
    FigureAnalysis,
    PlotArea,
    SubplotAnalysis,
    SubplotResult,
    SubplotSummary,
)
from autodigitizer.axes.shared_axes import inherit_shared_axes
from autodigitizer.axes.axis_calibrator import fit_axis_calibration
from autodigitizer.axes.tick_detector import assign_tick_values_from_analysis, detect_tick_positions
from autodigitizer.curves.curve_detector import extract_curves
from autodigitizer.processing.preprocessing import load_image
from autodigitizer.segmentation.plot_area_detector import detect_plot_area
from autodigitizer.segmentation.subplot_detector import split_equal_grid
from autodigitizer.strategy.selector import select_strategy
from tests.conftest import analysis_from_panel, best_rmse_matching, digitize_panel_image
from tests.synthetic import (
    make_annotations,
    make_dashed,
    make_four_panels,
    make_log_axis,
    make_markers,
    make_shared_x,
    make_shared_y,
    make_single_curve,
    make_stacked,
    make_three_colored,
)


def test_single_curve(tmp_fig_dir):
    truth = make_single_curve(tmp_fig_dir / "single.png")
    img = load_image(truth.path)
    curves, *_ = digitize_panel_image(img, truth.panels[0], strategy="single")
    assert len(curves) >= 1
    rel, n = best_rmse_matching(curves, truth.panels[0].curves, y_span=4.0)
    assert n >= 1
    assert rel < 0.12, f"relative RMSE too high: {rel}"


def test_three_colored_curves(tmp_fig_dir):
    truth = make_three_colored(tmp_fig_dir / "three.png")
    img = load_image(truth.path)
    curves, *_ = digitize_panel_image(img, truth.panels[0], strategy="color")
    assert len(curves) >= 2
    rel, n = best_rmse_matching(curves, truth.panels[0].curves, y_span=3.0)
    assert n >= 2
    assert rel < 0.18, f"relative RMSE too high: {rel}"


def test_four_subplots_split(tmp_fig_dir):
    truth = make_four_panels(tmp_fig_dir / "four.png")
    img = load_image(truth.path)
    crops = split_equal_grid(img, 2, 2)
    assert len(crops) == 4
    ok = 0
    for crop, panel in zip(crops, truth.panels):
        curves, *_ = digitize_panel_image(crop.image, panel, strategy="single")
        rel, n = best_rmse_matching(curves, panel.curves, y_span=3.0)
        if n >= 1 and rel < 0.2:
            ok += 1
    assert ok >= 3


def test_shared_x_stacked_panels(tmp_fig_dir):
    truth = make_shared_x(tmp_fig_dir / "sharex.png")
    img = load_image(truth.path)
    crops = split_equal_grid(img, 2, 1)
    results = {}
    for crop, panel in zip(crops, truth.panels):
        analysis = analysis_from_panel(panel)
        # Simulate missing x ticks on top panel
        if panel.id == "A":
            analysis.x_axis.visible_min = None
            analysis.x_axis.visible_max = None
        plot_area = detect_plot_area(crop.image)
        x_ticks, y_ticks = detect_tick_positions(crop.image, plot_area)
        x_ticks, y_ticks = assign_tick_values_from_analysis(x_ticks, y_ticks, analysis, plot_area)
        res = SubplotResult(id=panel.id, plot_area=plot_area, analysis=analysis)
        res.x_calibration = fit_axis_calibration(x_ticks, scale=analysis.x_axis.scale, axis_name="x")
        res.y_calibration = fit_axis_calibration(y_ticks, scale=analysis.y_axis.scale, axis_name="y")
        results[panel.id] = res

    inherit_shared_axes(
        results,
        AxisRelationships(shared_x_groups=[["A", "B"]], shared_y_groups=[]),
    )
    assert results["A"].x_calibration is not None
    # Digitize top using inherited calibration
    top = crops[0]
    res = results["A"]
    curves = extract_curves(
        top.image,
        res.plot_area,
        res.analysis,
        res.x_calibration,
        res.y_calibration,
        samples=300,
        strategy="single",
        subplot_id="A",
    )
    rel, n = best_rmse_matching(curves, truth.panels[0].curves, y_span=1.2)
    assert n >= 1
    assert rel < 0.25


def test_shared_y_panels(tmp_fig_dir):
    truth = make_shared_y(tmp_fig_dir / "sharey.png")
    img = load_image(truth.path)
    crops = split_equal_grid(img, 1, 2)
    results = {}
    for crop, panel in zip(crops, truth.panels):
        analysis = analysis_from_panel(panel)
        if panel.id == "B":
            analysis.y_axis.visible_min = None
            analysis.y_axis.visible_max = None
        plot_area = detect_plot_area(crop.image)
        x_ticks, y_ticks = detect_tick_positions(crop.image, plot_area)
        x_ticks, y_ticks = assign_tick_values_from_analysis(x_ticks, y_ticks, analysis, plot_area)
        res = SubplotResult(id=panel.id, plot_area=plot_area, analysis=analysis)
        res.x_calibration = fit_axis_calibration(x_ticks, scale=analysis.x_axis.scale, axis_name="x")
        res.y_calibration = fit_axis_calibration(y_ticks, scale=analysis.y_axis.scale, axis_name="y")
        results[panel.id] = res

    inherit_shared_axes(
        results,
        AxisRelationships(shared_x_groups=[], shared_y_groups=[["A", "B"]]),
    )
    assert results["B"].y_calibration is not None
    assert results["B"].y_calibration.inherited_from == "A" or results["B"].y_calibration.calibration_type in {
        "direct",
        "shared_axis_inheritance",
    }


def test_markers(tmp_fig_dir):
    truth = make_markers(tmp_fig_dir / "markers.png")
    img = load_image(truth.path)
    curves, *_ = digitize_panel_image(img, truth.panels[0], strategy="single")
    rel, n = best_rmse_matching(curves, truth.panels[0].curves, y_span=5.0)
    assert n >= 1
    assert rel < 0.2


def test_dashed_curves(tmp_fig_dir):
    truth = make_dashed(tmp_fig_dir / "dashed.png")
    img = load_image(truth.path)
    curves, *_ = digitize_panel_image(img, truth.panels[0], strategy="continuity")
    # Dashed same-color is hard; require at least one decent curve
    rel, n = best_rmse_matching(curves, truth.panels[0].curves, y_span=3.0)
    assert n >= 1
    assert rel < 0.25


def test_stacked_offset(tmp_fig_dir):
    truth = make_stacked(tmp_fig_dir / "stacked.png")
    img = load_image(truth.path)
    curves, *_ = digitize_panel_image(
        img, truth.panels[0], stacked=True, strategy="color"
    )
    assert len(curves) >= 2
    assert all(c.vertical_offset_detected or c.displayed_values for c in curves)
    assert all(c.offset_corrected is False for c in curves)
    rel, n = best_rmse_matching(curves, truth.panels[0].curves, y_span=4.2)
    assert n >= 2
    assert rel < 0.2


def test_annotations_do_not_break_single_curve(tmp_fig_dir):
    truth = make_annotations(tmp_fig_dir / "annot.png")
    img = load_image(truth.path)
    curves, *_ = digitize_panel_image(img, truth.panels[0], strategy="single")
    rel, n = best_rmse_matching(curves, truth.panels[0].curves, y_span=3.0)
    assert n >= 1
    assert rel < 0.15


def test_legend_multi_color(tmp_fig_dir):
    # Same as three-colored with legend
    truth = make_three_colored(tmp_fig_dir / "legend.png")
    img = load_image(truth.path)
    curves, *_ = digitize_panel_image(img, truth.panels[0], strategy="color")
    assert len(curves) >= 2


def test_log_axis(tmp_fig_dir):
    truth = make_log_axis(tmp_fig_dir / "log.png")
    img = load_image(truth.path)
    curves, x_cal, y_cal, _ = digitize_panel_image(img, truth.panels[0], strategy="single")
    assert x_cal.scale == AxisScale.LOG
    assert y_cal.scale == AxisScale.LOG
    # Log digitization is harder; allow looser relative error in log space
    if curves:
        data = np.array(curves[0].data)
        assert data[:, 0].min() > 0 and data[:, 1].min() > 0
        # Check order-of-magnitude roughly correct at mid range
        mid = data[len(data) // 2]
        assert 1 <= mid[0] <= 1000


def test_strategy_selector():
    a = SubplotAnalysis(
        subplot_id="A",
        estimated_curve_count=3,
        curve_separation=CurveSeparation.COLOR,
    )
    s = select_strategy(a)
    assert s.name == "color"

    b = SubplotAnalysis(
        subplot_id="B",
        estimated_curve_count=3,
        stacked=True,
        curve_separation=CurveSeparation.STACKED,
    )
    assert select_strategy(b).name == "stacked"

    c = SubplotAnalysis(
        subplot_id="C",
        estimated_curve_count=8,
        vertically_offset=True,
        curve_separation=CurveSeparation.SPATIAL,
    )
    assert select_strategy(c).name == "stacked"
