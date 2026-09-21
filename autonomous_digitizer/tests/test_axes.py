"""Unit tests for axis calibration and shared-axis inheritance."""

from __future__ import annotations

import numpy as np

from autodigitizer.axes.axis_calibrator import calibration_from_range, fit_axis_calibration
from autodigitizer.axes.shared_axes import inherit_shared_axes, transfer_x_calibration
from autodigitizer.models import (
    AxisCalibration,
    AxisRelationships,
    AxisScale,
    DigitizationStatus,
    PlotArea,
    SubplotResult,
    TickMark,
)


def test_linear_calibration_rmse():
    ticks = [
        TickMark(pixel=100, value=0, axis="x"),
        TickMark(pixel=200, value=10, axis="x"),
        TickMark(pixel=300, value=20, axis="x"),
    ]
    cal = fit_axis_calibration(ticks, scale=AxisScale.LINEAR)
    assert cal is not None
    assert abs(cal.pixel_to_data(100) - 0) < 1e-6
    assert abs(cal.pixel_to_data(300) - 20) < 1e-6
    assert cal.residual_rmse < 1e-9


def test_log_calibration():
    ticks = [
        TickMark(pixel=50, value=1, axis="x"),
        TickMark(pixel=150, value=10, axis="x"),
        TickMark(pixel=250, value=100, axis="x"),
    ]
    cal = fit_axis_calibration(ticks, scale=AxisScale.LOG)
    assert cal is not None
    assert abs(cal.pixel_to_data(50) - 1) / 1 < 0.05
    assert abs(cal.pixel_to_data(250) - 100) / 100 < 0.05


def test_y_axis_downward_pixels():
    # top pixel -> high value, bottom -> low
    cal = calibration_from_range(10, 110, 100, 0, scale=AxisScale.LINEAR)
    assert cal.pixel_to_data(10) > cal.pixel_to_data(110)


def test_relative_y_fallback():
    from autodigitizer.axes.axis_calibrator import relative_axis_calibration

    cal = relative_axis_calibration(10, 110, data_at_top=1.0, data_at_bottom=0.0)
    assert cal.calibration_type == "relative_unlabeled_axis"
    assert abs(cal.pixel_to_data(110) - 0.0) < 1e-6
    assert abs(cal.pixel_to_data(10) - 1.0) < 1e-6

    donor = SubplotResult(
        id="B",
        status=DigitizationStatus.SUCCESS,
        plot_area=PlotArea(left=50, right=250, top=20, bottom=180),
        x_calibration=calibration_from_range(50, 250, 10, 80),
    )
    target = SubplotResult(
        id="A",
        status=DigitizationStatus.PARTIAL,
        plot_area=PlotArea(left=48, right=248, top=20, bottom=180),
        x_calibration=None,
        y_calibration=calibration_from_range(20, 180, 1, 0),
    )
    results = {"A": target, "B": donor}
    inherit_shared_axes(
        results,
        AxisRelationships(shared_x_groups=[["A", "B"]], shared_y_groups=[]),
    )
    assert results["A"].x_calibration is not None
    assert results["A"].x_calibration.calibration_type == "shared_axis_inheritance"
    assert results["A"].x_calibration.inherited_from == "B"
    assert abs(results["A"].x_calibration.pixel_to_data(48) - 10) < 0.5
    assert abs(results["A"].x_calibration.pixel_to_data(248) - 80) < 0.5
