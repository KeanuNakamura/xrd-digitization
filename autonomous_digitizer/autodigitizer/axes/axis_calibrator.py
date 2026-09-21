"""Fit pixel↔data axis calibrations from labeled ticks."""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from autodigitizer.models import AxisCalibration, AxisScale, TickMark

logger = logging.getLogger(__name__)


def fit_axis_calibration(
    ticks: list[TickMark],
    scale: AxisScale = AxisScale.LINEAR,
    axis_name: str = "x",
) -> Optional[AxisCalibration]:
    labeled = [(t.pixel, t.value) for t in ticks if t.value is not None]
    if len(labeled) < 2:
        logger.warning("Insufficient labeled ticks for %s axis", axis_name)
        return None

    pixels = np.array([p for p, _ in labeled], dtype=float)
    values = np.array([v for _, v in labeled], dtype=float)

    if scale == AxisScale.LOG:
        if np.any(values <= 0):
            logger.warning("Non-positive values on log %s axis; falling back to linear", axis_name)
            scale = AxisScale.LINEAR
        else:
            values = np.log10(values)

    # data_mapped = a * pixel + b
    A = np.vstack([pixels, np.ones_like(pixels)]).T
    try:
        coef, residuals, _, _ = np.linalg.lstsq(A, values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b = float(coef[0]), float(coef[1])
    pred = a * pixels + b
    rmse = float(np.sqrt(np.mean((pred - values) ** 2)))

    # Confidence from residual relative to data span
    span = float(np.ptp(values)) or 1.0
    conf = float(np.clip(1.0 - rmse / span * 3.0, 0.1, 0.99))

    data_vals = [t.value for t in ticks if t.value is not None]
    cal = AxisCalibration(
        scale=scale,
        a=a,
        b=b,
        ticks=ticks,
        residual_rmse=rmse,
        confidence=conf,
        source="local_fit",
        calibration_type="direct",
        data_min=min(data_vals) if data_vals else None,
        data_max=max(data_vals) if data_vals else None,
    )
    logger.debug(
        "Calibrated %s: a=%.6g b=%.6g rmse=%.4g conf=%.2f scale=%s",
        axis_name,
        a,
        b,
        rmse,
        conf,
        scale.value,
    )
    return cal


def calibration_from_range(
    pixel_low: float,
    pixel_high: float,
    data_at_low: float,
    data_at_high: float,
    scale: AxisScale = AxisScale.LINEAR,
) -> AxisCalibration:
    """Two-point calibration (edges of plot rectangle)."""
    ticks = [
        TickMark(pixel=pixel_low, value=data_at_low, confidence=0.7),
        TickMark(pixel=pixel_high, value=data_at_high, confidence=0.7),
    ]
    cal = fit_axis_calibration(ticks, scale=scale)
    assert cal is not None
    return cal


def relative_axis_calibration(
    plot_top: float,
    plot_bottom: float,
    *,
    data_at_top: float = 1.0,
    data_at_bottom: float = 0.0,
    scale: AxisScale = AxisScale.LINEAR,
) -> AxisCalibration:
    """Fallback for unlabeled axes (e.g. XRD intensity in a.u.).

    Maps the plot rectangle to a relative data range. Does not invent absolute units.
    """
    cal = calibration_from_range(
        plot_top,
        plot_bottom,
        data_at_top,
        data_at_bottom,
        scale=scale,
    )
    cal.source = "relative_fallback"
    cal.calibration_type = "relative_unlabeled_axis"
    cal.confidence = min(cal.confidence, 0.55)
    return cal
