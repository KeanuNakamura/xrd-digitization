"""Shared-axis inheritance across subplots."""

from __future__ import annotations

import logging
from typing import Optional

from autodigitizer.models import (
    AxisCalibration,
    AxisRelationships,
    SubplotResult,
)
from autodigitizer.axes.axis_calibrator import calibration_from_range

logger = logging.getLogger(__name__)


def _group_map(groups: list[list[str]]) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    for g in groups:
        for sid in g:
            m[sid] = g
    return m


def inherit_shared_axes(
    results: dict[str, SubplotResult],
    relationships: AxisRelationships,
) -> None:
    """Fill missing calibrations by inheriting from shared-axis partners.

    Mutates results in place.
    """
    x_map = _group_map(relationships.shared_x_groups)
    y_map = _group_map(relationships.shared_y_groups)

    for sid, res in results.items():
        if res.plot_area is None:
            continue
        if res.x_calibration is None and sid in x_map:
            donor = _best_donor(results, x_map[sid], axis="x")
            if donor is not None:
                res.x_calibration = transfer_x_calibration(donor, res)
                res.warnings.append(
                    f"x calibration inherited from subplot {donor.id}"
                )
                logger.info("Inherited x calibration %s -> %s", donor.id, sid)
        if res.y_calibration is None and sid in y_map:
            donor = _best_donor(results, y_map[sid], axis="y")
            if donor is not None:
                res.y_calibration = transfer_y_calibration(donor, res)
                res.warnings.append(
                    f"y calibration inherited from subplot {donor.id}"
                )
                logger.info("Inherited y calibration %s -> %s", donor.id, sid)


def _best_donor(
    results: dict[str, SubplotResult],
    group: list[str],
    axis: str,
) -> Optional[SubplotResult]:
    candidates = []
    for sid in group:
        r = results.get(sid)
        if r is None or r.plot_area is None:
            continue
        cal = r.x_calibration if axis == "x" else r.y_calibration
        if cal is not None and cal.calibration_type == "direct":
            candidates.append((cal.confidence, r))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def transfer_x_calibration(donor: SubplotResult, target: SubplotResult) -> AxisCalibration:
    """Copy data-range relationship onto target's plot rectangle."""
    assert donor.x_calibration is not None and donor.plot_area and target.plot_area
    dcal = donor.x_calibration
    # Data at donor left/right
    data_left = dcal.pixel_to_data(donor.plot_area.left)
    data_right = dcal.pixel_to_data(donor.plot_area.right)
    new = calibration_from_range(
        float(target.plot_area.left),
        float(target.plot_area.right),
        data_left,
        data_right,
        scale=dcal.scale,
    )
    new.source = f"subplot_{donor.id}"
    new.calibration_type = "shared_axis_inheritance"
    new.inherited_from = donor.id
    new.confidence = min(dcal.confidence, 0.85)
    return new


def transfer_y_calibration(donor: SubplotResult, target: SubplotResult) -> AxisCalibration:
    assert donor.y_calibration is not None and donor.plot_area and target.plot_area
    dcal = donor.y_calibration
    data_top = dcal.pixel_to_data(donor.plot_area.top)
    data_bottom = dcal.pixel_to_data(donor.plot_area.bottom)
    new = calibration_from_range(
        float(target.plot_area.top),
        float(target.plot_area.bottom),
        data_top,
        data_bottom,
        scale=dcal.scale,
    )
    new.source = f"subplot_{donor.id}"
    new.calibration_type = "shared_axis_inheritance"
    new.inherited_from = donor.id
    new.confidence = min(dcal.confidence, 0.85)
    return new
