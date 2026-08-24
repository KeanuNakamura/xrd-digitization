"""Regression tests for OCR tick calibration and PlotDigitizer x-axis remapping."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plotdigitizer_pipeline import (  # noqa: E402
    PlotDigitizerCalibration,
    PlotDigitizerPoints,
    _csv_quality_score,
    _plotdigitizer_quality_acceptable,
    _plotdigitizer_x_output_range,
    build_plotdigitizer_calibration,
    calibration_to_plotdigitizer_points,
    digitize_figure_image,
)
from xrd_digitization.calibrate_axes import (  # noqa: E402
    _filter_monotonic_x_tick_pairs,
    _infer_snap_step,
    _resolve_nearby_tick_conflicts,
    _snap_tick_values,
    calibrate_axes,
)
from xrd_digitization.crop_plot_area import crop_plot_area  # noqa: E402


FIGURES_DIR = ROOT / "data" / "figures"
FIGURES_WITH_TEXT_DIR = ROOT / "data" / "figures_with_text"
CNRS_DIR = ROOT / "data" / "CNRS"
PEAK_ERROR_TOLERANCE_DEG = 1.0


def _identity_pd_calibration() -> PlotDigitizerCalibration:
    return PlotDigitizerCalibration(
        points=PlotDigitizerPoints(
            data_points=[(0.0, 0.0), (30.0, 0.0), (0.0, 500.0)],
            locations=[(0, 0), (100, 0), (0, 100)],
        ),
        image_height=1000,
        x_true_range=(20.0, 50.0),
        y_true_range=(0.0, 500.0),
        x_pd_range=(0.0, 30.0),
        y_pd_range=(0.0, 500.0),
        x_anchor_ticks=(20.0, 50.0),
    )


def _peak_two_theta(csv_path: Path) -> float:
    data = np.loadtxt(csv_path)
    if data.ndim == 1:
        data = data.reshape(1, 2)
    return float(data[data[:, 1].argmax(), 0])


def _cnrs_peak(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    x = np.array(payload["two_theta_values"], dtype=float)
    y = np.array(payload["intensities"], dtype=float)
    return float(x[y.argmax()])


class TickSnapTests(unittest.TestCase):
    def test_pattern_3_snap_preserves_five_degree_grid(self) -> None:
        """OCR ticks for pattern_3 should stay on 5,10,15,20,25,30 after snap."""
        raw_pairs = [
            (114, 5.0),
            (630, 10.0),
            (1146, 15.0),
            (1662, 20.0),
            (2178, 25.0),
            (2694, 30.0),
            (2280, 7.0),  # ghost outlier
        ]
        pairs = _resolve_nearby_tick_conflicts(raw_pairs)
        pairs = _snap_tick_values(pairs)
        values = [value for _, value in pairs]
        self.assertEqual(values, [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    def test_infer_snap_step_prefers_integer_five(self) -> None:
        pairs = [(114, 5.0), (630, 10.0), (1146, 15.0), (1662, 20.0), (2178, 25.0)]
        self.assertEqual(_infer_snap_step(pairs), 5.0)

    def test_uneven_gaps_do_not_invent_fifteen_degree_snap(self) -> None:
        """Missed labels (40,60,70) must not snap onto a fabricated 15° grid."""
        pairs = [(505, 40.0), (1012, 60.0), (1266, 70.0)]
        self.assertEqual(_infer_snap_step(pairs), 10.0)
        snapped = _snap_tick_values(pairs)
        self.assertEqual([value for _, value in snapped], [40.0, 60.0, 70.0])

    def test_misread_midpoint_tick_is_repaired(self) -> None:
        """OCR reading 50 as 30 at the midpoint between 40 and 60 should repair."""
        raw = [(505, 40.0), (759, 30.0), (1012, 60.0), (1266, 70.0)]
        repaired = _filter_monotonic_x_tick_pairs(raw)
        self.assertEqual(
            [value for _, value in repaired],
            [40.0, 50.0, 60.0, 70.0],
        )


@unittest.skipUnless(shutil.which("tesseract"), "tesseract not installed")
class OcrCalibrationTests(unittest.TestCase):
    def test_pattern_3_ocr_ticks_on_five_degree_grid(self) -> None:
        image = cv2.imread(str(FIGURES_DIR / "pattern_3.png"))
        self.assertIsNotNone(image)
        crop = crop_plot_area(image)
        result = calibrate_axes(crop, full_image_bgr=image)
        values = [round(value) for _, value in result.tick_pairs]
        for expected in (5, 10, 15, 20, 25, 30):
            self.assertIn(expected, values)

    def test_figure_9_clean_preserves_forty_degree_tick(self) -> None:
        """Regression: uneven OCR gaps must not snap the 40° label to 45°."""
        image_path = FIGURES_WITH_TEXT_DIR / "figure_9_clean.png"
        if not image_path.is_file():
            self.skipTest(f"missing {image_path}")
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image)
        crop = crop_plot_area(image)
        result = calibrate_axes(crop, full_image_bgr=image)
        values = [round(value) for _, value in result.tick_pairs]
        self.assertIn(40, values)
        self.assertNotIn(45, values)
        self.assertAlmostEqual(result.x_min, 20.0, delta=2.0)
        # 40° tick should land near the calibrated mapping of its pixel.
        tick_40 = next(px for px, value in result.tick_pairs if round(value) == 40)
        fit_x = (
            result.x_min
            + (tick_40 - result.plot_left)
            / max(result.plot_right - result.plot_left, 1)
            * (result.x_max - result.x_min)
        )
        self.assertAlmostEqual(fit_x, 40.0, delta=0.75)


class PlotDigitizerCalibrationTests(unittest.TestCase):
    def test_x_pd_range_matches_plotdigitizer_transform(self) -> None:
        image = cv2.imread(str(FIGURES_DIR / "pattern_3.png"))
        self.assertIsNotNone(image)
        crop = crop_plot_area(image)
        axis = calibrate_axes(crop, full_image_bgr=image)
        x0, y0, _, _ = crop.bbox
        pd_cal = build_plotdigitizer_calibration(
            axis,
            image_height=image.shape[0],
            frame_offset_x=x0,
            frame_offset_y=y0,
            full_image_bgr=image,
            crop_bbox=crop.bbox,
        )
        expected = _plotdigitizer_x_output_range(pd_cal.points)
        self.assertAlmostEqual(pd_cal.x_pd_range[0], expected[0], places=3)
        self.assertAlmostEqual(pd_cal.x_pd_range[1], expected[1], places=3)
        self.assertGreater(pd_cal.x_pd_range[1], pd_cal.x_pd_range[0])

    def test_calibration_wrapper_returns_points(self) -> None:
        image = cv2.imread(str(FIGURES_DIR / "pattern_0.png"))
        self.assertIsNotNone(image)
        crop = crop_plot_area(image)
        axis = calibrate_axes(crop, full_image_bgr=image)
        points = calibration_to_plotdigitizer_points(axis, image_height=image.shape[0])
        self.assertEqual(len(points.data_points), 3)
        self.assertEqual(len(points.locations), 3)


class PlotDigitizerQualityScoreTests(unittest.TestCase):
    def test_prefers_smooth_trace_over_midscale_plateaus(self) -> None:
        """Mid-gray cleanup ghosts create mid-scale plateaus; score must reject them."""
        cal = _identity_pd_calibration()
        x = np.linspace(0.0, 30.0, 400)
        good_y = 20.0 + 120.0 * np.exp(-0.5 * ((x - 15.0) / 1.5) ** 2)
        bad_y = good_y.copy()
        bad_y[::5] = 250.0

        out_dir = ROOT / "data" / "figures" / "_quality_score_tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        good_csv = out_dir / "good.csv"
        bad_csv = out_dir / "bad.csv"
        np.savetxt(good_csv, np.column_stack([x, good_y]))
        np.savetxt(bad_csv, np.column_stack([x, bad_y]))

        self.assertGreater(
            _csv_quality_score(good_csv, cal),
            _csv_quality_score(bad_csv, cal),
        )
        self.assertTrue(_plotdigitizer_quality_acceptable(good_csv, cal))
        self.assertFalse(_plotdigitizer_quality_acceptable(bad_csv, cal))


@unittest.skipUnless(shutil.which("plotdigitizer"), "plotdigitizer not installed")
class BenchmarkDigitizationTests(unittest.TestCase):
    def test_peak_positions_within_one_degree(self) -> None:
        for index in range(4):
            image_path = FIGURES_DIR / f"pattern_{index}.png"
            truth_path = CNRS_DIR / f"pattern_{index}.json"
            self.assertTrue(image_path.is_file(), image_path)
            self.assertTrue(truth_path.is_file(), truth_path)

            out_dir = FIGURES_DIR / f"pattern_{index}_digitized_test"
            result = digitize_figure_image(
                image_path,
                out_dir,
                figure_id=f"pattern_{index}",
            )
            self.assertTrue(result.bands, f"no bands for pattern_{index}")
            band = result.bands[0]
            self.assertTrue(band.success, band.error or band.warnings)

            truth_peak = _cnrs_peak(truth_path)
            digitized_peak = _peak_two_theta(band.csv_path)
            error = digitized_peak - truth_peak
            self.assertLess(
                abs(error),
                PEAK_ERROR_TOLERANCE_DEG,
                f"pattern_{index}: truth={truth_peak:.2f} "
                f"digitized={digitized_peak:.2f} err={error:+.2f}",
            )


if __name__ == "__main__":
    unittest.main()
