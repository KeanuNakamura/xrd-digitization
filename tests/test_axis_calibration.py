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
    _plotdigitizer_x_output_range,
    build_plotdigitizer_calibration,
    calibration_to_plotdigitizer_points,
    digitize_figure_image,
)
from xrd_digitization.calibrate_axes import (  # noqa: E402
    _infer_snap_step,
    _resolve_nearby_tick_conflicts,
    _snap_tick_values,
    calibrate_axes,
)
from xrd_digitization.crop_plot_area import crop_plot_area  # noqa: E402


FIGURES_DIR = ROOT / "data" / "figures"
CNRS_DIR = ROOT / "data" / "CNRS"
PEAK_ERROR_TOLERANCE_DEG = 1.0


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
