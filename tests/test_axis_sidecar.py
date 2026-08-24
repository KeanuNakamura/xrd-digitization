"""Tests for axis sidecar save/load and ClipDrop-clean digitize reuse."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xrd_digitization.axis_sidecar import (  # noqa: E402
    AxisSidecarError,
    assert_sidecar_matches_image,
    axis_calibration_from_dict,
    axis_calibration_to_dict,
    build_sidecar_from_image,
    extract_axis_sidecar_for_path,
    load_axis_sidecar,
    load_sidecar_for_digitize,
    resolve_axis_sidecar_path,
    save_axis_sidecar,
    x_calibration_is_usable,
)
from xrd_digitization.calibrate_axes import calibrate_axes  # noqa: E402
from xrd_digitization.crop_plot_area import crop_plot_area  # noqa: E402
from xrd_digitization.types import AxisCalibrationResult  # noqa: E402

FIGURES_WITH_TEXT = ROOT / "data" / "figures_with_text"


def _synthetic_calibration() -> AxisCalibrationResult:
    return AxisCalibrationResult(
        x_min=10.0,
        x_max=80.0,
        plot_left=5,
        plot_right=400,
        plot_top=10,
        plot_bottom=300,
        method="ocr_linear_regression",
        confidence=0.9,
        tick_pairs=[(5, 10.0), (200, 45.0), (400, 80.0)],
        y_min=0.0,
        y_max=250.0,
        y_tick_pairs=[(300, 0.0), (10, 250.0)],
        y_method="ocr_linear",
        warnings=[],
    )


class AxisSidecarRoundTripTests(unittest.TestCase):
    def test_calibration_dict_round_trip(self) -> None:
        cal = _synthetic_calibration()
        restored = axis_calibration_from_dict(axis_calibration_to_dict(cal))
        self.assertEqual(restored.x_min, cal.x_min)
        self.assertEqual(restored.x_max, cal.x_max)
        self.assertEqual(restored.tick_pairs, cal.tick_pairs)
        self.assertEqual(restored.method, cal.method)
        self.assertEqual(restored.plot_left, cal.plot_left)

    def test_sidecar_file_round_trip(self) -> None:
        cal = _synthetic_calibration()
        image = np.zeros((320, 420, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "figure_1.axes.json"
            from xrd_digitization.axis_sidecar import AxisSidecar

            sidecar = AxisSidecar(
                calibration=cal,
                image_width=420,
                image_height=320,
                plot_crop_bbox=(10, 10, 410, 310),
                source_image="figure_1.png",
                warnings=["ok"],
            )
            save_axis_sidecar(path, sidecar)
            loaded = load_axis_sidecar(path)
            self.assertEqual(loaded.image_width, 420)
            self.assertEqual(loaded.plot_crop_bbox, (10, 10, 410, 310))
            self.assertEqual(loaded.calibration.tick_pairs, cal.tick_pairs)
            assert_sidecar_matches_image(loaded, image)

    def test_shape_mismatch_raises(self) -> None:
        cal = _synthetic_calibration()
        from xrd_digitization.axis_sidecar import AxisSidecar

        sidecar = AxisSidecar(
            calibration=cal,
            image_width=100,
            image_height=100,
            plot_crop_bbox=(0, 0, 50, 50),
            source_image="x.png",
            warnings=[],
        )
        with self.assertRaises(AxisSidecarError):
            assert_sidecar_matches_image(sidecar, np.zeros((80, 100, 3), dtype=np.uint8))

    def test_unusable_default_range_rejected(self) -> None:
        cal = AxisCalibrationResult(
            x_min=5.0,
            x_max=80.0,
            plot_left=0,
            plot_right=100,
            plot_top=0,
            plot_bottom=100,
            method="default_range",
            confidence=0.35,
            tick_pairs=[],
        )
        usable, reasons = x_calibration_is_usable(cal)
        self.assertFalse(usable)
        self.assertTrue(any("method_not_ocr" in r for r in reasons))


class ResolveSidecarPathTests(unittest.TestCase):
    def test_clean_stem_resolves_base_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axes = root / "figure_1.axes.json"
            axes.write_text("{}", encoding="utf-8")
            clean = root / "figure_1_clean.png"
            clean.write_bytes(b"")
            self.assertEqual(resolve_axis_sidecar_path(clean), axes)


@unittest.skipUnless(FIGURES_WITH_TEXT.is_dir(), "figures_with_text not present")
@unittest.skipUnless(shutil.which("tesseract"), "tesseract not installed")
class FiguresWithTextSidecarTests(unittest.TestCase):
    def test_extract_and_reuse_crop_on_clean_figure_1(self) -> None:
        original = FIGURES_WITH_TEXT / "figure_1.png"
        clean = FIGURES_WITH_TEXT / "figure_1_clean.png"
        if not original.is_file() or not clean.is_file():
            self.skipTest("figure_1.png / figure_1_clean.png missing")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            orig_copy = tmp_dir / "figure_1.png"
            clean_copy = tmp_dir / "figure_1_clean.png"
            shutil.copy2(original, orig_copy)
            shutil.copy2(clean, clean_copy)

            sidecar = extract_axis_sidecar_for_path(orig_copy)
            self.assertTrue(str(sidecar.calibration.method).startswith("ocr"))
            self.assertGreaterEqual(len(sidecar.calibration.tick_pairs), 2)
            self.assertAlmostEqual(sidecar.calibration.x_min, 10.0, delta=1.5)
            self.assertAlmostEqual(sidecar.calibration.x_max, 80.0, delta=1.5)

            clean_bgr = cv2.imread(str(clean_copy))
            loaded = load_sidecar_for_digitize(clean_copy, clean_bgr)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            reused, plot_crop = loaded
            self.assertEqual(plot_crop.bbox, sidecar.plot_crop_bbox)
            self.assertEqual(reused.calibration.tick_pairs, sidecar.calibration.tick_pairs)

            # Re-cropping a fully text-stripped clean image can diverge; saved bbox must win.
            auto_crop = crop_plot_area(clean_bgr)
            if auto_crop.bbox == sidecar.plot_crop_bbox:
                self.skipTest(
                    "clean image still has enough frame context that auto-crop matches sidecar"
                )
            self.assertNotEqual(auto_crop.bbox, sidecar.plot_crop_bbox)

    def test_figure_4_x_range_near_10_80(self) -> None:
        original = FIGURES_WITH_TEXT / "figure_4.png"
        if not original.is_file():
            self.skipTest("figure_4.png missing")
        image = cv2.imread(str(original))
        sidecar = build_sidecar_from_image(image, source_image=original.name)
        usable, _ = x_calibration_is_usable(sidecar.calibration)
        self.assertTrue(usable)
        self.assertAlmostEqual(sidecar.calibration.x_min, 10.0, delta=1.5)
        self.assertAlmostEqual(sidecar.calibration.x_max, 80.0, delta=1.5)

    def test_clean_without_sidecar_falls_back_to_missing_ticks(self) -> None:
        """Legacy full-image ClipDrop cleans had no tick glyphs; OCR then fails.

        Axis-preserving cleans (inset ClipDrop) keep labels, so this only runs
        when the clean image truly has no recoverable ticks.
        """
        clean = FIGURES_WITH_TEXT / "figure_1_clean.png"
        if not clean.is_file():
            self.skipTest("figure_1_clean.png missing")
        image = cv2.imread(str(clean))
        crop = crop_plot_area(image)
        result = calibrate_axes(crop, full_image_bgr=image)
        if result.tick_pairs:
            self.skipTest(
                "clean image still has tick labels (axis-preserving ClipDrop); "
                "legacy no-tick fallback does not apply"
            )
        self.assertEqual(result.tick_pairs, [])
        self.assertFalse(str(result.method).startswith("ocr"))


@unittest.skipUnless(FIGURES_WITH_TEXT.is_dir(), "figures_with_text not present")
@unittest.skipUnless(shutil.which("tesseract"), "tesseract not installed")
@unittest.skipUnless(shutil.which("plotdigitizer"), "plotdigitizer not installed")
class CleanDigitizeWithSidecarTests(unittest.TestCase):
    def test_figure_1_clean_digitize_preserves_x_range(self) -> None:
        from plotdigitizer_pipeline import digitize_figure_image

        original = FIGURES_WITH_TEXT / "figure_1.png"
        clean = FIGURES_WITH_TEXT / "figure_1_clean.png"
        if not original.is_file() or not clean.is_file():
            self.skipTest("figure_1.png / figure_1_clean.png missing")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            shutil.copy2(original, tmp_dir / "figure_1.png")
            shutil.copy2(clean, tmp_dir / "figure_1_clean.png")
            extract_axis_sidecar_for_path(tmp_dir / "figure_1.png")

            out_dir = tmp_dir / "out"
            result = digitize_figure_image(
                tmp_dir / "figure_1_clean.png",
                out_dir,
                figure_id="figure_1_clean",
                require_axes_sidecar=True,
            )
            self.assertTrue(result.bands)
            band = result.bands[0]
            self.assertTrue(band.success, msg=band.error or band.warnings)
            self.assertIn("axes_sidecar_reused", result.warnings)
            calib = band.calibration
            self.assertAlmostEqual(float(calib["x_min"]), 10.0, delta=1.5)
            self.assertAlmostEqual(float(calib["x_max"]), 80.0, delta=1.5)
            csv_path = band.csv_path
            self.assertTrue(csv_path.is_file())
            data = np.loadtxt(csv_path)
            if data.ndim == 1:
                data = data.reshape(1, 2)
            self.assertGreaterEqual(data[:, 0].min(), 8.0)
            self.assertLessEqual(data[:, 0].max(), 82.0)
            # Tallest peak should land near the (104) reflection (~33°), not default-scale junk.
            peak_x = float(data[data[:, 1].argmax(), 0])
            self.assertGreater(peak_x, 25.0)
            self.assertLess(peak_x, 40.0)
