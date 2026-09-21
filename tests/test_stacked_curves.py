"""Tests for stacked multi-curve helpers (no ClipDrop / no OpenAI)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xrd_digitization.stacked_curves import (  # noqa: E402
    CurveLabelHint,
    associate_labels_with_bands,
    baseline_corrected_xy,
    detect_curve_bands,
    digitize_stacked_figure,
    isolate_curve_band,
    remove_vertical_guides,
    validate_band_separation,
)
from xrd_digitization.types import AxisCalibrationResult  # noqa: E402

MULTI_CURVE = ROOT / "data" / "multi_curve"
FIGURE_1_CLEAN = MULTI_CURVE / "figure_1" / "figure_1_clean.png"
FIGURE_1_ORIG = MULTI_CURVE / "figure_1" / "figure_1.png"
FIGURE_3_CLEAN = MULTI_CURVE / "figure_3" / "figure_3_clean.png"
FIGURE_3_ORIG = MULTI_CURVE / "figure_3" / "figure_3.png"


def _synthetic_stacked(n_curves: int = 4, width: int = 240, band_h: int = 40) -> np.ndarray:
    """White canvas with dark horizontal baselines in separate bands."""
    height = n_curves * band_h + 20
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    # Frame
    cv2.rectangle(img, (10, 5), (width - 10, height - 5), (0, 0, 0), 1)
    for i in range(n_curves):
        y = 10 + i * band_h + band_h // 2
        cv2.line(img, (20, y), (width - 20, y), (0, 0, 200 - 20 * i), 2)
        # Small peak
        peak_x = 80 + i * 15
        pts = np.array(
            [[peak_x - 4, y], [peak_x, y - 12], [peak_x + 4, y]],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts], False, (0, 0, 200 - 20 * i), 2)
    return img


class ValidateBandsTests(unittest.TestCase):
    def test_success_for_comparable_bands(self) -> None:
        bands = [(10, 40), (50, 80), (90, 120)]
        result = validate_band_separation(bands, estimated_curve_count=3, plot_height=130)
        self.assertEqual(result.status, "success")

    def test_low_confidence_count_mismatch(self) -> None:
        bands = [(10, 40), (50, 80)]
        result = validate_band_separation(bands, estimated_curve_count=8, plot_height=100)
        self.assertEqual(result.status, "low_confidence")
        self.assertIn("estimated 8", result.reason or "")

    def test_label_association_by_order(self) -> None:
        bands = [(0, 10), (10, 20), (20, 30)]
        baselines = [5.0, 15.0, 25.0]
        labels = [
            CurveLabelHint("715 nm", 0),
            CurveLabelHint("660 nm", 1),
            CurveLabelHint("641 nm", 2),
        ]
        assigned = associate_labels_with_bands(bands, baselines, labels)
        self.assertEqual(assigned, ["715 nm", "660 nm", "641 nm"])


class GuideRemovalTests(unittest.TestCase):
    def test_removes_tall_thin_line(self) -> None:
        img = np.full((200, 120, 3), 255, dtype=np.uint8)
        # Light-gray guide (low saturation).
        cv2.line(img, (60, 10), (60, 190), (160, 160, 160), 1)
        cleaned = remove_vertical_guides(img)
        center = cleaned[:, 60]
        self.assertGreater(int((center > 250).sum()), 150)

    def test_preserves_colored_peak(self) -> None:
        img = np.full((200, 120, 3), 255, dtype=np.uint8)
        # Saturated red peak stroke should survive.
        cv2.line(img, (40, 40), (40, 160), (0, 0, 220), 2)
        cleaned = remove_vertical_guides(img)
        self.assertTrue(bool((cleaned[:, 40, 2] > 100).any()))


class IsolationTests(unittest.TestCase):
    def test_rect_isolation_shape(self) -> None:
        img = _synthetic_stacked()
        iso = isolate_curve_band(img, 20, 50, curve_id=0, baseline_px=35.0)
        self.assertEqual(iso.mode, "rect")
        self.assertEqual(iso.image_bgr.shape[0], iso.crop_y1 - iso.crop_y0)
        self.assertIsNone(iso.mask)

    def test_baseline_corrected_xy(self) -> None:
        xy = [[10.0, 5.0], [11.0, 5.2], [12.0, 20.0]]
        corrected = baseline_corrected_xy(xy)
        self.assertEqual(len(corrected), 3)
        # Baseline near 5 → first points near 0.
        self.assertLess(abs(corrected[0][1]), 1.0)


@unittest.skipUnless(FIGURE_3_CLEAN.is_file(), "multi_curve figure_3 clean missing")
class Figure3StackedIntegrationTests(unittest.TestCase):
    def test_detects_multiple_bands(self) -> None:
        from xrd_digitization.calibrate_axes import calibrate_axes
        from xrd_digitization.crop_plot_area import crop_plot_area

        bgr = cv2.imread(str(FIGURE_3_CLEAN))
        self.assertIsNotNone(bgr)
        crop = crop_plot_area(bgr)
        calibration = calibrate_axes(crop, full_image_bgr=bgr)
        bands = detect_curve_bands(crop.cropped_bgr, calibration, estimated_curve_count=3)
        self.assertGreaterEqual(len(bands), 2)

    def test_digitize_stacked_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = digitize_stacked_figure(
                FIGURE_3_ORIG if FIGURE_3_ORIG.is_file() else FIGURE_3_CLEAN,
                out,
                figure_id="figure_3",
                cleaned_image=FIGURE_3_CLEAN,
                estimated_curve_count=3,
                curve_labels=[
                    {"text": "(c)", "vertical_order": 0},
                    {"text": "(b)", "vertical_order": 1},
                    {"text": "(a)", "vertical_order": 2},
                ],
                debug=True,
            )
            self.assertTrue((out / "figure_3_stacked.json").is_file())
            payload = json.loads((out / "figure_3_stacked.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["plot_type"], "stacked_xrd")
            self.assertIn(payload["status"], {"success", "low_confidence"})
            self.assertGreaterEqual(len(payload["curves"]), 2)
            self.assertIn("x_axis", payload)
            self.assertTrue((out / "figure_3_stacked.csv").is_file())
            self.assertTrue((out / "digitized_stacked.png").is_file())
            self.assertTrue((out / "reconstructed_overlay.png").is_file())
            succeeded = sum(1 for curve in payload["curves"] if curve.get("success"))
            self.assertGreaterEqual(succeeded, 1)
            self.assertTrue((out / "debug" / "detected_baselines.png").is_file())
            self.assertIsNotNone(result.x_axis.get("min"))
            self.assertIsNotNone(result.x_axis.get("max"))


@unittest.skipUnless(FIGURE_1_CLEAN.is_file(), "multi_curve figure_1 clean missing")
class Figure1StackedIntegrationTests(unittest.TestCase):
    def test_digitize_stacked_figure_1(self) -> None:
        labels = [
            CurveLabelHint(f"{nm} nm", i)
            for i, nm in enumerate([715, 712, 703, 691, 678, 660, 649, 641])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = digitize_stacked_figure(
                FIGURE_1_ORIG if FIGURE_1_ORIG.is_file() else FIGURE_1_CLEAN,
                out,
                figure_id="figure_1",
                cleaned_image=FIGURE_1_CLEAN,
                estimated_curve_count=8,
                curve_labels=labels,
                debug=True,
            )
            self.assertTrue((out / "figure_1_stacked.json").is_file())
            self.assertTrue((out / "figure_1_stacked.csv").is_file())
            self.assertTrue((out / "digitized_stacked.png").is_file())
            self.assertGreaterEqual(len(result.curves), 2)
            # Soft: allow low_confidence if count is off, but expect some successes.
            succeeded = sum(1 for c in result.curves if c.success)
            self.assertGreaterEqual(succeeded, 1)


if __name__ == "__main__":
    unittest.main()
