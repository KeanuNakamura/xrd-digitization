"""Tests for in-plot text removal + curve reconstruction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
for path in (str(LEGACY), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from preprocess_text_removal import (  # noqa: E402
    apply_removal_method,
    build_centerline_mask,
    build_complete_glyph_removal_mask,
    estimate_curve_y_per_column,
    preprocess_removable_text,
    reconstruct_curve_stroke,
)


class TextRemovalTests(unittest.TestCase):
    def _synthetic_plot(self) -> tuple[np.ndarray, list[dict], np.ndarray]:
        image = np.full((220, 320, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 20), (300, 170), (0, 0, 0), 2)
        # Curve with a peak under a vertical label.
        for x in range(50, 290):
            y = 130 - int(40 * np.exp(-((x - 160) ** 2) / 280.0))
            image[y : y + 2, x] = (0, 0, 0)

        members = []
        for i, ch in enumerate(["(", "2", "1", "1", ")"]):
            org = (152, 40 + i * 16)
            cv2.putText(
                image,
                ch,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            members.append([150.0, 28.0 + i * 16, 174.0, 46.0 + i * 16])

        det = {
            "bbox": [148.0, 28.0, 176.0, 120.0],
            "role": "removable_peak_annotation",
            "orientation": "vertical",
            "alignment": "vertical",
            "n_components": 5,
            "member_boxes": members,
        }
        preserved = np.zeros(image.shape[:2], dtype=np.uint8)
        preserved[175:210, 40:300] = 255
        cv2.putText(
            image,
            "20",
            (120, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return image, [det], preserved

    def test_estimate_curve_y_is_continuous(self) -> None:
        image, _dets, preserved = self._synthetic_plot()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        curve_y = estimate_curve_y_per_column(
            gray,
            inner_plot=(40, 20, 300, 170),
            axis_penalty_mask=preserved,
        )
        valid = curve_y[50:290]
        self.assertTrue(np.isfinite(valid).all())
        # Peak should rise above baseline.
        self.assertLess(float(np.nanmin(valid)), float(np.nanmedian(valid)) - 10)

    def test_centerline_mask_is_thin(self) -> None:
        image, _dets, _ = self._synthetic_plot()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        curve_y = estimate_curve_y_per_column(gray, inner_plot=(40, 20, 300, 170))
        band = build_centerline_mask(curve_y, gray.shape, radius=1)
        # Thin band coverage should be tiny vs full plot.
        self.assertLess(float((band > 0).mean()), 0.03)

    def test_full_glyph_erase_then_reconstruct(self) -> None:
        image, dets, preserved = self._synthetic_plot()
        glyph = build_complete_glyph_removal_mask(
            image, dets, preserved_axis_mask=preserved
        )
        erased, applied, removed = apply_removal_method(
            image, glyph, method="local_background", protected_curve_mask=None
        )
        self.assertGreater(removed, 40)
        # After erase, label column should be mostly white.
        x0, y0, x1, y1 = [int(v) for v in dets[0]["bbox"]]
        self.assertGreater(float(erased[y0:y1, x0:x1, 0].mean()), 230.0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        curve_y = estimate_curve_y_per_column(
            gray, inner_plot=(40, 20, 300, 170), text_penalty_mask=glyph
        )
        reconstructed, final_y = reconstruct_curve_stroke(
            erased,
            image,
            curve_y,
            removal_mask=applied,
            detections=dets,
            stroke_radius=1,
            inner_plot=(40, 20, 300, 170),
        )
        # Reconstructed stroke should restore dark ink near the peak.
        band = reconstructed[80:140, 150:175, 0]
        self.assertLess(float(band.min()), 100.0)
        # Continuous centerline through the annotation x-span.
        span_y = final_y[150:175]
        self.assertTrue(np.isfinite(span_y).all())
        for x in range(150, 175):
            yi = int(round(final_y[x]))
            self.assertLess(int(reconstructed[yi, x, 0]), 100)

    def test_preprocess_reports_partial_or_better(self) -> None:
        image, dets, preserved = self._synthetic_plot()
        region = np.zeros(image.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = [int(v) for v in dets[0]["bbox"]]
        region[y0:y1, x0:x1] = 255
        result = preprocess_removable_text(
            image,
            detections=dets,
            region_mask=region,
            preserved_axis_mask=preserved,
            plot_bbox=(0, 0, 320, 220),
            inner_plot_bbox=(40, 20, 300, 170),
            removal_method="local_background",
        )
        self.assertIn(result.status, {"success", "partial", "failed"})
        self.assertIsNotNone(result.curve_y)
        self.assertGreater(result.removed_pixel_count, 0)
        self.assertIsNotNone(result.residual_debug_bgr)
        self.assertIsNotNone(result.curve_damage_debug_bgr)
        # Axis tick preserved.
        self.assertLess(float(result.glyph_bgr[185:200, 120:145, 0].min()), 80.0)

    def test_mask_only_does_not_modify_pixels(self) -> None:
        image, dets, preserved = self._synthetic_plot()
        result = preprocess_removable_text(
            image,
            detections=dets,
            preserved_axis_mask=preserved,
            removal_method="mask_only",
        )
        self.assertEqual(result.removed_pixel_count, 0)
        np.testing.assert_array_equal(result.glyph_bgr, image)


if __name__ == "__main__":
    unittest.main()
