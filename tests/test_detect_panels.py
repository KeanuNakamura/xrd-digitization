"""Tests for multi-panel figure detection and cropping."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xrd_digitization.detect_panels import detect_plot_panels  # noqa: E402
from xrd_digitization.split_figure_panels import (  # noqa: E402
    call_openai_panel_locator,
    split_figure_into_panels,
    write_panel_crops,
)


def _draw_axes_frame(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
    # Fake curve ink inside the frame.
    for offset in (40, 90, 140):
        y = y0 + offset
        cv2.line(img, (x0 + 20, y), (x1 - 20, y - 15), (180, 80, 40), 2)


class DetectPanelsTests(unittest.TestCase):
    def test_side_by_side_whitespace_split(self) -> None:
        img = np.full((400, 900, 3), 255, dtype=np.uint8)
        _draw_axes_frame(img, 20, 30, 420, 360)
        _draw_axes_frame(img, 480, 30, 880, 360)
        panels = detect_plot_panels(img)
        self.assertGreaterEqual(len(panels), 2)
        labels = [p.label for p in panels]
        self.assertEqual(labels[0], "A")
        self.assertEqual(labels[1], "B")
        # Left panel should be left of right panel.
        self.assertLess(panels[0].bbox[0], panels[1].bbox[0])

    def test_vertical_stack_still_splits(self) -> None:
        img = np.full((900, 400, 3), 255, dtype=np.uint8)
        _draw_axes_frame(img, 30, 20, 370, 400)
        _draw_axes_frame(img, 30, 480, 370, 860)
        panels = detect_plot_panels(img)
        self.assertGreaterEqual(len(panels), 2)

    def test_lithium_borate_fig1_splits_two_panels(self) -> None:
        path = (
            ROOT
            / "pipeline_output"
            / "lithium_borate"
            / "figures"
            / "fig_1"
            / "fig_1.png"
        )
        if not path.is_file():
            self.skipTest("lithium_borate fig_1.png not present")
        image = cv2.imread(str(path))
        self.assertIsNotNone(image)
        panels = detect_plot_panels(image)
        self.assertGreaterEqual(len(panels), 2, msg=[(p.label, p.bbox, p.detection_method) for p in panels])
        self.assertEqual({p.label for p in panels[:2]}, {"A", "B"})

    def test_lithium_borate_padded_crop_keeps_x_ticks(self) -> None:
        from xrd_digitization.axis_sidecar import (
            extract_axis_sidecar_for_path,
            x_calibration_is_usable,
        )
        from xrd_digitization.split_figure_panels import split_figure_into_panels

        path = (
            ROOT
            / "pipeline_output"
            / "lithium_borate"
            / "figures"
            / "fig_1"
            / "fig_1.png"
        )
        if not path.is_file():
            self.skipTest("lithium_borate fig_1.png not present")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            src = out / "fig_1.png"
            src.write_bytes(path.read_bytes())
            result = split_figure_into_panels(src, figure_id="fig_1")
            self.assertGreaterEqual(len(result.panels), 2)
            for rec in result.panels:
                self.assertIsNotNone(rec.axes_bbox)
                self.assertIsNotNone(rec.panel_source_bbox)
                ax0, ay0, ax1, ay1 = rec.axes_bbox  # type: ignore[misc]
                sx0, sy0, sx1, sy1 = rec.panel_source_bbox  # type: ignore[misc]
                # Source crop must be strictly larger than axes frame (padding).
                self.assertLessEqual(sx0, ax0)
                self.assertLessEqual(sy0, ay0)
                self.assertGreaterEqual(sx1, ax1)
                self.assertGreaterEqual(sy1, ay1)
                self.assertGreater(sy1 - ay1, int(0.10 * (ay1 - ay0)))
                sidecar = extract_axis_sidecar_for_path(rec.image_path)
                usable, reasons = x_calibration_is_usable(sidecar.calibration)
                self.assertTrue(usable, msg=reasons)
                self.assertAlmostEqual(sidecar.calibration.x_min, 10.0, delta=2.0)
                self.assertAlmostEqual(sidecar.calibration.x_max, 70.0, delta=2.0)
                self.assertGreaterEqual(len(sidecar.calibration.tick_pairs), 2)


class SplitFigurePanelsTests(unittest.TestCase):
    def test_write_panel_crops_keeps_source(self) -> None:
        img = np.full((200, 500, 3), 255, dtype=np.uint8)
        _draw_axes_frame(img, 10, 10, 230, 180)
        _draw_axes_frame(img, 270, 10, 490, 180)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "fig_1.png"
            cv2.imwrite(str(src), img)
            panels = detect_plot_panels(img)
            result = write_panel_crops(
                src,
                panels,
                figure_id="fig_1",
                figure_caption="Figure 1. Full caption shared by all panels.",
                figure_body_text="See Figure 1A for glasses. Figure 1B shows ceramics.",
            )
            self.assertTrue(src.is_file())
            self.assertGreaterEqual(len(result.panels), 2)
            self.assertTrue((root / "fig_1_A.png").is_file())
            self.assertTrue((root / "fig_1_B.png").is_file())
            self.assertTrue((root / "fig_1.panels.json").is_file())
            payload = json.loads((root / "fig_1.panels.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["panel_count"], len(result.panels))
            for rec in result.panels:
                self.assertEqual(
                    rec.figure_caption,
                    "Figure 1. Full caption shared by all panels.",
                )

    def test_openai_panel_locator_mocked(self) -> None:
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        payload = {
            "panels": [
                {
                    "label": "A",
                    "bbox_norm": [0.0, 0.0, 0.48, 1.0],
                    "plot_type": "stacked_xrd",
                    "digitizable_after_split": True,
                },
                {
                    "label": "B",
                    "bbox_norm": [0.52, 0.0, 1.0, 1.0],
                    "plot_type": "stacked_xrd",
                    "digitizable_after_split": True,
                },
            ]
        }
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        http_post = MagicMock(return_value=response)
        panels = call_openai_panel_locator(img, api_key="sk-test", http_post=http_post)
        self.assertEqual(len(panels), 2)
        self.assertEqual(panels[0].label, "A")
        self.assertEqual(panels[1].label, "B")

    def test_split_force_vision_fallback(self) -> None:
        img = np.full((120, 240, 3), 255, dtype=np.uint8)
        payload = {
            "panels": [
                {"label": "A", "bbox_norm": [0.0, 0.0, 0.5, 1.0]},
                {"label": "B", "bbox_norm": [0.5, 0.0, 1.0, 1.0]},
            ]
        }
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        http_post = MagicMock(return_value=response)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "fig_9.png"
            cv2.imwrite(str(src), img)
            result = split_figure_into_panels(
                src,
                force_vision=True,
                http_post=http_post,
            )
            self.assertEqual(result.method, "openai_vision")
            self.assertEqual(len(result.panels), 2)
            self.assertTrue((Path(tmp) / "fig_9_A.png").is_file())


if __name__ == "__main__":
    unittest.main()
