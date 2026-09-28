"""Integration-style acceptance checks for lithium_borate multipanel routing."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(LEGACY), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from figure_triage import digitization_route, parse_triage_payload  # noqa: E402
from scrape_and_digitize import process_figure  # noqa: E402
from xrd_digitization.split_figure_panels import split_figure_into_panels  # noqa: E402

LITHIUM_FIG = (
    ROOT / "pipeline_output" / "lithium_borate" / "figures" / "fig_1" / "fig_1.png"
)


@unittest.skipUnless(LITHIUM_FIG.is_file(), "lithium_borate fig_1.png not present")
class LithiumBorateMultipanelAcceptanceTests(unittest.TestCase):
    def test_existing_triage_no_longer_unsupported(self) -> None:
        triage_path = LITHIUM_FIG.with_name("fig_1.triage.json")
        payload = json.loads(triage_path.read_text(encoding="utf-8"))
        result = parse_triage_payload(payload)
        self.assertEqual(result.curve_layout, "multiple_subplots")
        self.assertEqual(digitization_route(result), "multipanel")
        self.assertNotEqual(digitization_route(result), "unsupported")

    def test_splits_into_a_and_b_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            src = out / "fig_1.png"
            src.write_bytes(LITHIUM_FIG.read_bytes())
            result = split_figure_into_panels(src, figure_id="fig_1")
            self.assertGreaterEqual(len(result.panels), 2)
            labels = {rec.panel.label for rec in result.panels}
            self.assertIn("A", labels)
            self.assertIn("B", labels)
            self.assertTrue((out / "fig_1.png").is_file())
            self.assertTrue((out / "fig_1_A.png").is_file())
            self.assertTrue((out / "fig_1_B.png").is_file())
            # Source bytes unchanged.
            self.assertEqual(src.read_bytes(), LITHIUM_FIG.read_bytes())
            for rec in result.panels:
                self.assertIsNotNone(rec.axes_bbox)
                self.assertIsNotNone(rec.panel_source_bbox)
                self.assertNotEqual(rec.axes_bbox, rec.panel_source_bbox)

    def test_process_figure_attempts_independent_panel_digitization(self) -> None:
        """Mock triage + digitize so acceptance does not need live APIs."""
        figure_triage = {
            "digitizable": True,
            "curve_count": 6,
            "curve_layout": "multiple_subplots",
            "shared_x_axis": False,
            "vertically_offset": True,
            "curves_cross": False,
            "needs_clipdrop": True,
            "curve_labels": [],
            "reason": "two subplot panels",
            "preprocessing_reasons": ["multiple_panels"],
        }
        panel_triage = {
            "digitizable": True,
            "curve_count": 3,
            "curve_layout": "multiple_stacked",
            "shared_x_axis": True,
            "vertically_offset": True,
            "curves_cross": False,
            "needs_clipdrop": False,
            "curve_labels": [
                {"text": "c1", "vertical_order": 0},
                {"text": "c2", "vertical_order": 1},
                {"text": "c3", "vertical_order": 2},
            ],
            "reason": "stacked XRD panel",
        }

        responses = [
            {"is_xrd": True, "reason": "powder XRD diffractogram"},
            figure_triage,
            panel_triage,
            {**panel_triage, "needs_clipdrop": True},
        ]

        def _fake_http_post(url, headers=None, json=None, timeout=None):
            payload = responses.pop(0) if responses else panel_triage
            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json.return_value = {
                "choices": [
                    {"message": {"content": __import__("json").dumps(payload)}}
                ]
            }
            return response

        def _fake_digitize(png_path, output_dir, **kwargs):
            figure_id = Path(png_path).stem
            work = Path(output_dir) / figure_id
            work.mkdir(parents=True, exist_ok=True)
            (work / f"{figure_id}.png").write_bytes(Path(png_path).read_bytes())
            (work / f"{figure_id}_stacked.csv").write_text(
                "curve_id,label,x,y\n0,a,10,1\n",
                encoding="utf-8",
            )
            (work / f"{figure_id}_stacked.json").write_text("{}", encoding="utf-8")
            (work / "digitized_stacked.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            return work

        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "lithium_borate"
            figures = paper / "figures"
            figures.mkdir(parents=True)
            src = figures / "fig_1.png"
            src.write_bytes(LITHIUM_FIG.read_bytes())
            (paper / "lithium_borate.figure_analysis.json").write_text(
                json.dumps(
                    [
                        {
                            "figure": 1,
                            "caption": "Figure 1. Shared full caption.",
                            "text": "Figure 1A glasses. Figure 1B ceramics.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch("scrape_and_digitize.digitize_one_figure", side_effect=_fake_digitize):
                with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                    entry = process_figure(
                        src,
                        figures_dir=figures,
                        model="test-model",
                        overwrite=True,
                        http_post=_fake_http_post,
                        paper_dir=paper,
                    )

            self.assertNotEqual(entry.get("status"), "skipped_unsupported_multiple_subplots")
            self.assertIn(entry.get("status"), {"digitized", "digitized_partial"})
            panels = entry.get("panels") or []
            self.assertGreaterEqual(len(panels), 2)
            labels = {p.get("panel_label") for p in panels}
            self.assertIn("A", labels)
            self.assertIn("B", labels)
            figure_dir = Path(entry["figure_dir"])
            self.assertTrue((figure_dir / "fig_1.png").is_file())
            self.assertTrue((figure_dir / "fig_1_A.png").is_file())
            self.assertTrue((figure_dir / "fig_1_B.png").is_file())
            for panel in panels:
                self.assertEqual(panel.get("layout"), "stacked")
                self.assertEqual(panel.get("status"), "digitized")
                self.assertEqual(
                    panel.get("triage", {}).get("curve_count"),
                    3,
                )


if __name__ == "__main__":
    unittest.main()
