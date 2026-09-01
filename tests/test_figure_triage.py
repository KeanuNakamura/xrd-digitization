"""Tests for OpenAI figure triage and scrape/digitize output rearrangement."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(LEGACY), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from figure_triage import (  # noqa: E402
    FigureTriageResult,
    _normalize_api_key,
    is_digitizable,
    parse_triage_payload,
    triage_figure_image,
)
from scrape_and_digitize import (  # noqa: E402
    rearrange_digitize_outputs,
    stage_figure_directory,
)


class FigureTriageParseTests(unittest.TestCase):
    def test_single_curve_digitizable(self) -> None:
        result = parse_triage_payload(
            {
                "digitizable": True,
                "curve_count": 1,
                "curve_layout": "single",
                "needs_clipdrop": False,
                "reason": "one clean curve",
            }
        )
        self.assertTrue(result.digitizable)
        self.assertTrue(is_digitizable(result))
        self.assertFalse(result.needs_clipdrop)

    def test_multi_curve_forced_not_digitizable(self) -> None:
        result = parse_triage_payload(
            {
                "digitizable": True,  # model inconsistent
                "curve_count": 3,
                "curve_layout": "overlay",
                "needs_clipdrop": True,
                "reason": "three overlays",
            }
        )
        self.assertFalse(result.digitizable)
        self.assertFalse(is_digitizable(result))
        self.assertEqual(result.curve_layout, "overlay")
        self.assertTrue(result.needs_clipdrop)

    def test_stacked_not_digitizable(self) -> None:
        result = parse_triage_payload(
            {
                "digitizable": True,
                "curve_count": 1,
                "curve_layout": "stacked",
                "needs_clipdrop": False,
                "reason": "stacked panels",
            }
        )
        self.assertFalse(is_digitizable(result))

    def test_needs_clipdrop_single_curve(self) -> None:
        result = parse_triage_payload(
            {
                "digitizable": True,
                "curve_count": 1,
                "curve_layout": "single",
                "needs_clipdrop": True,
                "reason": "Miller labels on peaks",
            }
        )
        self.assertTrue(is_digitizable(result))
        self.assertTrue(result.needs_clipdrop)

    def test_normalize_api_key_strips_curly_quotes(self) -> None:
        self.assertEqual(_normalize_api_key("\u2019sk-abc\u2019"), "sk-abc")
        self.assertEqual(_normalize_api_key("'sk-abc'"), "sk-abc")
        self.assertEqual(_normalize_api_key("  sk-abc  "), "sk-abc")
        self.assertIsNone(_normalize_api_key("   "))
        self.assertIsNone(_normalize_api_key(None))

    def test_mocked_http_triage(self) -> None:
        payload = {
            "digitizable": True,
            "curve_count": 1,
            "curve_layout": "single",
            "needs_clipdrop": False,
            "reason": "ok",
        }
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        http_post = MagicMock(return_value=response)

        image = np.full((40, 60, 3), 255, dtype=np.uint8)
        result = triage_figure_image(
            image,
            api_key="test-key",
            http_post=http_post,
        )
        self.assertTrue(is_digitizable(result))
        http_post.assert_called_once()
        _, kwargs = http_post.call_args
        self.assertIn("json", kwargs)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_mocked_http_triage_strips_curly_quoted_key(self) -> None:
        payload = {
            "digitizable": True,
            "curve_count": 1,
            "curve_layout": "single",
            "needs_clipdrop": False,
            "reason": "ok",
        }
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        http_post = MagicMock(return_value=response)

        image = np.full((40, 60, 3), 255, dtype=np.uint8)
        triage_figure_image(
            image,
            api_key="\u2019sk-real\u2019",
            http_post=http_post,
        )
        _, kwargs = http_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-real")


class RearrangeOutputsTests(unittest.TestCase):
    def test_stage_figure_directory_moves_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            figures = Path(tmp) / "figures"
            figures.mkdir()
            src = figures / "fig_6_2.png"
            src.write_bytes(b"\x89PNG\r\n\x1a\n")

            figure_id, figure_dir, staged = stage_figure_directory(src, figures)

            self.assertEqual(figure_id, "fig_6_2")
            self.assertEqual(figure_dir, figures / "fig_6_2")
            self.assertEqual(staged, figures / "fig_6_2" / "fig_6_2.png")
            self.assertTrue(staged.is_file())
            self.assertFalse(src.exists())

    def test_copies_csv_and_digitized_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work" / "fig_1"
            work.mkdir(parents=True)
            (work / "fig_1.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (work / "fig_1_digitized.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (work / "fig_1_clean.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            figure_dir = root / "figures" / "fig_1"
            figure_dir.mkdir(parents=True)
            (figure_dir / "fig_1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            placed = rearrange_digitize_outputs(
                work,
                figure_id="fig_1",
                figure_dir=figure_dir,
            )

            self.assertTrue((figure_dir / "fig_1.csv").is_file())
            self.assertTrue((figure_dir / "fig_1_digitized.png").is_file())
            self.assertTrue((figure_dir / "fig_1_clean.png").is_file())
            self.assertEqual(placed["csv"], str(figure_dir / "fig_1.csv"))
            self.assertIn("clean_png", placed)


if __name__ == "__main__":
    unittest.main()
