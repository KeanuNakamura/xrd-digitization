"""Tests for OpenAI figure triage and scrape/digitize output rearrangement."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(LEGACY), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from figure_triage import (  # noqa: E402
    classify_xrd_figure_image,
    digitization_route,
    is_digitizable,
    is_multipanel_digitizable,
    is_stacked_digitizable,
    normalize_curve_layout,
    parse_triage_payload,
    parse_xrd_classify_payload,
    triage_figure_image,
    _normalize_api_key,
)
from scrape_and_digitize import (  # noqa: E402
    process_figure,
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
        self.assertEqual(digitization_route(result), "single")
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
        self.assertEqual(result.curve_layout, "multiple_overlapping")
        self.assertEqual(digitization_route(result), "unsupported")
        self.assertTrue(result.needs_clipdrop)

    def test_stacked_routes_to_stacked(self) -> None:
        result = parse_triage_payload(
            {
                "digitizable": True,
                "curve_count": 8,
                "curve_layout": "stacked",
                "shared_x_axis": True,
                "vertically_offset": True,
                "curves_cross": False,
                "needs_clipdrop": False,
                "curve_labels": [
                    {"text": "715 nm", "vertical_order": 0},
                    {"text": "641 nm", "vertical_order": 7},
                ],
                "reason": "stacked XRD traces",
            }
        )
        self.assertEqual(result.curve_layout, "multiple_stacked")
        self.assertFalse(is_digitizable(result))  # single-path still false
        self.assertTrue(is_stacked_digitizable(result))
        self.assertEqual(digitization_route(result), "stacked")
        self.assertEqual(len(result.curve_labels), 2)
        self.assertEqual(result.curve_labels[0].text, "715 nm")

    def test_stacked_crossing_unsupported(self) -> None:
        result = parse_triage_payload(
            {
                "digitizable": True,
                "curve_count": 4,
                "curve_layout": "multiple_stacked",
                "shared_x_axis": True,
                "curves_cross": True,
                "needs_clipdrop": False,
                "reason": "peaks cross neighbors",
            }
        )
        self.assertFalse(result.digitizable)
        self.assertEqual(digitization_route(result), "unsupported")

    def test_layout_aliases(self) -> None:
        self.assertEqual(normalize_curve_layout("stacked"), "multiple_stacked")
        self.assertEqual(normalize_curve_layout("overlay"), "multiple_overlapping")
        self.assertEqual(normalize_curve_layout("other"), "not_applicable")
        self.assertEqual(normalize_curve_layout("multiple_subplots"), "multiple_subplots")

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
        self.assertIn("overlapping_annotations", result.preprocessing_reasons)

    def test_multipanel_routes_to_multipanel(self) -> None:
        result = parse_triage_payload(
            {
                "digitizable": True,
                "curve_count": 6,
                "curve_layout": "multiple_subplots",
                "shared_x_axis": False,
                "vertically_offset": True,
                "curves_cross": False,
                "needs_clipdrop": True,
                "preprocessing_reasons": ["multiple_panels", "overlapping_annotations"],
                "reason": "two subplot panels with stacked XRD curves",
            }
        )
        self.assertEqual(result.curve_layout, "multiple_subplots")
        self.assertTrue(result.digitizable)
        self.assertTrue(is_multipanel_digitizable(result))
        self.assertEqual(digitization_route(result), "multipanel")
        self.assertIn("multiple_panels", result.preprocessing_reasons)
        self.assertIn("overlapping_annotations", result.preprocessing_reasons)

    def test_multipanel_digitizable_false_still_routes(self) -> None:
        """Legacy model rejects for annotations; still split panels."""
        result = parse_triage_payload(
            {
                "digitizable": False,
                "curve_count": 6,
                "curve_layout": "multiple_subplots",
                "shared_x_axis": False,
                "vertically_offset": True,
                "needs_clipdrop": True,
                "reason": "separate panels and overlapping annotations",
            }
        )
        self.assertFalse(result.digitizable)
        self.assertEqual(digitization_route(result), "multipanel")

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

    def test_parse_xrd_classify_payload(self) -> None:
        result = parse_xrd_classify_payload(
            {"is_xrd": True, "reason": "2θ powder pattern"}
        )
        self.assertTrue(result.is_xrd)
        self.assertEqual(result.reason, "2θ powder pattern")

    def test_mocked_http_xrd_classify(self) -> None:
        payload = {"is_xrd": False, "reason": "SEM micrograph"}
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        http_post = MagicMock(return_value=response)

        image = np.full((40, 60, 3), 255, dtype=np.uint8)
        result = classify_xrd_figure_image(
            image,
            api_key="test-key",
            http_post=http_post,
        )
        self.assertFalse(result.is_xrd)
        self.assertIn("SEM", result.reason)
        http_post.assert_called_once()
        _, kwargs = http_post.call_args
        messages = kwargs["json"]["messages"]
        user_text = messages[1]["content"][0]["text"]
        self.assertIn("is_xrd", user_text)


class ProcessFigureXrdFilterTests(unittest.TestCase):
    def test_non_xrd_deleted_by_default(self) -> None:
        payload = {"is_xrd": False, "reason": "FTIR spectrum"}
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        http_post = MagicMock(return_value=response)

        with tempfile.TemporaryDirectory() as tmp:
            figures = Path(tmp) / "figures"
            figures.mkdir()
            src = figures / "fig_sem.png"
            cv2.imwrite(str(src), np.full((20, 30, 3), 120, dtype=np.uint8))

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                entry = process_figure(
                    src,
                    figures_dir=figures,
                    model="test-model",
                    overwrite=True,
                    http_post=http_post,
                )

            self.assertEqual(entry["status"], "skipped_not_xrd")
            self.assertFalse(entry["xrd_classification"]["is_xrd"])
            self.assertFalse((figures / "fig_sem").exists())
            self.assertFalse(src.exists())
            http_post.assert_called_once()

    def test_all_figures_keeps_non_xrd_for_triage(self) -> None:
        classify_payload = {"is_xrd": False, "reason": "schematic"}
        triage_payload = {
            "digitizable": False,
            "curve_count": 0,
            "curve_layout": "not_applicable",
            "needs_clipdrop": False,
            "reason": "not a plot",
        }
        responses = [classify_payload, triage_payload]

        def http_post(url, headers=None, json=None, timeout=None):  # noqa: A002
            payload = responses.pop(0)
            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json.return_value = {
                "choices": [
                    {"message": {"content": __import__("json").dumps(payload)}}
                ]
            }
            return response

        with tempfile.TemporaryDirectory() as tmp:
            figures = Path(tmp) / "figures"
            figures.mkdir()
            src = figures / "fig_schema.png"
            cv2.imwrite(str(src), np.full((20, 30, 3), 80, dtype=np.uint8))

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                entry = process_figure(
                    src,
                    figures_dir=figures,
                    model="test-model",
                    overwrite=True,
                    http_post=http_post,
                    xrd_only=False,
                )

            self.assertEqual(entry["status"], "skipped_not_digitizable")
            self.assertFalse(entry["xrd_classification"]["is_xrd"])
            self.assertTrue((figures / "fig_schema" / "fig_schema.png").is_file())
            self.assertTrue(
                (figures / "fig_schema" / "fig_schema.xrd_classify.json").is_file()
            )


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
