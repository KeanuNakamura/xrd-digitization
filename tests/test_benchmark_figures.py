"""Tests for --benchmark OpenAI XRD classification bucketing."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
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

from scrape_and_digitize import (  # noqa: E402
    BENCHMARK_NOT_XRD_DIR,
    BENCHMARK_XRD_DIR,
    benchmark_figure_basename,
    copy_figure_to_benchmark_bucket,
    process_pdf_benchmark,
    reset_benchmark_dirs,
)


@dataclass
class _FakeFigure:
    figure_id: str | None
    label: str | None
    caption: str
    xrd_score: float = 0.0
    is_likely_xrd: bool = False
    is_caption_xrd: bool = False
    image_paths: list[str] = field(default_factory=list)


@dataclass
class _FakeDocument:
    figures: list[_FakeFigure]


class BenchmarkBucketTests(unittest.TestCase):
    def test_basename_is_unique_across_pdfs(self) -> None:
        path = Path("fig_1.png")
        self.assertEqual(
            benchmark_figure_basename("paper_a", path),
            "paper_a__fig_1.png",
        )

    def test_copy_into_global_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xrd_dir, not_xrd_dir = reset_benchmark_dirs(root)
            src = root / "source.png"
            cv2.imwrite(str(src), np.zeros((20, 30, 3), dtype=np.uint8))

            dest_xrd = copy_figure_to_benchmark_bucket(
                src,
                pdf_stem="demo",
                is_xrd=True,
                xrd_dir=xrd_dir,
                not_xrd_dir=not_xrd_dir,
            )
            dest_not = copy_figure_to_benchmark_bucket(
                src,
                pdf_stem="demo",
                is_xrd=False,
                xrd_dir=xrd_dir,
                not_xrd_dir=not_xrd_dir,
            )

            self.assertTrue(dest_xrd.is_file())
            self.assertTrue(dest_not.is_file())
            self.assertEqual(dest_xrd.parent.name, BENCHMARK_XRD_DIR)
            self.assertEqual(dest_not.parent.name, BENCHMARK_NOT_XRD_DIR)
            self.assertEqual(dest_xrd.name, "demo__source.png")

    def test_process_pdf_benchmark_uses_openai_not_captions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_figures = root / "paper" / "figures"
            paper_figures.mkdir(parents=True)
            # Caption keywords would mark the first as XRD and second as not —
            # OpenAI mock returns the opposite to prove captions are ignored.
            misleading_xrd_caption = paper_figures / "fig_sem.png"
            misleading_other_caption = paper_figures / "fig_xrd.png"
            cv2.imwrite(
                str(misleading_xrd_caption),
                np.full((40, 50, 3), 40, dtype=np.uint8),
            )
            cv2.imwrite(
                str(misleading_other_caption),
                np.full((40, 50, 3), 200, dtype=np.uint8),
            )

            document = _FakeDocument(
                figures=[
                    _FakeFigure(
                        figure_id="fig_sem",
                        label="Figure 1",
                        caption="XRD pattern of the sample",
                        xrd_score=3.0,
                        is_likely_xrd=True,
                        is_caption_xrd=True,
                        image_paths=[str(misleading_xrd_caption)],
                    ),
                    _FakeFigure(
                        figure_id="fig_xrd",
                        label="Figure 2",
                        caption="SEM micrograph",
                        xrd_score=0.0,
                        is_likely_xrd=False,
                        is_caption_xrd=False,
                        image_paths=[str(misleading_other_caption)],
                    ),
                    _FakeFigure(
                        figure_id="fig_missing",
                        label="Figure 3",
                        caption="",
                        image_paths=[],
                    ),
                ]
            )
            pdf_path = root / "demo_paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            xrd_dir, not_xrd_dir = reset_benchmark_dirs(root / "out")

            # OpenAI: first image not XRD, second is XRD (opposite of captions).
            classify_queue = [
                {"is_xrd": False, "reason": "SEM micrograph"},
                {"is_xrd": True, "reason": "2θ diffractogram"},
            ]

            def http_post(url, headers=None, json=None, timeout=None):  # noqa: A002
                payload = classify_queue.pop(0)
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": __import__("json").dumps(payload)
                            }
                        }
                    ]
                }
                return response

            with patch(
                "scrape_and_digitize.parse_pdf",
                return_value=document,
            ) as parse_mock:
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    summary = process_pdf_benchmark(
                        pdf_path,
                        root / "out",
                        grobid_url="http://localhost:8070",
                        figure_dpi=72,
                        overwrite=True,
                        xrd_dir=xrd_dir,
                        not_xrd_dir=not_xrd_dir,
                        http_post=http_post,
                    )

            parse_mock.assert_called_once()
            kwargs = parse_mock.call_args.kwargs
            self.assertFalse(kwargs["xrd_figures_only"])
            self.assertEqual(summary["classifier"], "openai")
            self.assertEqual(summary["copied_xrd"], 1)
            self.assertEqual(summary["copied_not_xrd"], 1)
            self.assertEqual(summary["not_extracted"], 1)
            # Buckets follow OpenAI, not caption keywords.
            self.assertTrue((not_xrd_dir / "demo_paper__fig_sem.png").is_file())
            self.assertTrue((xrd_dir / "demo_paper__fig_xrd.png").is_file())
            self.assertFalse((xrd_dir / "demo_paper__fig_sem.png").exists())


if __name__ == "__main__":
    unittest.main()
