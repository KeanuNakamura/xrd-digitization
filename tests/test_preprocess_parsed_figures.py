"""Tests for batch preprocessing of parsed PDF figure outputs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
for path in (str(LEGACY), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from preprocess_parsed_figures import (  # noqa: E402
    default_output_root,
    discover_paper_dirs,
    find_parsed_json,
    preprocess_paper,
    resolve_pdf_path,
)


class DiscoveryTests(unittest.TestCase):
    def test_discover_single_paper_and_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "carbonStacking"
            extra = paper / "extra"
            extra.mkdir(parents=True)
            parsed = extra / "carbonStacking.parsed.json"
            parsed.write_text(json.dumps({"figures": []}), encoding="utf-8")
            (root / "figures_without_text").mkdir()
            (root / "noise").mkdir()

            self.assertEqual(discover_paper_dirs(paper), [paper.resolve()])
            self.assertEqual(discover_paper_dirs(root), [paper.resolve()])
            self.assertEqual(discover_paper_dirs(parsed), [paper.resolve()])
            self.assertEqual(
                default_output_root(root),
                (root / "figures_without_text").resolve(),
            )
            self.assertEqual(
                default_output_root(paper),
                (root / "figures_without_text").resolve(),
            )

    def test_resolve_pdf_prefers_source_then_pdf_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "demo"
            paper.mkdir()
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            pdf = pdf_dir / "demo.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            missing = root / "missing.pdf"
            doc = {"source_pdf": str(missing)}
            resolved = resolve_pdf_path(doc, paper, pdf_dir=pdf_dir)
            self.assertEqual(resolved, pdf)


class PreprocessPaperTests(unittest.TestCase):
    def test_preprocess_paper_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "demo"
            extra = paper / "extra"
            extra.mkdir(parents=True)
            pdf = root / "demo.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            (extra / "demo.parsed.json").write_text(
                json.dumps(
                    {
                        "source_pdf": str(pdf),
                        "figures": [
                            {
                                "figure_id": "fig_1",
                                "coords": "1,10,10,100,100",
                                "graphic_coords": "1,10,10,100,100",
                                "is_likely_xrd": True,
                            },
                            {
                                "figure_id": "fig_2",
                                "coords": None,
                                "graphic_coords": None,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out" / "demo"

            fake_result = {
                "output_paths": {
                    "fig_1": {
                        "original": str(out / "fig_1_original.png"),
                        "combined_text_mask": str(out / "fig_1_combined_text_mask.png"),
                    }
                },
                "pages": [
                    {
                        "page_number": 1,
                        "figure_rect": [10, 10, 110, 110],
                        "text_free_render": {"created": False},
                    }
                ],
            }

            with mock.patch(
                "preprocess_parsed_figures.select_figure_crop_coords",
                side_effect=lambda coords, graphic, pdf_path=None: graphic or coords,
            ), mock.patch(
                "preprocess_parsed_figures.extract_figure",
                return_value=fake_result,
            ) as extract:
                manifest = preprocess_paper(paper, out, dpi=150, mode="all")

            extract.assert_called_once()
            self.assertEqual(manifest["counts"]["ok"], 1)
            self.assertEqual(manifest["counts"]["skipped_no_coords"], 1)
            self.assertTrue((out / "manifest.json").is_file())
            self.assertEqual(find_parsed_json(paper).name, "demo.parsed.json")


if __name__ == "__main__":
    unittest.main()
