"""Regression tests for figure dedup, stem parsing, and orphan recovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
SCRIPTS = ROOT / "scripts"
AUTO = ROOT / "autonomous_digitizer"
for path in (ROOT, LEGACY, SCRIPTS, AUTO):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from figure_analysis.schemas import FigureAnalysisResult, XrdCurve
from pdf_parser import (
    Author,
    DocumentMetadata,
    Figure,
    Paragraph,
    ParsedDocument,
    build_figure_analysis_dataset,
    primary_figure_number_from_stem,
)
import scrape_and_digitize as sad


def _make_png(path: Path, color: tuple[int, int, int] = (200, 200, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return path


def _doc(figures: list[Figure], paragraphs: list[Paragraph] | None = None) -> ParsedDocument:
    return ParsedDocument(
        source_pdf="paper.pdf",
        source_sha256="abc",
        metadata=DocumentMetadata(
            title="Test",
            doi=None,
            abstract=None,
            journal=None,
            publication_date=None,
            authors=[Author(given_name="A", surname="Author")],
            keywords=[],
        ),
        sections=[],
        paragraphs=paragraphs or [],
        figures=figures,
        tables=[],
        references=[],
        counts={},
    )


def test_primary_figure_number_from_multi_component_stems():
    assert primary_figure_number_from_stem("fig_1") == 1
    assert primary_figure_number_from_stem("fig_1_2") == 1
    assert primary_figure_number_from_stem("fig_7_2") == 7
    assert primary_figure_number_from_stem("figure_3") == 3
    assert primary_figure_number_from_stem("figure_10_page_2") == 10
    assert primary_figure_number_from_stem("fig_1_2.png") == 1
    # Must not treat trailing suffix as the figure number.
    assert primary_figure_number_from_stem("fig_1_2") != 2


def test_dataset_prefers_image_backed_duplicate_label(tmp_path: Path):
    bad = Figure(
        figure_id="fig_1",
        label="1",
        normalized_label="1",
        caption="Section heading text mentioning XRD patterns",
        graphic_targets=[],
        coords=None,
        graphic_coords=None,
        figure_type=None,
        xrd_score=1.0,
        is_likely_xrd=False,
        is_caption_xrd=False,
        image_paths=[],
    )
    good_png = _make_png(tmp_path / "fig_1_2.png", (255, 0, 0))
    good = Figure(
        figure_id="fig_1_2",
        label="1",
        normalized_label="1",
        caption="Fig. 1 XRD profile of CsPbBr3 nanocrystals",
        graphic_targets=[],
        coords="1,0,0,10,10",
        graphic_coords="1,0,0,10,10",
        figure_type=None,
        xrd_score=5.0,
        is_likely_xrd=True,
        is_caption_xrd=True,
        image_paths=[str(good_png)],
    )
    dataset = build_figure_analysis_dataset(_doc([bad, good]))
    assert len(dataset) == 1
    assert dataset[0]["figure"] == 1
    assert dataset[0]["figure_id"] == "fig_1_2"
    assert Path(dataset[0]["figure_path"]).name == "fig_1_2.png"
    assert "XRD profile" in dataset[0]["caption"]


def test_dataset_keeps_distinct_images_with_same_label(tmp_path: Path):
    png_a = _make_png(tmp_path / "fig_2.png", (1, 2, 3))
    png_b = _make_png(tmp_path / "fig_2_2.png", (4, 5, 6))
    figs = [
        Figure(
            figure_id="fig_2",
            label="2",
            normalized_label="2",
            caption="Fig. 2 panel set A",
            graphic_targets=[],
            coords="1,0,0,1,1",
            graphic_coords="1,0,0,1,1",
            figure_type=None,
            xrd_score=1.0,
            is_likely_xrd=False,
            is_caption_xrd=False,
            image_paths=[str(png_a)],
        ),
        Figure(
            figure_id="fig_2_2",
            label="2",
            normalized_label="2",
            caption="Fig. 2 alternate crop",
            graphic_targets=[],
            coords="2,0,0,1,1",
            graphic_coords="2,0,0,1,1",
            figure_type=None,
            xrd_score=1.0,
            is_likely_xrd=False,
            is_caption_xrd=False,
            image_paths=[str(png_b)],
        ),
    ]
    dataset = build_figure_analysis_dataset(_doc(figs))
    paths = sorted(Path(e["figure_path"]).name for e in dataset)
    assert paths == ["fig_2.png", "fig_2_2.png"]
    assert all(e["figure"] == 2 for e in dataset)


def test_dataset_includes_image_with_missing_caption(tmp_path: Path):
    png = _make_png(tmp_path / "fig_4.png")
    fig = Figure(
        figure_id="fig_4",
        label="4",
        normalized_label="4",
        caption="",
        graphic_targets=[],
        coords="1,0,0,1,1",
        graphic_coords="1,0,0,1,1",
        figure_type=None,
        xrd_score=0.0,
        is_likely_xrd=False,
        is_caption_xrd=False,
        image_paths=[str(png)],
    )
    dataset = build_figure_analysis_dataset(_doc([fig]))
    assert len(dataset) == 1
    assert dataset[0]["figure_path"]
    assert dataset[0]["caption"] == ""


def test_dataset_includes_unlabeled_image_via_stem(tmp_path: Path):
    png = _make_png(tmp_path / "fig_7_2.png")
    fig = Figure(
        figure_id="fig_7_2",
        label=None,
        normalized_label=None,
        caption="conductivity plot",
        graphic_targets=[],
        coords="1,0,0,1,1",
        graphic_coords=None,
        figure_type=None,
        xrd_score=0.0,
        is_likely_xrd=False,
        is_caption_xrd=False,
        image_paths=[str(png)],
    )
    dataset = build_figure_analysis_dataset(_doc([fig]))
    assert len(dataset) == 1
    assert dataset[0]["figure"] == 7
    assert dataset[0]["figure_id"] == "fig_7_2"


def test_find_figure_image_prefers_exact_over_suffix(tmp_path: Path):
    paper = tmp_path / "paper"
    figures = paper / "figures"
    exact = _make_png(figures / "fig_1.png", (1, 0, 0))
    _make_png(figures / "fig_1_2.png", (0, 1, 0))
    found = sad.find_figure_image_for_number(paper, 1)
    assert found == exact.resolve()


def test_find_figure_image_suffix_only_when_unique(tmp_path: Path):
    paper = tmp_path / "paper"
    figures = paper / "figures"
    only = _make_png(figures / "fig_1_2.png")
    found = sad.find_figure_image_for_number(paper, 1)
    assert found == only.resolve()

    _make_png(figures / "fig_1_3.png")
    # Ambiguous suffixed matches → do not guess.
    assert sad.find_figure_image_for_number(paper, 1) is None


def test_recover_orphan_images(tmp_path: Path):
    paper = tmp_path / "paper"
    figures = paper / "figures"
    claimed = _make_png(figures / "fig_3.png")
    orphan = _make_png(figures / "fig_1_2.png")
    entries = [
        {
            "figure": 3,
            "figure_id": "fig_3",
            "figure_path": str(claimed),
            "caption": "other",
            "text": "other",
        }
    ]
    out = sad.recover_orphan_figure_images(paper, entries)
    assert len(out) == 2
    recovered = [e for e in out if e.get("association_status") == "unresolved"]
    assert len(recovered) == 1
    assert Path(recovered[0]["figure_path"]).resolve() == orphan.resolve()
    assert recovered[0]["figure"] == 1
    assert recovered[0]["figure_id"] == "fig_1_2"


def test_enrich_analyzes_orphan_xrd_and_dedupes_rerun(tmp_path: Path):
    paper = tmp_path / "capping"
    figures = paper / "figures"
    xrd_png = _make_png(figures / "fig_1_2.png", (10, 20, 30))
    other = _make_png(figures / "fig_7_2.png", (40, 50, 60))
    fa = paper / "capping.figure_analysis.json"
    # Simulate the old bug: only a text-only / empty-path figure 1 entry.
    fa.write_text(
        json.dumps(
            {
                "title": "Capping ligands",
                "doi": "10.x",
                "authors": ["A"],
                "figures": [
                    {
                        "figure": 1,
                        "figure_path": "",
                        "caption": "wrong section text",
                        "text": "wrong section text",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_analyze(image_path, **kwargs):
        name = Path(image_path).name
        calls.append(name)
        # Cache key must be paper-scoped (different papers can share filenames).
        assert "capping" in str(kwargs.get("cache_key") or "")
        if name == "fig_1_2.png":
            return FigureAnalysisResult(
                is_xrd=True,
                sample="CsPbBr3",
                curves=[
                    XrdCurve(
                        peak_positions=[15.2, 30.7],
                        phases=["orthorhombic CsPbBr3"],
                    )
                ],
            )
        return FigureAnalysisResult(is_xrd=False)

    sad.analyze_figure_image = fake_analyze
    sad.enrich_figure_analysis_json(paper, client=object())

    data = json.loads(fa.read_text(encoding="utf-8"))
    assert "fig_1_2.png" in calls
    assert "fig_7_2.png" in calls
    assert len(data["figures"]) == 1
    assert data["figures"][0]["is_xrd"] is True
    assert Path(data["figures"][0]["figure_path"]).name == "fig_1_2.png"
    assert data["figures"][0]["analysis"]["curves"][0]["peak_positions"] == [
        15.2,
        30.7,
    ]
    assert not xrd_png.exists()
    assert not other.exists()
    assert (paper / "xrd_figures" / "fig_1_2.png").is_file()
    assert (paper / "non_xrd_figures" / "fig_7_2.png").is_file()
    assert "xrd_figures" in data["figures"][0]["figure_path"]

    # Rerun should not invent duplicates (only the kept XRD entry remains;
    # orphan recovery finds nothing new).
    calls.clear()
    sad.enrich_figure_analysis_json(paper, client=object())
    data2 = json.loads(fa.read_text(encoding="utf-8"))
    assert len(data2["figures"]) == 1
    assert Path(data2["figures"][0]["figure_path"]).name == "fig_1_2.png"


def test_enrich_cache_keys_differ_across_papers(tmp_path: Path):
    keys: list[str] = []

    def fake_analyze(image_path, **kwargs):
        keys.append(str(kwargs.get("cache_key") or ""))
        return FigureAnalysisResult(is_xrd=False)

    sad.analyze_figure_image = fake_analyze

    for name in ("paper_a", "paper_b"):
        paper = tmp_path / name
        figures = paper / "figures"
        _make_png(figures / "fig_1.png")
        fa = paper / f"{name}.figure_analysis.json"
        fa.write_text(
            json.dumps(
                {
                    "title": name,
                    "doi": None,
                    "authors": [],
                    "figures": [
                        {
                            "figure": 1,
                            "figure_id": "fig_1",
                            "figure_path": str(figures / "fig_1.png"),
                            "caption": "x",
                            "text": "x",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        sad.enrich_figure_analysis_json(paper, client=object())

    assert len(keys) == 2
    assert keys[0] != keys[1]
    assert "paper_a" in keys[0]
    assert "paper_b" in keys[1]
