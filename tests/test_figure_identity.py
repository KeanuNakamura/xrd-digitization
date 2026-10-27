"""Tests for figure labeling, Extended Data ids, and collision refinement."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
for path in (ROOT, LEGACY):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from pdf_parser import (
    Figure,
    FIGURE_KIND_EXTENDED,
    FIGURE_KIND_MAIN,
    expand_merged_caption_figures,
    figure_label_from_caption,
    is_valid_figure_label,
    parse_all_figure_identities,
    parse_figure_identity,
    primary_figure_number_from_stem,
    refine_figure_identities,
    reindex_tei_figure_ids,
    validate_extracted_figures,
)


def _figure(**kwargs) -> Figure:
    defaults = dict(
        figure_id="fig_x",
        label=None,
        normalized_label=None,
        caption="",
        graphic_targets=[],
        coords=None,
        graphic_coords=None,
        figure_type=None,
        xrd_score=0.0,
        is_likely_xrd=False,
        is_caption_xrd=False,
        image_paths=[],
    )
    defaults.update(kwargs)
    return Figure(**defaults)


def test_parse_main_and_extended_identities():
    assert parse_figure_identity("Fig. 1 XRD profile of CsPbBr3") == (
        parse_figure_identity("Figure 1. Le Bail fit")
    )
    main = parse_figure_identity("Fig. 1 XRD profile of CsPbBr3")
    assert main is not None
    assert main.kind == FIGURE_KIND_MAIN
    assert main.number == "1"
    assert main.figure_id == "fig_1"

    ed = parse_figure_identity(
        "Fig. 1 |Extended Data Fig. 3 |Extended Data Fig. 4 |Extended Data Fig. 5 |"
    )
    assert ed is not None
    assert ed.kind == FIGURE_KIND_EXTENDED
    assert ed.number == "3"
    assert ed.figure_id == "ed_fig_3"

    truncated = parse_figure_identity(
        "Data Fig. 1 |Extended Data Fig. 7 | SoS of RGB QD films."
    )
    assert truncated is not None
    assert truncated.kind == FIGURE_KIND_EXTENDED
    assert truncated.number == "1"


def test_rejects_buried_and_junk_labels():
    assert figure_label_from_caption(
        "3. 1 Structural properties ... shown in Fig. 1"
    ) is None
    assert not is_valid_figure_label(
        "1",
        "3. 1 Structural properties ... shown in Fig. 1",
    )
    assert parse_figure_identity("5 HalfFig. 3 | Fig. 3 | EL performance") is not None
    assert parse_figure_identity("5 HalfFig. 3 | Fig. 3 | EL performance").number == "3"
    assert parse_figure_identity("label 1345789") is None


def test_refine_collapses_phantom_fig_1_collision():
    phantom = _figure(
        figure_id="fig_1",
        label="1",
        normalized_label="1",
        caption="3. 1 Structural and morphological properties shown in Fig. 1",
        coords="3,10,10,20,10",
        graphic_coords=None,
    )
    real = _figure(
        figure_id="fig_1_2",
        label="1",
        normalized_label="1",
        caption="Fig. 1 XRD profile of CsPbBr3 nanocrystals",
        coords="4,10,10,200,200",
        graphic_coords="4,10,10,200,200",
        image_paths=["fig_1_2.png"],
    )
    notes = refine_figure_identities([phantom, real])
    assert real.figure_id == "fig_1"
    assert phantom.figure_id != "fig_1"
    assert phantom.figure_id != "fig_1_2"
    assert any("collapsed" in note or "renamed" in note or "demoted" in note for note in notes)


def test_refine_marks_extended_data_bundle():
    main = _figure(
        figure_id="fig_1",
        label="1",
        normalized_label="1",
        caption="Fig. 1 | Main article figure",
        graphic_coords="1,10,10,100,100",
        image_paths=["fig_1.png"],
    )
    bundle = _figure(
        figure_id="fig_1_2",
        label="1345789",
        normalized_label="1345789",
        caption=(
            "Fig. 1 |Extended Data Fig. 3 |Extended Data Fig. 4 |"
            "Extended Data Fig. 5 |"
        ),
        coords="12,10,10,50,20",
        graphic_coords=None,
        image_paths=["fig_1_2_page_12.png"],
    )
    refine_figure_identities([main, bundle])
    assert main.figure_id == "fig_1"
    assert bundle.figure_id == "ed_fig_3"


def test_reindex_prefers_graphic_for_canonical_id():
    tei = b"""<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <figure xml:id="fig_0" coords="3,10,10,40,12">
          <head><label>1</label></head>
          <figDesc>3. 1 Structural section text mentioning Fig. 1</figDesc>
        </figure>
        <figure xml:id="fig_1" coords="4,10,10,200,200">
          <head><label>1</label></head>
          <figDesc>Fig. 1 XRD profile of nanocrystals</figDesc>
          <graphic coords="4,10,10,200,200"/>
        </figure>
        <figure xml:id="fig_2" coords="5,10,10,50,20">
          <figDesc>Fig. 1 |Extended Data Fig. 2 |Extended Data Fig. 3 |</figDesc>
        </figure>
      </body></text>
    </TEI>
    """
    out = reindex_tei_figure_ids(tei).decode("utf-8")
    assert 'xml:id="fig_1"' in out
    assert "XRD profile" in out
    # Extended Data bundle must not steal fig_1.
    assert "ed_fig_" in out
    assert "fig_1_2" not in out or out.count('xml:id="fig_1"') == 1


def test_primary_stem_supports_extended_and_supp():
    assert primary_figure_number_from_stem("ed_fig_3") == 3
    assert primary_figure_number_from_stem("supp_fig_2_page_4") == 2
    assert primary_figure_number_from_stem("fig_1_2") == 1


def test_validate_flags_duplicate_extractions():
    figs = [
        _figure(
            figure_id="fig_2",
            caption="Fig. 2 a",
            graphic_coords="1,0,0,10,10",
            image_paths=["a.png"],
        ),
        _figure(
            figure_id="fig_2_2",
            caption="Fig. 2 b",
            graphic_coords="1,0,0,10,10",
            image_paths=["b.png"],
        ),
    ]
    warnings = validate_extracted_figures(figs)
    assert any("multiple extractions for fig_2" in w for w in warnings)


def test_parse_all_identities_from_merged_caption():
    identities = parse_all_figure_identities(
        "Fig. 1 | Fig. 2 | Extended Data Fig. 3 | Extended Data Fig. 5 |"
    )
    ids = [identity.figure_id for identity in identities]
    assert ids == ["fig_1", "fig_2", "ed_fig_3", "ed_fig_5"]


def test_expand_merged_caption_adds_sibling_stubs():
    figs = [
        _figure(
            figure_id="fig_1",
            caption="Fig. 1 | Fig. 2 | Extended Data Fig. 5 |",
            graphic_coords="1,0,0,100,100",
            image_paths=["fig_1.png"],
        )
    ]
    notes = expand_merged_caption_figures(figs)
    assert any("fig_2" in note for note in notes)
    assert any("ed_fig_5" in note for note in notes)
    ids = {figure.figure_id for figure in figs}
    assert ids >= {"fig_1", "fig_2", "ed_fig_5"}


def test_pdf_caption_anchors_reject_body_callouts(tmp_path):
    import pymupdf
    from parse_figures import find_pdf_figure_caption_anchors

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Fig. 16), in agreement with previous reports.")
    page.insert_text((72, 200), "Fig. 2 | Optical properties of fluoride-treated nanocrystals")
    page.insert_text((72, 400), "Figure 3. Schematic of the device stack used in this work.")
    pdf_path = tmp_path / "anchors.pdf"
    doc.save(pdf_path)
    doc.close()

    anchors = find_pdf_figure_caption_anchors(pdf_path)
    ids = {anchor.figure_id for anchor in anchors}
    assert "fig_16" not in ids
    assert "fig_2" in ids
    assert "fig_3" in ids
