"""Tests for PDF figure structure inspection and text-region masks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
for path in (str(LEGACY), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pdf_figure_structure import (  # noqa: E402
    DIAGNOSTIC_LABELS,
    EXPERIMENTAL_RECONSTRUCTION_NOTE,
    apply_text_mask_soft_penalty,
    build_pdf_text_mask,
    check_diagnostic_labels,
    classify_figure_structure,
    classify_outlined_or_flattened_text,
    dpi_to_zoom,
    extract_figure,
    inspect_figure_region,
    render_without_pdf_text,
    resolve_dpi,
    score_curve_like_component,
)
from raster_text_detection import (  # noqa: E402
    detect_raster_text_regions,
    split_figure_caption_region,
    validate_text_geometry,
)
from parse_figures import (  # noqa: E402
    crop_figure_from_grobid_coords,
    resolve_figure_page_clips,
)
from xrd_digitization.text_regions import (  # noqa: E402
    apply_text_mask_soft_penalty as soft_penalty,
)


def _make_raster_png_bytes(
    width: int = 80,
    height: int = 60,
    fill=(0.85, 0.85, 0.9),
    stroke=(0.1, 0.1, 0.6),
) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(
        page.rect,
        color=stroke,
        fill=fill,
        width=2,
    )
    page.draw_line(
        pymupdf.Point(5, height * 0.7),
        pymupdf.Point(width - 5, height * 0.3),
        color=(0.8, 0.1, 0.1),
        width=2,
    )
    pixmap = page.get_pixmap(dpi=72)
    data = pixmap.tobytes("png")
    doc.close()
    return data


def _save_temp_pdf(doc: pymupdf.Document, directory: Path, name: str) -> Path:
    path = directory / name
    doc.save(path)
    doc.close()
    return path


def _zero_padding_kwargs() -> dict:
    return {
        "padding": 0.0,
        "padding_left": 0.0,
        "padding_top": 0.0,
        "padding_right": 0.0,
        "padding_bottom": 0.0,
    }


class ResolveDpiTests(unittest.TestCase):
    def test_zoom_converts_to_dpi(self) -> None:
        self.assertEqual(resolve_dpi(zoom=3.0), 216)
        self.assertAlmostEqual(dpi_to_zoom(216), 3.0)

    def test_rejects_both_dpi_and_zoom(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dpi(dpi=150, zoom=2.0)


class ClassificationTests(unittest.TestCase):
    def test_heuristic_labels(self) -> None:
        self.assertEqual(
            classify_figure_structure(
                has_pdf_text=True,
                has_vector_drawings=True,
                has_raster_images=False,
                vector_drawing_count=3,
            ),
            "mostly_vector",
        )
        self.assertEqual(
            classify_figure_structure(
                has_pdf_text=True,
                has_vector_drawings=False,
                has_raster_images=True,
                vector_drawing_count=0,
            ),
            "raster_with_pdf_text_overlay",
        )
        self.assertEqual(
            classify_figure_structure(
                has_pdf_text=False,
                has_vector_drawings=False,
                has_raster_images=True,
                vector_drawing_count=0,
            ),
            "likely_fully_rasterized",
        )
        self.assertEqual(
            classify_figure_structure(
                has_pdf_text=True,
                has_vector_drawings=True,
                has_raster_images=True,
                vector_drawing_count=2,
            ),
            "mixed",
        )
        self.assertEqual(
            classify_figure_structure(
                has_pdf_text=False,
                has_vector_drawings=False,
                has_raster_images=False,
                vector_drawing_count=0,
            ),
            "unknown",
        )
        self.assertEqual(
            classify_figure_structure(
                has_pdf_text=False,
                has_vector_drawings=True,
                has_raster_images=False,
                vector_drawing_count=5,
                has_outlined_or_flattened_text=True,
            ),
            "vector_with_outlined_text",
        )


class DiagnosticLabelTests(unittest.TestCase):
    def test_check_diagnostic_labels_reports_presence(self) -> None:
        words = [{"text": "FO"}, {"text": "curve"}, {"text": "BS"}]
        result = check_diagnostic_labels(words, DIAGNOSTIC_LABELS)
        self.assertTrue(result["label_present_in_pdf_words"]["FO"])
        self.assertTrue(result["label_present_in_pdf_words"]["BS"])
        self.assertFalse(result["label_present_in_pdf_words"]["MI"])
        self.assertIn("MI", result["missing_labels"])

    def test_outlined_requires_visible_evidence(self) -> None:
        labels = check_diagnostic_labels([], DIAGNOSTIC_LABELS)
        # Missing FO/BS/... alone is not enough when PDF text exists.
        with_pdf = classify_outlined_or_flattened_text(
            label_diagnostics=labels,
            has_pdf_text=True,
            has_vector_drawings=True,
            has_raster_images=False,
            visible_labels_from_raster=None,
        )
        self.assertFalse(with_pdf["has_outlined_or_flattened_text"])

        no_pdf = classify_outlined_or_flattened_text(
            label_diagnostics=labels,
            has_pdf_text=False,
            has_vector_drawings=True,
            has_raster_images=False,
        )
        self.assertTrue(no_pdf["has_outlined_or_flattened_text"])

        raster_evidence = classify_outlined_or_flattened_text(
            label_diagnostics=labels,
            has_pdf_text=True,
            has_vector_drawings=True,
            has_raster_images=False,
            visible_labels_from_raster=["FO", "BS"],
        )
        self.assertTrue(raster_evidence["has_outlined_or_flattened_text"])


class SoftPenaltyTests(unittest.TestCase):
    def test_curve_continues_through_text_region(self) -> None:
        curve = np.zeros((40, 120), dtype=np.uint8)
        curve[18:22, 5:115] = 255  # long thin horizontal curve
        text = np.zeros_like(curve)
        text[10:30, 50:70] = 255  # label blob overlapping the curve

        kept = soft_penalty(
            curve,
            text,
            plot_left=0,
            plot_top=0,
            plot_right=120,
            plot_bottom=40,
        )
        # Soft penalty must not hard-delete the curve under the label.
        self.assertGreater(int(kept[18:22, 50:70].sum()), 0)
        self.assertGreater(_coverage(kept), 50)

    def test_compact_text_component_can_be_removed(self) -> None:
        mask = np.zeros((40, 120), dtype=np.uint8)
        mask[8:20, 40:52] = 255  # compact glyph-like blob
        text = mask.copy()
        score = score_curve_like_component(
            width=12,
            height=12,
            area=144,
            horizontal_coverage=12,
            plot_width=120,
            plot_height=40,
            text_overlap_fraction=1.0,
            centroid_xy=(46.0, 14.0),
            plot_rect=(0, 0, 120, 40),
        )
        self.assertTrue(score["likely_text"] or score["curve_score"] < 0.15)
        kept = apply_text_mask_soft_penalty(
            mask,
            text,
            plot_left=0,
            plot_top=0,
            plot_right=120,
            plot_bottom=40,
        )
        self.assertEqual(int(kept.sum()), 0)


def _coverage(mask: np.ndarray) -> int:
    return int(np.sum(mask.sum(axis=0) > 0))


class PdfFigureStructureTests(unittest.TestCase):
    def test_vector_plot_with_pdf_text_defaults_to_masks_not_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=300, height=220)
            page.draw_rect(pymupdf.Rect(40, 30, 260, 170), color=(0, 0, 0), width=1)
            page.draw_line(
                pymupdf.Point(50, 150),
                pymupdf.Point(250, 50),
                color=(0.8, 0.0, 0.0),
                width=1.5,
            )
            page.insert_text(pymupdf.Point(120, 200), "2-theta", fontsize=10)
            page.insert_text(pymupdf.Point(8, 100), "I", fontsize=10)
            pdf_path = _save_temp_pdf(doc, tmp_path, "vector_text.pdf")

            out = tmp_path / "out"
            result = extract_figure(
                pdf_path,
                "1,0,0,300,220",
                out,
                "figure_vector",
                mode="all",
                dpi=144,
                detect_raster_text=False,
                **_zero_padding_kwargs(),
            )
            page_info = result["pages"][0]
            self.assertEqual(page_info["inspection"]["classification"], "mostly_vector")
            self.assertTrue(page_info["inspection"]["has_pdf_text"])
            self.assertTrue(page_info["inspection"]["has_vector_drawings"])
            self.assertFalse(page_info["text_free_render"]["created"])
            self.assertTrue(page_info["text_free_render"]["experimental"])
            self.assertIn(
                EXPERIMENTAL_RECONSTRUCTION_NOTE,
                " ".join(page_info["text_free_render"]["limitations"]),
            )

            original = out / "figure_vector_original.png"
            text_free = out / "figure_vector_without_pdf_text.png"
            mask = out / "figure_vector_pdf_text_mask.png"
            raster_mask = out / "figure_vector_raster_text_mask.png"
            combined = out / "figure_vector_combined_text_mask.png"
            structure = out / "figure_vector_pdf_structure.json"
            self.assertTrue(original.exists())
            self.assertFalse(text_free.exists())
            self.assertTrue(mask.exists())
            self.assertTrue(raster_mask.exists())
            self.assertTrue(combined.exists())
            self.assertTrue(structure.exists())

            payload = json.loads(structure.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(payload["text_spans"]), 1)
            self.assertIn("pdf_words", payload)
            self.assertIn("diagnostics", payload)
            self.assertIn("vector_drawing_count", payload["diagnostics"])
            self.assertIn("diagnostic_labels", payload["diagnostics"])
            self.assertFalse(payload["text_free_render"]["created"])
            self.assertFalse((out / "figure_vector.png").exists())

    def test_fully_rasterized_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=200, height=160)
            page.insert_image(
                pymupdf.Rect(10, 10, 190, 150),
                stream=_make_raster_png_bytes(),
            )
            pdf_path = _save_temp_pdf(doc, tmp_path, "raster_only.pdf")

            out = tmp_path / "out"
            result = extract_figure(
                pdf_path,
                "1,0,0,200,160",
                out,
                "figure_raster",
                mode="all",
                dpi=120,
                detect_raster_text=False,
                **_zero_padding_kwargs(),
            )
            page_info = result["pages"][0]
            self.assertEqual(
                page_info["inspection"]["classification"],
                "likely_fully_rasterized",
            )
            self.assertFalse(page_info["inspection"]["has_pdf_text"])
            self.assertTrue(page_info["inspection"]["has_outlined_or_flattened_text"])
            self.assertFalse(page_info["text_free_render"]["created"])
            self.assertTrue((out / "figure_raster_original.png").exists())
            self.assertFalse((out / "figure_raster_without_pdf_text.png").exists())
            self.assertTrue((out / "figure_raster_pdf_structure.json").exists())
            self.assertTrue((out / "figure_raster_combined_text_mask.png").exists())

    def test_embedded_image_with_pdf_text_overlay_no_default_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=220, height=180)
            page.insert_image(
                pymupdf.Rect(20, 20, 200, 140),
                stream=_make_raster_png_bytes(),
            )
            page.insert_text(pymupdf.Point(70, 165), "overlay label", fontsize=11)
            pdf_path = _save_temp_pdf(doc, tmp_path, "raster_overlay.pdf")

            out = tmp_path / "out"
            result = extract_figure(
                pdf_path,
                "1,0,0,220,180",
                out,
                "figure_overlay",
                mode="all",
                dpi=120,
                detect_raster_text=False,
                **_zero_padding_kwargs(),
            )
            page_info = result["pages"][0]
            self.assertEqual(
                page_info["inspection"]["classification"],
                "raster_with_pdf_text_overlay",
            )
            self.assertFalse(page_info["text_free_render"]["created"])
            self.assertFalse((out / "figure_overlay_without_pdf_text.png").exists())
            self.assertTrue((out / "figure_overlay_combined_text_mask.png").exists())

    def test_mixed_vectors_images_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=260, height=200)
            page.draw_rect(pymupdf.Rect(20, 20, 150, 150), color=(0, 0, 0), width=1)
            page.draw_line(
                pymupdf.Point(30, 130),
                pymupdf.Point(140, 40),
                color=(0, 0.5, 0),
                width=1.2,
            )
            page.insert_image(
                pymupdf.Rect(160, 30, 240, 100),
                stream=_make_raster_png_bytes(40, 30),
            )
            page.insert_text(pymupdf.Point(30, 180), "mixed figure", fontsize=10)
            pdf_path = _save_temp_pdf(doc, tmp_path, "mixed.pdf")

            out = tmp_path / "out"
            result = extract_figure(
                pdf_path,
                "1,0,0,260,200",
                out,
                "figure_mixed",
                mode="all",
                dpi=120,
                detect_raster_text=False,
                **_zero_padding_kwargs(),
            )
            inspection = result["pages"][0]["inspection"]
            self.assertEqual(inspection["classification"], "mixed")
            self.assertTrue(inspection["has_pdf_text"])
            self.assertTrue(inspection["has_vector_drawings"])
            self.assertTrue(inspection["has_raster_images"])
            self.assertFalse(result["pages"][0]["text_free_render"]["created"])

    def test_experimental_reconstruction_only_when_forced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=200, height=200)
            page.draw_line(
                pymupdf.Point(20, 180),
                pymupdf.Point(180, 20),
                color=(1, 0, 0),
                width=3,
            )
            page.insert_text(pymupdf.Point(70, 110), "PEAK", fontsize=18)
            pdf_path = _save_temp_pdf(doc, tmp_path, "overlap.pdf")

            document = pymupdf.open(pdf_path)
            try:
                src_page = document[0]
                clip = pymupdf.Rect(0, 0, 200, 200)
                inspection = inspect_figure_region(src_page, clip)
                self.assertTrue(inspection["has_pdf_text"])
                self.assertTrue(inspection["has_vector_drawings"])

                skipped, skipped_meta = render_without_pdf_text(
                    src_page,
                    clip,
                    dpi=144,
                    inspection=inspection,
                    force=False,
                )
                self.assertIsNone(skipped)
                self.assertFalse(skipped_meta["created"])

                text_free, meta = render_without_pdf_text(
                    src_page,
                    clip,
                    dpi=144,
                    inspection=inspection,
                    force=True,
                )
                self.assertIsNotNone(text_free)
                assert text_free is not None
                self.assertTrue(meta["created"])
                self.assertTrue(meta["experimental"])
                self.assertEqual(meta["method"], "experimental_vector_reconstruction")

                samples = text_free.samples
                redish = 0
                for i in range(0, len(samples), 3):
                    r, g, b = samples[i], samples[i + 1], samples[i + 2]
                    if r > 180 and g < 80 and b < 80:
                        redish += 1
                self.assertGreater(redish, 50)
            finally:
                document.close()

    def test_text_span_intersection_filtering(self) -> None:
        doc = pymupdf.open()
        page = doc.new_page(width=300, height=300)
        page.insert_text(pymupdf.Point(20, 40), "inside", fontsize=12)
        page.insert_text(pymupdf.Point(220, 260), "outside", fontsize=12)
        page.draw_rect(pymupdf.Rect(10, 10, 120, 80), color=(0, 0, 0), width=1)

        figure_rect = pymupdf.Rect(0, 0, 150, 100)
        inspection = inspect_figure_region(page, figure_rect)
        texts = [span["text"].strip() for span in inspection["text_spans"]]
        self.assertIn("inside", texts)
        self.assertNotIn("outside", texts)
        self.assertIn("pdf_words", inspection)
        self.assertIn("diagnostic_labels", inspection)
        doc.close()

    def test_mask_alignment_at_different_zoom_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=200, height=120)
            page.draw_line(
                pymupdf.Point(10, 60),
                pymupdf.Point(190, 60),
                color=(0, 0, 0),
                width=1,
            )
            page.insert_text(pymupdf.Point(40, 40), "Tick", fontsize=14)
            pdf_path = _save_temp_pdf(doc, tmp_path, "mask_zoom.pdf")

            for zoom in (2.0, 3.0):
                out = tmp_path / f"out_z{zoom}"
                result = extract_figure(
                    pdf_path,
                    "1,20,10,160,90",
                    out,
                    "figure_mask",
                    mode="inspect",
                    zoom=zoom,
                    detect_raster_text=False,
                    **_zero_padding_kwargs(),
                )
                page_info = result["pages"][0]
                width = page_info["original_size"]["width"]
                height = page_info["original_size"]["height"]

                original = pymupdf.Pixmap(str(out / "figure_mask_original.png"))
                mask = pymupdf.Pixmap(str(out / "figure_mask_pdf_text_mask.png"))
                combined = pymupdf.Pixmap(
                    str(out / "figure_mask_combined_text_mask.png")
                )
                self.assertEqual(original.width, width)
                self.assertEqual(original.height, height)
                self.assertEqual(mask.width, original.width)
                self.assertEqual(mask.height, original.height)
                self.assertEqual(combined.width, original.width)

                white = 0
                samples = mask.samples
                for i in range(0, len(samples), mask.n):
                    if samples[i] > 200:
                        white += 1
                self.assertGreater(white, 10)

                payload = json.loads(
                    (out / "figure_mask_pdf_structure.json").read_text(encoding="utf-8")
                )
                span = payload["text_spans"][0]
                clip = payload["figure_rect"]
                dpi = payload["dpi"]
                scale = dpi / 72.0
                expected_x0 = (span["bbox"][0] - clip[0]) * scale
                self.assertGreaterEqual(expected_x0, 0)
                self.assertLess(expected_x0, width)

    def test_empty_figure_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=200, height=200)
            page.insert_text(pymupdf.Point(10, 20), "far away", fontsize=10)
            page.draw_line(
                pymupdf.Point(10, 30),
                pymupdf.Point(50, 30),
                color=(0, 0, 0),
                width=1,
            )
            pdf_path = _save_temp_pdf(doc, tmp_path, "empty_region.pdf")

            out = tmp_path / "out"
            result = extract_figure(
                pdf_path,
                "1,120,120,60,60",
                out,
                "figure_empty",
                mode="all",
                dpi=100,
                detect_raster_text=False,
                **_zero_padding_kwargs(),
            )
            page_info = result["pages"][0]
            self.assertEqual(page_info["inspection"]["classification"], "unknown")
            self.assertFalse(page_info["text_free_render"]["created"])
            self.assertTrue((out / "figure_empty_original.png").exists())
            self.assertTrue((out / "figure_empty_pdf_structure.json").exists())
            self.assertFalse((out / "figure_empty_without_pdf_text.png").exists())

    def test_original_crop_workflow_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            page = doc.new_page(width=100, height=100)
            page.draw_rect(pymupdf.Rect(10, 10, 90, 90), color=(0, 0, 1), width=1)
            pdf_path = _save_temp_pdf(doc, tmp_path, "legacy_crop.pdf")

            legacy_out = tmp_path / "legacy"
            paths = crop_figure_from_grobid_coords(
                pdf_path,
                "1,10,10,80,80",
                legacy_out,
                "figure_9",
                dpi=72,
                padding=0,
                padding_left=0,
                padding_top=0,
                padding_right=0,
                padding_bottom=0,
            )
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].name, "figure_9.png")
            self.assertTrue(paths[0].exists())

            structured_out = tmp_path / "structured"
            extract_figure(
                pdf_path,
                "1,10,10,80,80",
                structured_out,
                "figure_9",
                mode="original",
                dpi=72,
                **_zero_padding_kwargs(),
            )
            self.assertTrue((structured_out / "figure_9_original.png").exists())
            self.assertFalse((structured_out / "figure_9.png").exists())

    def test_structure_json_is_serializable(self) -> None:
        doc = pymupdf.open()
        page = doc.new_page(width=120, height=120)
        page.insert_text(pymupdf.Point(20, 40), "A", fontsize=12)
        page.draw_rect(pymupdf.Rect(10, 50, 100, 100), color=(0, 0, 0), width=1)
        inspection = inspect_figure_region(page, pymupdf.Rect(0, 0, 120, 120))
        encoded = json.dumps(inspection)
        self.assertIn("text_spans", encoded)
        self.assertIn("pdf_words", encoded)
        doc.close()

    def test_build_pdf_text_mask_dimensions(self) -> None:
        spans = [
            {
                "text": "x",
                "bbox": [10.0, 10.0, 30.0, 22.0],
                "font": "Helvetica",
                "size": 10.0,
            }
        ]
        clip = pymupdf.Rect(0, 0, 100, 80)
        dpi = 144
        width = int(round(clip.width * dpi / 72.0))
        height = int(round(clip.height * dpi / 72.0))
        mask = build_pdf_text_mask(
            spans,
            clip,
            width=width,
            height=height,
            dpi=dpi,
            padding_px=1,
        )
        self.assertEqual(mask.width, width)
        self.assertEqual(mask.height, height)

    def test_resolve_figure_page_clips_uses_one_based_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            doc = pymupdf.open()
            doc.new_page(width=100, height=100)
            doc.new_page(width=100, height=100)
            pdf_path = _save_temp_pdf(doc, tmp_path, "pages.pdf")

            document = pymupdf.open(pdf_path)
            try:
                clips = resolve_figure_page_clips(
                    document,
                    "2,10,10,40,40",
                    padding=0,
                    padding_left=0,
                    padding_top=0,
                    padding_right=0,
                    padding_bottom=0,
                )
                self.assertEqual(len(clips), 1)
                page_number, page, clip = clips[0]
                self.assertEqual(page_number, 2)
                self.assertEqual(page.number, 1)
                self.assertEqual(list(clip), [10.0, 10.0, 50.0, 50.0])
            finally:
                document.close()

    def test_scherrer_crops_include_x_axis_tick_strip(self) -> None:
        """Rasterized axis strips just above captions must stay in the crop."""
        pdf = ROOT / "pdf_files" / "sample_pdfs" / "Scherrer_equation.pdf"
        parsed = (
            ROOT
            / "grobid_output"
            / "sample_pdfs"
            / "Scherrer_equation"
            / "extra"
            / "Scherrer_equation.parsed.json"
        )
        if not pdf.exists() or not parsed.exists():
            self.skipTest("sample Scherrer_equation assets not available")

        from parse_figures import (
            CAPTION_CROP_CLEARANCE,
            parse_grobid_coords,
            resolve_figure_page_clips,
            select_figure_crop_coords,
        )

        doc_json = json.loads(parsed.read_text(encoding="utf-8"))
        # Bottom embedded strips end ~6pt above the caption; the old 12pt
        # clearance dropped them and clipped tick values (20–50).
        expected_bottoms = {
            "fig_2": 318.0,
            "fig_6_2": 611.0,
        }
        document = pymupdf.open(pdf)
        try:
            for figure_id, min_y1 in expected_bottoms.items():
                fig = next(
                    f for f in doc_json["figures"] if f["figure_id"] == figure_id
                )
                coords = select_figure_crop_coords(
                    fig.get("coords"), fig.get("graphic_coords"), pdf_path=pdf
                )
                self.assertIsNotNone(coords)
                crop_y1 = max(
                    y + h for _page, _x, y, _w, h in parse_grobid_coords(coords)
                )
                self.assertGreaterEqual(crop_y1, min_y1)

                caption_y0 = min(
                    y for _page, _x, y, _w, _h in parse_grobid_coords(fig["coords"])
                )
                clips = resolve_figure_page_clips(
                    document,
                    coords,
                    caption_coords=fig["coords"],
                )
                self.assertEqual(len(clips), 1)
                _page_number, _page, clip = clips[0]
                self.assertGreaterEqual(clip.y1, min_y1)
                self.assertLessEqual(
                    clip.y1, caption_y0 - CAPTION_CROP_CLEARANCE + 1e-6
                )
        finally:
            document.close()


class RasterTextDetectionTests(unittest.TestCase):
    def test_rejects_plot_sized_short_word_box(self) -> None:
        reason = validate_text_geometry(
            [10, 10, 700, 1200],
            image_width=852,
            image_height=1546,
            text="BS",
            panel_box=(0, 0, 852, 1360),
        )
        self.assertIsNotNone(reason)

    def test_accepts_tight_short_label(self) -> None:
        reason = validate_text_geometry(
            [690, 470, 730, 495],
            image_width=852,
            image_height=1546,
            text="BS",
            panel_box=(0, 0, 852, 1360),
        )
        self.assertIsNone(reason)

    def test_caption_split_keeps_ticks_in_plot(self) -> None:
        # Synthetic figure: plot body, tick band, then caption text.
        image = np.full((1000, 600, 3), 255, dtype=np.uint8)
        image[50:820, 80:560] = 230  # plot interior
        image[860:890, 100:500] = 40  # tick / axis-label ink
        image[940:980, 80:520] = 30  # caption ink
        plot_bbox, caption_bbox = split_figure_caption_region(image)
        self.assertGreaterEqual(plot_bbox[3], 860)
        self.assertIsNotNone(caption_bbox)
        assert caption_bbox is not None
        self.assertGreaterEqual(caption_bbox[1], int(1000 * 0.96))

    def test_detect_does_not_emit_huge_mask(self) -> None:
        # White canvas with a few dark glyph-like blobs and a large dark frame.
        image = np.full((400, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (280, 320), (0, 0, 0), 2)
        cv2.putText(
            image,
            "BS",
            (220, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "10",
            (40, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        result = detect_raster_text_regions(image, use_mser_proposals=True)
        self.assertLessEqual(result.coverage, 0.15)
        for det in result.accepted:
            x0, y0, x1, y1 = det["bbox"]
            self.assertLess((x1 - x0) * (y1 - y0) / (400 * 300), 0.08)

    def test_nms_removes_duplicate_proposals(self) -> None:
        from raster_text_detection import _nms_xywh_proposals

        proposals = [
            (10, 10, 20, 20),
            (11, 11, 20, 20),
            (12, 12, 19, 19),
            (100, 100, 15, 15),
        ]
        kept, removed = _nms_xywh_proposals(proposals, iou_threshold=0.5)
        self.assertGreaterEqual(removed, 2)
        self.assertLessEqual(len(kept), 2)

    def test_rejected_debug_is_capped(self) -> None:
        from raster_text_detection import _select_rejected_for_debug

        rejected = [
            {
                "bbox": [float(i), float(i), float(i + 5), float(i + 5)],
                "detection_confidence": float(i % 40),
                "reject_reason": "low_confidence",
                "text": None,
                "source": "mser_component",
            }
            for i in range(200)
        ]
        selected = _select_rejected_for_debug(rejected, max_count=50)
        self.assertLessEqual(len(selected), 50)

    def test_in_plot_groups_accepted_without_ocr(self) -> None:
        """Geometry groups inside the plot are maskable without OCR text."""
        image = np.full((500, 700, 3), 255, dtype=np.uint8)
        # Fake plot frame.
        cv2.rectangle(image, (80, 40), (660, 420), (0, 0, 0), 2)
        # Vertical peak-like glyph stack inside the plot.
        xs = 200
        for i, ch in enumerate(["(", "2", "1", "1", ")"]):
            cv2.putText(
                image,
                ch,
                (xs, 120 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        result = detect_raster_text_regions(image, use_tesseract=False)
        removable = [
            d
            for d in result.accepted
            if str(d.get("role") or "").startswith("removable_")
        ]
        self.assertGreaterEqual(len(removable), 1)
        self.assertIsNotNone(result.removable_glyph_mask)
        self.assertGreater(float((result.removable_glyph_mask > 0).sum()), 0)

    def test_carbonstacking_fig1_masks_in_plot_labels(self) -> None:
        """Sample validation: in-plot labels masked; axis text preserved."""
        pdf = ROOT / "pdf_files" / "sample_pdfs" / "carbonStacking.pdf"
        parsed = (
            ROOT
            / "grobid_output"
            / "sample_pdfs"
            / "carbonStacking"
            / "extra"
            / "carbonStacking.parsed.json"
        )
        if not pdf.exists() or not parsed.exists():
            self.skipTest("sample carbonStacking assets not available")

        import json

        from parse_figures import resolve_figure_page_clips, select_figure_crop_coords

        doc_json = json.loads(parsed.read_text(encoding="utf-8"))
        fig = next(f for f in doc_json["figures"] if f["figure_id"] == "fig_1")
        coords = select_figure_crop_coords(
            fig.get("coords"), fig.get("graphic_coords"), pdf_path=pdf
        )
        document = pymupdf.open(pdf)
        try:
            clips = resolve_figure_page_clips(
                document,
                coords,
                padding=0,
                padding_left=0,
                padding_top=0,
                padding_right=0,
                padding_bottom=0,
            )
            _page_number, page, clip = clips[0]
            pix = page.get_pixmap(clip=clip, dpi=300, alpha=False, annots=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        finally:
            document.close()

        result = detect_raster_text_regions(image_bgr)
        self.assertFalse(result.failed)
        self.assertLessEqual(result.coverage, 0.15)
        removable = [
            d
            for d in result.accepted
            if str(d.get("role") or "").startswith("removable_")
        ]
        preserved = [
            d
            for d in result.accepted
            if str(d.get("role") or "").startswith("preserved_")
        ]
        self.assertGreaterEqual(len(removable), 4)
        self.assertGreaterEqual(len(preserved), 1)
        # Glyph mask should be non-empty but far smaller than a full plot wipe.
        self.assertGreater(float((result.removable_glyph_mask > 0).mean()), 0.0005)
        self.assertLess(float((result.removable_glyph_mask > 0).mean()), 0.08)

    def test_scherrer_fig62_masks_vertical_peak_labels(self) -> None:
        """Validation targets for this sample only — not used by the detector."""
        pdf = ROOT / "pdf_files" / "sample_pdfs" / "Scherrer_equation.pdf"
        parsed = (
            ROOT
            / "grobid_output"
            / "sample_pdfs"
            / "Scherrer_equation"
            / "extra"
            / "Scherrer_equation.parsed.json"
        )
        if not pdf.exists() or not parsed.exists():
            self.skipTest("sample Scherrer_equation assets not available")

        import json

        from parse_figures import resolve_figure_page_clips, select_figure_crop_coords

        doc_json = json.loads(parsed.read_text(encoding="utf-8"))
        fig = next(f for f in doc_json["figures"] if f["figure_id"] == "fig_6_2")
        coords = select_figure_crop_coords(
            fig.get("coords"), fig.get("graphic_coords"), pdf_path=pdf
        )
        self.assertIsNotNone(coords)
        document = pymupdf.open(pdf)
        try:
            clips = resolve_figure_page_clips(
                document,
                coords,
                padding=0,
                padding_left=0,
                padding_top=0,
                padding_right=0,
                padding_bottom=0,
            )
            _page_number, page, clip = clips[0]
            pix = page.get_pixmap(clip=clip, dpi=300, alpha=False, annots=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        finally:
            document.close()

        expected_peaks = [
            "(200)",
            "(111)",
            "(201)",
            "(002)",
            "(102)",
            "(210)",
            "(211)",
            "(112)",
            "(300)",
            "(202)",
            "(301)",
            "(212)",
            "(310)",
            "(311)",
            "(113)",
            "(203)",
            "(222)",
            "(312)",
            "(320)",
            "(213)",
        ]
        result = detect_raster_text_regions(
            image_bgr,
            expected_labels=expected_peaks,
        )
        self.assertFalse(result.failed, msg=result.failure_reason)
        diag = result.diagnostics
        self.assertGreaterEqual(int(diag.get("vertical_groups") or 0), 18)
        self.assertGreaterEqual(int(diag.get("in_plot_text_groups") or 0), 18)
        self.assertGreaterEqual(
            len(result.by_role.get("removable_peak_annotation") or []),
            18,
        )
        self.assertGreaterEqual(int(diag.get("preserved_axis_labels") or 0), 1)
        self.assertLessEqual(float(diag.get("mask_coverage") or 0.0), 0.08)
        self.assertLessEqual(
            float(diag.get("axis_band_removable_overlap") or 0.0), 0.05
        )
        # OCR transcription of peaks is optional diagnostics only. Geometry
        # recall (vertical groups above) is the acceptance criterion.
        self.assertIsInstance(diag.get("missed_expected_labels"), list)
        # Removable and preserved masks should not substantially overlap.
        rem = result.removable_region_mask > 0
        pres = result.preserved_axis_mask > 0
        if rem.any() and pres.any():
            self.assertLess(float(np.logical_and(rem, pres).mean()), 0.02)
        # Axis title should be preserved, not removable.
        preserved_roles = {
            str(d.get("role"))
            for d in result.accepted
            if str(d.get("role") or "").startswith("preserved_")
        }
        self.assertTrue(
            "preserved_axis_title" in preserved_roles
            or "preserved_axis_tick" in preserved_roles
        )

if __name__ == "__main__":
    unittest.main()
