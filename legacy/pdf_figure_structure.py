"""
Inspect PDF figure regions and produce text-region metadata for digitization.

Canonical figure image
----------------------
The authoritative rendering is always ``page.get_pixmap(clip=...)``. That
composites text, vectors, and images exactly as the PDF viewer shows them.

Text handling strategy
----------------------
Visible labels are often *not* extractable PDF text objects: publishers
frequently convert glyphs to vector outlines or flatten text into rasters.
Therefore this module:

1. Saves the normal high-resolution pixmap as the canonical figure image.
2. Builds a mask of genuine PDF text spans (honestly incomplete).
3. Runs a raster text detector for labels not present as PDF text.
4. Combines both into a text-region mask used as *metadata* for curve
   extraction (soft penalties), never as an erase/inpaint target.

Vector reconstruction (``without_pdf_text.png``)
------------------------------------------------
Replaying ``page.get_drawings()`` cannot reliably separate outlined label
glyphs from scientific curves: both are generic drawing paths. Known
fidelity issues in the experimental reconstructor include:

- incomplete graphics-state replay (transforms, blend modes, soft masks)
- stroke / fill opacity and color-space mismatches
- filled glyph paths often becoming stroked outlines
- duplicated or fragmented path commits
- stroke-width scaling that does not match the original CTM
- outlined text and plot curves treated identically as drawing items

Do **not** try to "fix" this by darkening the reconstruction. The
experimental path is disabled by default and only runs when PDF text is
proven stored separately from the data graphics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import pymupdf

from parse_figures import resolve_figure_page_clips
from preprocess_text_removal import (
    DEFAULT_REMOVAL_METHOD,
    RemovalMethod,
    preprocess_removable_text,
)
from raster_text_detection import (
    MAX_RASTER_MASK_COVERAGE,
    detect_raster_text_regions as _detect_raster_text_regions_impl,
    pdf_span_boxes_to_pixels,
    validate_mask_for_combine,
)

LOGGER = logging.getLogger(__name__)

ExtractMode = Literal["original", "inspect", "all", "vectors_only"]

POINTS_PER_INCH = 72.0
DEFAULT_TEXT_MASK_PADDING_PX = 1
MINIMAL_VECTOR_COUNT = 1

# Common sample labels used to detect outlined / flattened text.
DIAGNOSTIC_LABELS: tuple[str, ...] = ("FO", "BS", "MI", "EB", "PI", "NL", "HE")

RASTER_TEXT_LIMITATION = (
    "Visible text appears to be flattened into an embedded raster image."
)
NO_RECONSTRUCTIBLE_CONTENT = (
    "No reconstructible vector drawings or embedded images were found "
    "in the figure region."
)
UNSUPPORTED_DRAWING_OPS_LIMITATION = (
    "Some drawing operations were unsupported and omitted from reconstruction."
)
OUTLINED_TEXT_LIMITATION = (
    "Visible labels are not present as extractable PDF text objects; they "
    "are likely vector outlines or flattened into the raster. Vector "
    "reconstruction cannot remove them without also harming curves."
)
EXPERIMENTAL_RECONSTRUCTION_NOTE = (
    "without_pdf_text.png is experimental. Outlined glyphs and plot curves "
    "are both drawing paths; reconstruction does not prove text-only removal."
)
PDF_TEXT_MASK_COVERAGE_NOTE = (
    "The PDF-text mask covers extractable text spans only. It does not claim "
    "to cover all visible text (outlined glyphs or rasterized labels)."
)
CANONICAL_IMAGE_NOTE = (
    "The canonical figure image is the normal page.get_pixmap(clip=...) "
    "rendering. Text masks are metadata for curve extraction and must not "
    "be used to erase or inpaint the canonical image."
)

VECTOR_RECONSTRUCTION_KNOWN_ISSUES: tuple[str, ...] = (
    "Graphics state (CTM / transforms) is not fully preserved when replaying "
    "get_drawings() items onto a translated page.",
    "Stroke and fill opacity may default incorrectly when missing on a path.",
    "Color values may not match the original color space / conversion.",
    "Filled glyph outlines can be finished as stroked paths.",
    "Each drawing dict is finished/committed independently, which can "
    "duplicate or fragment strokes relative to the original content stream.",
    "Stroke width is taken from the drawing dict without re-applying the "
    "original transformation matrix scale.",
    "Outlined text glyphs and scientific curves are indistinguishable as "
    "generic vector paths, so text cannot be removed selectively.",
)


def resolve_dpi(*, dpi: int | None = None, zoom: float | None = None) -> int:
    """Resolve render DPI. ``zoom`` maps as ``dpi = round(zoom * 72)``."""
    if dpi is not None and zoom is not None:
        raise ValueError("Pass only one of dpi or zoom, not both.")
    if zoom is not None:
        if zoom <= 0:
            raise ValueError(f"zoom must be positive, got {zoom!r}")
        return max(1, int(round(zoom * POINTS_PER_INCH)))
    if dpi is None:
        return 300
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi!r}")
    return int(dpi)


def dpi_to_zoom(dpi: int) -> float:
    return dpi / POINTS_PER_INCH


def _rect_to_list(rect: pymupdf.Rect) -> list[float]:
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


def _bbox_intersects(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    """Return True if rectangles overlap, including zero-area strokes."""
    a_use = pymupdf.Rect(
        a.x0,
        a.y0,
        max(a.x1, a.x0 + 1e-3),
        max(a.y1, a.y0 + 1e-3),
    )
    b_use = pymupdf.Rect(
        b.x0,
        b.y0,
        max(b.x1, b.x0 + 1e-3),
        max(b.y1, b.y0 + 1e-3),
    )
    return a_use.intersects(b_use)


def _drawing_intersects_clip(drawing: dict[str, Any], clip: pymupdf.Rect) -> bool:
    """Return True if a ``get_drawings()`` item intersects ``clip``."""
    drawing_rect = drawing.get("rect")
    if drawing_rect is not None:
        drawing_rect = pymupdf.Rect(drawing_rect)
        if _bbox_intersects(drawing_rect, clip):
            return True

    for item in drawing.get("items") or []:
        for part in item[1:]:
            if isinstance(part, pymupdf.Point):
                if (
                    clip.x0 - 1e-3 <= part.x <= clip.x1 + 1e-3
                    and clip.y0 - 1e-3 <= part.y <= clip.y1 + 1e-3
                ):
                    return True
            elif isinstance(part, pymupdf.Rect) and _bbox_intersects(part, clip):
                return True
            elif isinstance(part, pymupdf.Quad) and _bbox_intersects(part.rect, clip):
                return True
    return False


def _normalize_color(value: Any) -> list[float] | int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value) if isinstance(value, int) or float(value).is_integer() else float(value)
    if isinstance(value, (list, tuple)):
        return [float(component) for component in value]
    return None


def extract_pdf_words(
    page: pymupdf.Page,
    figure_rect: pymupdf.Rect | list[float] | tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Return ``page.get_text('words')`` entries intersecting ``figure_rect``."""
    clip = pymupdf.Rect(figure_rect)
    words: list[dict[str, Any]] = []
    for word in page.get_text("words") or []:
        # word tuple: x0, y0, x1, y1, text, block, line, word
        if len(word) < 5:
            continue
        x0, y0, x1, y1, text = word[:5]
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if not _bbox_intersects(rect, clip):
            continue
        words.append(
            {
                "text": str(text),
                "bbox": _rect_to_list(rect),
                "block_no": int(word[5]) if len(word) > 5 else None,
                "line_no": int(word[6]) if len(word) > 6 else None,
                "word_no": int(word[7]) if len(word) > 7 else None,
            }
        )
    return words


def check_diagnostic_labels(
    words: Sequence[dict[str, Any]],
    labels: Sequence[str] = DIAGNOSTIC_LABELS,
) -> dict[str, Any]:
    """
    Report whether diagnostic labels appear in extractable PDF words.

    Matching is case-insensitive and also checks whether a label is a token
    inside a longer word (e.g. ``FO`` in ``FO-1``).
    """
    normalized_words = [str(w.get("text", "")).strip() for w in words]
    upper_words = [w.upper() for w in normalized_words if w]
    joined = " ".join(upper_words)

    present: dict[str, bool] = {}
    missing: list[str] = []
    for label in labels:
        key = label.upper()
        found = any(
            word == key or key in word.split("-") or key in word.split("_")
            for word in upper_words
        )
        if not found:
            # Fallback: whole-word-ish presence in joined text.
            found = f" {key} " in f" {joined} "
        present[label] = found
        if not found:
            missing.append(label)

    return {
        "labels_checked": list(labels),
        "label_present_in_pdf_words": present,
        "missing_labels": missing,
        "any_label_missing": bool(missing),
        "all_labels_missing": bool(labels) and len(missing) == len(labels),
    }


def classify_outlined_or_flattened_text(
    *,
    label_diagnostics: dict[str, Any],
    has_pdf_text: bool,
    has_vector_drawings: bool,
    has_raster_images: bool,
    visible_labels_from_raster: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Classify outlined/flattened text when visible labels are absent from PDF text.

    ``visible_labels_from_raster`` should list diagnostic labels observed by the
    raster detector / OCR. Missing PDF-word labels alone do not imply outlined
    text (the figure may simply not use those tags).
    """
    missing = list(label_diagnostics.get("missing_labels") or [])
    present = label_diagnostics.get("label_present_in_pdf_words") or {}
    present_upper = {str(k).upper(): bool(v) for k, v in present.items()}
    raster_visible = [
        str(label).strip().upper()
        for label in (visible_labels_from_raster or [])
        if str(label).strip()
    ]

    raster_missing_from_pdf = [
        label for label in raster_visible if not present_upper.get(label, False)
    ]

    if raster_missing_from_pdf:
        return {
            "has_outlined_or_flattened_text": True,
            "text_encoding_class": "outlined_or_flattened_text",
            "reason": (
                "Raster text detection found labels "
                f"{raster_missing_from_pdf} that are absent from "
                "page.get_text('words'); they are likely vector outlines "
                "or flattened into the raster."
            ),
            "missing_labels": missing,
            "raster_labels_missing_from_pdf": raster_missing_from_pdf,
        }

    # No extractable PDF text while vectors (or rasters) carry the figure:
    # visible labels cannot be PDF text objects.
    if not has_pdf_text and (has_vector_drawings or has_raster_images):
        return {
            "has_outlined_or_flattened_text": True,
            "text_encoding_class": "outlined_or_flattened_text",
            "reason": (
                "No extractable PDF text spans were found in the figure "
                "region; any visible labels are outlined or flattened."
            ),
            "missing_labels": missing,
            "raster_labels_missing_from_pdf": raster_missing_from_pdf,
        }

    return {
        "has_outlined_or_flattened_text": False,
        "text_encoding_class": None,
        "reason": None,
        "missing_labels": missing,
        "raster_labels_missing_from_pdf": raster_missing_from_pdf,
    }


def text_separable_from_graphics(inspection: dict[str, Any]) -> bool:
    """
    Return True only when reconstruction could plausibly omit text objects
    without needing to delete drawing paths that may be outlined glyphs.
    """
    if inspection.get("has_outlined_or_flattened_text"):
        return False
    if not inspection.get("has_pdf_text"):
        return False
    if inspection.get("classification") == "likely_fully_rasterized":
        return False
    # Require that checked diagnostic labels, when any were found missing,
    # are not dominating. Separable only if no outlined classification.
    return True


def classify_figure_structure(
    *,
    has_pdf_text: bool,
    has_vector_drawings: bool,
    has_raster_images: bool,
    vector_drawing_count: int,
    has_outlined_or_flattened_text: bool = False,
) -> str:
    """
    Heuristic figure classification. Not guaranteed to match visual content.
    """
    if has_outlined_or_flattened_text and has_vector_drawings and not has_raster_images:
        return "vector_with_outlined_text"

    if has_outlined_or_flattened_text and has_raster_images and not has_vector_drawings:
        return "likely_fully_rasterized"

    if not has_pdf_text and not has_vector_drawings and not has_raster_images:
        return "unknown"

    if has_raster_images and has_pdf_text and not has_vector_drawings:
        return "raster_with_pdf_text_overlay"

    if (
        has_raster_images
        and not has_pdf_text
        and vector_drawing_count < MINIMAL_VECTOR_COUNT
    ):
        return "likely_fully_rasterized"

    if has_vector_drawings and has_raster_images:
        return "mixed"

    if has_vector_drawings and not has_raster_images:
        return "mostly_vector"

    if has_raster_images and has_pdf_text:
        return "raster_with_pdf_text_overlay"

    if has_raster_images:
        return "likely_fully_rasterized"

    if has_pdf_text and not has_vector_drawings and not has_raster_images:
        return "unknown"

    return "unknown"


def inspect_figure_region(
    page: pymupdf.Page,
    figure_rect: pymupdf.Rect | list[float] | tuple[float, float, float, float],
    *,
    diagnostic_labels: Sequence[str] = DIAGNOSTIC_LABELS,
) -> dict[str, Any]:
    """
    Inspect the low-level PDF objects intersecting a figure region.

    Reports text spans, PDF words, vector drawings, image blocks, diagnostic
    label presence, and whether text appears outlined/flattened.
    """
    clip = pymupdf.Rect(figure_rect)
    text_spans: list[dict[str, Any]] = []
    image_blocks: list[dict[str, Any]] = []
    seen_image_keys: set[tuple[float, float, float, float, int, int]] = set()

    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        block_type = block.get("type")
        block_rect = pymupdf.Rect(block.get("bbox", clip))

        if block_type == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if not str(span_text).strip():
                        continue
                    span_rect = pymupdf.Rect(span["bbox"])
                    if not _bbox_intersects(span_rect, clip):
                        continue
                    text_spans.append(
                        {
                            "text": str(span_text),
                            "bbox": _rect_to_list(span_rect),
                            "font": span.get("font"),
                            "size": float(span.get("size", 0.0)),
                            "color": _normalize_color(span.get("color")),
                            "flags": span.get("flags"),
                        }
                    )
        elif block_type == 1:
            if not _bbox_intersects(block_rect, clip):
                continue
            width = int(block.get("width") or 0)
            height = int(block.get("height") or 0)
            key = (
                round(block_rect.x0, 2),
                round(block_rect.y0, 2),
                round(block_rect.x1, 2),
                round(block_rect.y1, 2),
                width,
                height,
            )
            if key in seen_image_keys:
                continue
            seen_image_keys.add(key)
            image_blocks.append(
                {
                    "bbox": _rect_to_list(block_rect),
                    "width": width,
                    "height": height,
                }
            )

    for image in page.get_images(full=True):
        xref = image[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception as exc:  # noqa: BLE001 - defensive around malformed PDFs
            LOGGER.debug("Could not get image rects for xref %s: %s", xref, exc)
            continue
        for rect in rects:
            if not _bbox_intersects(rect, clip):
                continue
            try:
                pixmap = pymupdf.Pixmap(page.parent, xref)
                width = int(pixmap.width)
                height = int(pixmap.height)
            except Exception:  # noqa: BLE001
                width = 0
                height = 0
            key = (
                round(rect.x0, 2),
                round(rect.y0, 2),
                round(rect.x1, 2),
                round(rect.y1, 2),
                width,
                height,
            )
            if key in seen_image_keys:
                continue
            seen_image_keys.add(key)
            image_blocks.append(
                {
                    "bbox": _rect_to_list(rect),
                    "width": width,
                    "height": height,
                    "xref": int(xref),
                }
            )

    vector_drawings = []
    unsupported_ops: list[str] = []
    for drawing in page.get_drawings():
        if not _drawing_intersects_clip(drawing, clip):
            continue

        drawing_rect = drawing.get("rect")
        if drawing_rect is None:
            drawing_rect = pymupdf.Rect(clip)
        else:
            drawing_rect = pymupdf.Rect(drawing_rect)

        for item in drawing.get("items") or []:
            op = item[0]
            if op not in {"l", "re", "qu", "c"}:
                unsupported_ops.append(str(op))

        vector_drawings.append(
            {
                "rect": _rect_to_list(drawing_rect),
                "type": drawing.get("type"),
                "item_count": len(drawing.get("items") or []),
            }
        )

    pdf_words = extract_pdf_words(page, clip)
    label_diagnostics = check_diagnostic_labels(pdf_words, diagnostic_labels)

    vector_drawing_count = len(vector_drawings)
    has_pdf_text = bool(text_spans)
    has_vector_drawings = vector_drawing_count > 0
    has_raster_images = bool(image_blocks)

    outlined = classify_outlined_or_flattened_text(
        label_diagnostics=label_diagnostics,
        has_pdf_text=has_pdf_text,
        has_vector_drawings=has_vector_drawings,
        has_raster_images=has_raster_images,
    )

    classification = classify_figure_structure(
        has_pdf_text=has_pdf_text,
        has_vector_drawings=has_vector_drawings,
        has_raster_images=has_raster_images,
        vector_drawing_count=vector_drawing_count,
        has_outlined_or_flattened_text=outlined["has_outlined_or_flattened_text"],
    )

    return {
        "text_spans": text_spans,
        "pdf_words": pdf_words,
        "vector_drawing_count": vector_drawing_count,
        "vector_drawings": vector_drawings,
        "image_blocks": image_blocks,
        "image_block_count": len(image_blocks),
        "has_pdf_text": has_pdf_text,
        "has_vector_drawings": has_vector_drawings,
        "has_raster_images": has_raster_images,
        "classification": classification,
        "unsupported_drawing_ops": sorted(set(unsupported_ops)),
        "figure_rect": _rect_to_list(clip),
        "diagnostic_labels": label_diagnostics,
        "has_outlined_or_flattened_text": outlined["has_outlined_or_flattened_text"],
        "text_encoding": outlined,
        "text_separable_from_graphics": text_separable_from_graphics(
            {
                "has_outlined_or_flattened_text": outlined[
                    "has_outlined_or_flattened_text"
                ],
                "has_pdf_text": has_pdf_text,
                "classification": classification,
            }
        ),
        "notes": [
            PDF_TEXT_MASK_COVERAGE_NOTE,
            CANONICAL_IMAGE_NOTE,
        ],
    }


def pixmap_to_bgr(pixmap: pymupdf.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF pixmap to a BGR ``uint8`` array."""
    if pixmap.alpha:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    if pixmap.n == 1:
        gray = samples.reshape(pixmap.height, pixmap.width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = samples.reshape(pixmap.height, pixmap.width, pixmap.n)[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ndarray_to_pixmap(mask: np.ndarray) -> pymupdf.Pixmap:
    """Convert an HxW or HxWx3 ``uint8`` array to a PyMuPDF pixmap."""
    if mask.ndim == 2:
        height, width = mask.shape
        rgb = np.stack([mask, mask, mask], axis=-1)
    elif mask.ndim == 3 and mask.shape[2] >= 3:
        height, width = mask.shape[:2]
        rgb = mask[:, :, :3]
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    return pymupdf.Pixmap(
        pymupdf.csRGB,
        width,
        height,
        bytearray(np.ascontiguousarray(rgb).tobytes()),
        False,
    )


def build_box_mask(
    boxes: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    padding_px: int = DEFAULT_TEXT_MASK_PADDING_PX,
) -> np.ndarray:
    """Build a white-on-black uint8 mask from pixel-space boxes ``[x0,y0,x1,y1]``."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        if len(box) < 4:
            continue
        x0 = int(np.floor(box[0] - padding_px))
        y0 = int(np.floor(box[1] - padding_px))
        x1 = int(np.ceil(box[2] + padding_px))
        y1 = int(np.ceil(box[3] + padding_px))
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def build_pdf_text_mask(
    text_spans: list[dict[str, Any]],
    figure_rect: pymupdf.Rect | list[float] | tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    dpi: int,
    padding_px: int = DEFAULT_TEXT_MASK_PADDING_PX,
) -> pymupdf.Pixmap:
    """
    Build a white-on-black mask of extractable PDF text span boxes.

    Coordinates are translated relative to ``figure_rect`` and scaled by
    ``dpi / 72``. The mask only marks PDF text objects; outlined or
    flattened text is not represented.
    """
    clip = pymupdf.Rect(figure_rect)
    zoom = dpi_to_zoom(dpi)
    boxes: list[list[float]] = []
    for span in text_spans:
        bbox = span.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        boxes.append(
            [
                (x0 - clip.x0) * zoom,
                (y0 - clip.y0) * zoom,
                (x1 - clip.x0) * zoom,
                (y1 - clip.y0) * zoom,
            ]
        )
    return ndarray_to_pixmap(
        build_box_mask(boxes, width=width, height=height, padding_px=padding_px)
    )


def detect_raster_text_regions(
    image_bgr: np.ndarray,
    *,
    use_tesseract: bool = True,
    use_mser_proposals: bool = True,
    pdf_text_boxes_px: Sequence[Sequence[float]] | None = None,
    padding_px: int = DEFAULT_TEXT_MASK_PADDING_PX,
):
    """
    Detect text-like regions for in-plot masking (geometry-first).

    Returns a ``RasterTextDetectionResult``. OCR transcription is optional;
    acceptance is driven by character-like geometry, grouping, and spatial role.
    """
    return _detect_raster_text_regions_impl(
        image_bgr,
        use_tesseract=use_tesseract,
        use_mser_proposals=use_mser_proposals,
        pdf_text_boxes_px=pdf_text_boxes_px,
        padding_px=padding_px,
    )


def build_raster_text_mask(
    detections: Sequence[dict[str, Any]] | np.ndarray,
    *,
    width: int,
    height: int,
    padding_px: int = DEFAULT_TEXT_MASK_PADDING_PX,
) -> pymupdf.Pixmap:
    if isinstance(detections, np.ndarray):
        mask = detections
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {mask.shape} does not match {(height, width)}"
            )
        return ndarray_to_pixmap(mask.astype(np.uint8))
    boxes = [d["bbox"] for d in detections if d.get("bbox")]
    return ndarray_to_pixmap(
        build_box_mask(boxes, width=width, height=height, padding_px=padding_px)
    )


def _pixmap_to_gray_array(mask: np.ndarray | pymupdf.Pixmap) -> np.ndarray:
    if isinstance(mask, pymupdf.Pixmap):
        arr = np.frombuffer(mask.samples, dtype=np.uint8)
        if mask.n >= 3:
            arr = arr.reshape(mask.height, mask.width, mask.n)[:, :, 0]
        else:
            arr = arr.reshape(mask.height, mask.width)
        return arr.astype(np.uint8)
    arr = mask
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def combine_text_masks(
    *masks: np.ndarray | pymupdf.Pixmap,
    require_validation: bool = False,
    max_coverage: float = MAX_RASTER_MASK_COVERAGE,
    region: tuple[int, int, int, int] | None = None,
) -> pymupdf.Pixmap:
    """
    Union white-on-black masks into one pixmap.

    When ``require_validation`` is True, each mask must pass coverage checks
    before being unioned; failing masks are skipped (treated as empty).
    """
    arrays: list[np.ndarray] = []
    for mask in masks:
        arr = _pixmap_to_gray_array(mask)
        if require_validation:
            ok, _coverage, reason = validate_mask_for_combine(
                arr,
                max_coverage=max_coverage,
                region=region,
            )
            if not ok:
                LOGGER.info(
                    "Skipping mask in combine_text_masks (%s); using empty",
                    reason,
                )
                arr = np.zeros_like(arr)
        arrays.append(arr)

    if not arrays:
        raise ValueError("At least one mask is required")

    combined = arrays[0].copy()
    for arr in arrays[1:]:
        if arr.shape != combined.shape:
            raise ValueError(
                f"Mask shape mismatch: {arr.shape} vs {combined.shape}"
            )
        combined = np.maximum(combined, arr)
    return ndarray_to_pixmap(combined)


def score_curve_like_component(
    *,
    width: int,
    height: int,
    area: int,
    horizontal_coverage: int,
    plot_width: int,
    plot_height: int,
    skeleton_length: int | None = None,
    centroid_xy: tuple[float, float] | None = None,
    plot_rect: tuple[int, int, int, int] | None = None,
    text_overlap_fraction: float = 0.0,
) -> dict[str, Any]:
    """Delegate to the shared curve-vs-text scorer used in digitization."""
    try:
        from xrd_digitization.text_regions import (
            score_curve_like_component as _score,
        )
    except ImportError:  # pragma: no cover - script/legacy-only path
        from text_regions import score_curve_like_component as _score  # type: ignore

    return _score(
        width=width,
        height=height,
        area=area,
        horizontal_coverage=horizontal_coverage,
        plot_width=plot_width,
        plot_height=plot_height,
        skeleton_length=skeleton_length,
        centroid_xy=centroid_xy,
        plot_rect=plot_rect,
        text_overlap_fraction=text_overlap_fraction,
    )


def apply_text_mask_soft_penalty(
    curve_mask: np.ndarray,
    text_mask: np.ndarray | None,
    *,
    plot_left: int | None = None,
    plot_top: int | None = None,
    plot_right: int | None = None,
    plot_bottom: int | None = None,
) -> np.ndarray:
    """Delegate to the shared soft-penalty filter used in digitization."""
    try:
        from xrd_digitization.text_regions import (
            apply_text_mask_soft_penalty as _apply,
        )
    except ImportError:  # pragma: no cover - script/legacy-only path
        from text_regions import apply_text_mask_soft_penalty as _apply  # type: ignore

    return _apply(
        curve_mask,
        text_mask,
        plot_left=plot_left,
        plot_top=plot_top,
        plot_right=plot_right,
        plot_bottom=plot_bottom,
    )


def _translate_point(
    point: pymupdf.Point,
    origin: pymupdf.Point,
) -> pymupdf.Point:
    return pymupdf.Point(point.x - origin.x, point.y - origin.y)


def _translate_rect(
    rect: pymupdf.Rect,
    origin: pymupdf.Point,
) -> pymupdf.Rect:
    return pymupdf.Rect(
        rect.x0 - origin.x,
        rect.y0 - origin.y,
        rect.x1 - origin.x,
        rect.y1 - origin.y,
    )


def _replay_drawings_on_page(
    source_page: pymupdf.Page,
    dest_page: pymupdf.Page,
    clip: pymupdf.Rect,
) -> list[str]:
    """
    Replay vector drawings intersecting ``clip`` onto ``dest_page``.

    Experimental only. See ``VECTOR_RECONSTRUCTION_KNOWN_ISSUES``.
    """
    origin = pymupdf.Point(clip.x0, clip.y0)
    unsupported: list[str] = []

    for drawing in source_page.get_drawings():
        if not _drawing_intersects_clip(drawing, clip):
            continue

        shape = dest_page.new_shape()
        drew_anything = False
        for item in drawing.get("items") or []:
            op = item[0]
            if op == "l":
                p1 = _translate_point(item[1], origin)
                p2 = _translate_point(item[2], origin)
                shape.draw_line(p1, p2)
                drew_anything = True
            elif op == "re":
                shape.draw_rect(_translate_rect(pymupdf.Rect(item[1]), origin))
                drew_anything = True
            elif op == "qu":
                quad = item[1]
                points = [
                    _translate_point(quad.ul, origin),
                    _translate_point(quad.ur, origin),
                    _translate_point(quad.lr, origin),
                    _translate_point(quad.ll, origin),
                ]
                shape.draw_polyline(points)
                drew_anything = True
            elif op == "c":
                shape.draw_bezier(
                    _translate_point(item[1], origin),
                    _translate_point(item[2], origin),
                    _translate_point(item[3], origin),
                    _translate_point(item[4], origin),
                )
                drew_anything = True
            else:
                unsupported.append(str(op))
                LOGGER.warning(
                    "Unsupported drawing operation %r omitted from reconstruction",
                    op,
                )

        if not drew_anything:
            continue

        stroke_opacity = drawing.get("stroke_opacity")
        fill_opacity = drawing.get("fill_opacity")
        if stroke_opacity is None:
            stroke_opacity = 1.0
        if fill_opacity is None:
            fill_opacity = 1.0

        line_cap = drawing.get("lineCap")
        if isinstance(line_cap, (list, tuple)):
            line_cap = line_cap[0] if line_cap else 0

        finish_kwargs: dict[str, Any] = {
            "color": drawing.get("color"),
            "fill": drawing.get("fill"),
            "width": drawing.get("width") if drawing.get("width") is not None else 1,
            "stroke_opacity": stroke_opacity,
            "fill_opacity": fill_opacity,
            "closePath": bool(drawing.get("closePath")),
            "even_odd": bool(drawing.get("even_odd")),
        }
        dashes = drawing.get("dashes")
        if dashes:
            finish_kwargs["dashes"] = dashes
        if line_cap is not None:
            finish_kwargs["lineCap"] = line_cap
        if drawing.get("lineJoin") is not None:
            finish_kwargs["lineJoin"] = drawing.get("lineJoin")

        try:
            shape.finish(**finish_kwargs)
            shape.commit()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to finish drawing path: %s", exc)
            unsupported.append("finish_error")

    return sorted(set(unsupported))


def _copy_images_to_page(
    source_page: pymupdf.Page,
    dest_page: pymupdf.Page,
    clip: pymupdf.Rect,
) -> int:
    """Copy embedded images intersecting ``clip`` onto ``dest_page``."""
    origin = pymupdf.Point(clip.x0, clip.y0)
    document = source_page.parent
    copied = 0
    seen: set[tuple[int, float, float, float, float]] = set()

    for image in source_page.get_images(full=True):
        xref = image[0]
        try:
            rects = source_page.get_image_rects(xref)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Skipping image xref %s: %s", xref, exc)
            continue

        for rect in rects:
            if not _bbox_intersects(rect, clip):
                continue
            key = (
                int(xref),
                round(rect.x0, 2),
                round(rect.y0, 2),
                round(rect.x1, 2),
                round(rect.y1, 2),
            )
            if key in seen:
                continue
            seen.add(key)
            dest_rect = _translate_rect(rect, origin)
            try:
                pixmap = pymupdf.Pixmap(document, xref)
                if pixmap.n - pixmap.alpha > 3:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                dest_page.insert_image(dest_rect, pixmap=pixmap)
                copied += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to copy image xref %s into reconstruction: %s",
                    xref,
                    exc,
                )
    return copied


def render_without_pdf_text(
    page: pymupdf.Page,
    figure_rect: pymupdf.Rect | list[float] | tuple[float, float, float, float],
    *,
    dpi: int = 300,
    include_images: bool = True,
    inspection: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[pymupdf.Pixmap | None, dict[str, Any]]:
    """
    Experimental text-free rendering by reconstructing vectors (and images).

    Disabled unless ``force=True`` or inspection proves PDF text is stored
    separately from the data graphics. Even then, results are marked
    experimental because outlined glyphs remain indistinguishable from curves.
    """
    clip = pymupdf.Rect(figure_rect)
    inspection = inspection or inspect_figure_region(page, clip)
    classification = inspection.get("classification", "unknown")
    limitations: list[str] = list(VECTOR_RECONSTRUCTION_KNOWN_ISSUES)
    limitations.append(EXPERIMENTAL_RECONSTRUCTION_NOTE)

    if not force:
        return None, {
            "created": False,
            "method": None,
            "experimental": True,
            "limitations": [
                "Vector reconstruction is disabled unless force=True "
                "(experimental vectors_only / experimental_vector_reconstruction).",
                *limitations,
            ],
        }

    if inspection.get("has_outlined_or_flattened_text"):
        return None, {
            "created": False,
            "method": None,
            "experimental": True,
            "limitations": [OUTLINED_TEXT_LIMITATION, *limitations],
        }

    if not text_separable_from_graphics(inspection):
        return None, {
            "created": False,
            "method": None,
            "experimental": True,
            "limitations": [
                "Vector reconstruction skipped: text is not proven separable "
                "from data graphics.",
                *limitations,
            ],
        }

    if classification == "likely_fully_rasterized":
        return None, {
            "created": False,
            "method": None,
            "experimental": True,
            "limitations": [RASTER_TEXT_LIMITATION, *limitations],
        }

    has_vectors = bool(inspection.get("has_vector_drawings"))
    has_images = bool(inspection.get("has_raster_images"))
    if not has_vectors and not (include_images and has_images):
        return None, {
            "created": False,
            "method": None,
            "experimental": True,
            "limitations": [NO_RECONSTRUCTIBLE_CONTENT, *limitations],
        }

    if classification == "raster_with_pdf_text_overlay":
        limitations.append(
            "PDF text objects were omitted; text baked into the embedded "
            "raster image cannot be removed."
        )

    recon_doc = pymupdf.open()
    try:
        dest_page = recon_doc.new_page(width=clip.width, height=clip.height)
        dest_page.draw_rect(
            dest_page.rect,
            color=(1, 1, 1),
            fill=(1, 1, 1),
            width=0,
        )

        unsupported = _replay_drawings_on_page(page, dest_page, clip)
        images_copied = 0
        if include_images:
            images_copied = _copy_images_to_page(page, dest_page, clip)

        if unsupported:
            limitations.append(UNSUPPORTED_DRAWING_OPS_LIMITATION)
            limitations.append(
                "Unsupported operations: " + ", ".join(unsupported)
            )

        if not has_vectors and images_copied == 0:
            return None, {
                "created": False,
                "method": None,
                "experimental": True,
                "limitations": [NO_RECONSTRUCTIBLE_CONTENT, *limitations],
            }

        pixmap = dest_page.get_pixmap(dpi=dpi, alpha=False, annots=False)
        return pixmap, {
            "created": True,
            "method": "experimental_vector_reconstruction",
            "experimental": True,
            "limitations": limitations,
            "unsupported_drawing_ops": unsupported,
            "images_copied": images_copied,
            "known_issues": list(VECTOR_RECONSTRUCTION_KNOWN_ISSUES),
        }
    finally:
        recon_doc.close()


def _page_suffix(page_number: int, page_count_in_figure: int) -> str:
    if page_count_in_figure > 1:
        return f"_page_{page_number}"
    return ""


def extract_figure(
    pdf_path: str | Path,
    figure_coordinates: str,
    output_directory: str | Path,
    figure_id: str,
    mode: ExtractMode = "all",
    dpi: int | None = None,
    zoom: float | None = None,
    padding: float = 8.0,
    padding_left: float | None = None,
    padding_top: float | None = None,
    padding_right: float | None = None,
    padding_bottom: float | None = None,
    text_mask_padding_px: int = DEFAULT_TEXT_MASK_PADDING_PX,
    diagnostic_labels: Sequence[str] = DIAGNOSTIC_LABELS,
    detect_raster_text: bool = True,
    experimental_vector_reconstruction: bool = False,
    removal_method: RemovalMethod = DEFAULT_REMOVAL_METHOD,
    create_preprocessed: bool = True,
    caption_coords: str | None = None,
) -> dict[str, Any]:
    """
    Extract a figure region with structure diagnostics and text-region masks.

    Modes:
        original     - save normal rendering only (canonical image)
        inspect      - canonical image + PDF/raster/combined text masks + JSON
        all          - same as inspect (vector reconstruction is NOT default)
        vectors_only - experimental reconstruction only when text is separable
                       from graphics (or ``experimental_vector_reconstruction``)

    Output files (when applicable):
        {figure_id}_original.png
        {figure_id}_plot_only.png              (untouched crop)
        {figure_id}_preprocessed.png           (glyph-cleaned; default)
        {figure_id}_preprocessed_glyph.png
        {figure_id}_preprocessed_region.png    (diagnostic / more destructive)
        {figure_id}_preprocessed_overlay.png
        {figure_id}_pdf_text_mask.png
        {figure_id}_raster_text_mask.png
        {figure_id}_combined_text_mask.png
        {figure_id}_pdf_structure.json
        {figure_id}_without_pdf_text.png   (experimental only)
    """
    if mode not in {"original", "inspect", "all", "vectors_only"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    resolved_dpi = resolve_dpi(dpi=dpi, zoom=zoom)
    pdf_path = Path(pdf_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "figure_id": figure_id,
        "mode": mode,
        "dpi": resolved_dpi,
        "zoom": dpi_to_zoom(resolved_dpi),
        "pages": [],
        "output_paths": {},
        "notes": [
            CANONICAL_IMAGE_NOTE,
            PDF_TEXT_MASK_COVERAGE_NOTE,
            EXPERIMENTAL_RECONSTRUCTION_NOTE,
            "plot_only.png is an untouched crop; cleaned images are *_preprocessed*.png.",
        ],
    }

    document = pymupdf.open(pdf_path)
    try:
        clips = resolve_figure_page_clips(
            document,
            figure_coordinates,
            padding=padding,
            padding_left=padding_left,
            padding_top=padding_top,
            padding_right=padding_right,
            padding_bottom=padding_bottom,
            caption_coords=caption_coords,
        )
        page_count_in_figure = len(clips)

        for page_number, page, clip in clips:
            suffix = _page_suffix(page_number, page_count_in_figure)
            stem = f"{figure_id}{suffix}"
            page_outputs: dict[str, str] = {}

            # Canonical rendering: never replace this with reconstruction.
            original_pixmap = page.get_pixmap(
                clip=clip,
                dpi=resolved_dpi,
                alpha=False,
                annots=False,
            )
            original_path = output_directory / f"{stem}_original.png"
            original_pixmap.save(original_path)
            page_outputs["original"] = str(original_path)

            page_result: dict[str, Any] = {
                "page_number": page_number,
                "figure_rect": _rect_to_list(clip),
                "original_size": {
                    "width": original_pixmap.width,
                    "height": original_pixmap.height,
                },
                "canonical_image": "original",
            }

            if mode == "original":
                page_result["text_free_render"] = {
                    "created": False,
                    "method": None,
                    "experimental": True,
                    "limitations": [
                        "Not requested in mode=original",
                        EXPERIMENTAL_RECONSTRUCTION_NOTE,
                    ],
                }
                results["pages"].append(page_result)
                results["output_paths"][stem] = page_outputs
                continue

            inspection = inspect_figure_region(
                page,
                clip,
                diagnostic_labels=diagnostic_labels,
            )
            text_free_meta: dict[str, Any] = {
                "created": False,
                "method": None,
                "experimental": True,
                "limitations": [
                    "Vector reconstruction is disabled by default.",
                    EXPERIMENTAL_RECONSTRUCTION_NOTE,
                ],
            }

            pdf_mask = build_pdf_text_mask(
                inspection["text_spans"],
                clip,
                width=original_pixmap.width,
                height=original_pixmap.height,
                dpi=resolved_dpi,
                padding_px=text_mask_padding_px,
            )
            pdf_mask_path = output_directory / f"{stem}_pdf_text_mask.png"
            pdf_mask.save(pdf_mask_path)
            page_outputs["pdf_text_mask"] = str(pdf_mask_path)

            raster_detections: list[dict[str, Any]] = []
            raster_rejected: list[dict[str, Any]] = []
            raster_meta: dict[str, Any] = {
                "failed": False,
                "failure_reason": None,
                "coverage": 0.0,
                "plot_bbox": None,
                "caption_bbox": None,
                "panels": [],
                "notes": [],
            }
            image_bgr = pixmap_to_bgr(original_pixmap)
            full_crop_path = output_directory / f"{stem}_full_figure.png"
            cv2.imwrite(str(full_crop_path), image_bgr)
            page_outputs["full_figure"] = str(full_crop_path)

            if detect_raster_text:
                pdf_boxes_px = pdf_span_boxes_to_pixels(
                    inspection["text_spans"],
                    clip,
                    dpi=resolved_dpi,
                )
                raster_result = detect_raster_text_regions(
                    image_bgr,
                    pdf_text_boxes_px=pdf_boxes_px,
                    padding_px=min(text_mask_padding_px, 1),
                )
                raster_detections = list(raster_result.accepted)
                raster_rejected = list(raster_result.rejected)
                raster_meta = {
                    "failed": bool(raster_result.failed),
                    "failure_reason": raster_result.failure_reason,
                    "coverage": float(raster_result.coverage),
                    "plot_bbox": list(raster_result.plot_bbox)
                    if raster_result.plot_bbox
                    else None,
                    "caption_bbox": list(raster_result.caption_bbox)
                    if raster_result.caption_bbox
                    else None,
                    "inner_plot_bbox": list(raster_result.inner_plot_bbox)
                    if getattr(raster_result, "inner_plot_bbox", None)
                    else None,
                    "axis_bands": {
                        k: list(v)
                        for k, v in (getattr(raster_result, "axis_bands", None) or {}).items()
                    },
                    "panels": [list(p) for p in raster_result.panels],
                    "notes": list(raster_result.notes),
                    "accepted_count": len(raster_detections),
                    "rejected_count": len(raster_rejected),
                    "by_role": {
                        role: [
                            {
                                "text": d.get("recognized_text") or d.get("text"),
                                "recognized_text": d.get("recognized_text"),
                                "detection_confidence": d.get("detection_confidence"),
                                "ocr_confidence": d.get("ocr_confidence"),
                                "confidence": d.get("confidence"),
                                "bbox": d.get("bbox"),
                                "source": d.get("source"),
                                "role": d.get("role"),
                                "orientation": d.get("orientation"),
                                "n_components": d.get("n_components"),
                            }
                            for d in dets
                        ]
                        for role, dets in (raster_result.by_role or {}).items()
                    },
                    "diagnostics": dict(raster_result.diagnostics or {}),
                }

                if raster_result.plot_bbox is not None:
                    x0, y0, x1, y1 = raster_result.plot_bbox
                    plot_crop = image_bgr[y0:y1, x0:x1]
                    plot_path = output_directory / f"{stem}_plot_only.png"
                    cv2.imwrite(str(plot_path), plot_crop)
                    page_outputs["plot_only"] = str(plot_path)
                if raster_result.caption_bbox is not None:
                    x0, y0, x1, y1 = raster_result.caption_bbox
                    caption_crop = image_bgr[y0:y1, x0:x1]
                    caption_path = output_directory / f"{stem}_caption.png"
                    cv2.imwrite(str(caption_path), caption_crop)
                    page_outputs["caption"] = str(caption_path)

                def _save_mask_array(name: str, arr: np.ndarray | None) -> None:
                    if arr is None:
                        return
                    path = output_directory / f"{stem}_{name}.png"
                    cv2.imwrite(str(path), arr)
                    page_outputs[name] = str(path)

                _save_mask_array(
                    "all_text_candidate_mask",
                    getattr(raster_result, "all_text_candidate_mask", None),
                )
                _save_mask_array(
                    "removable_in_plot_text_mask",
                    getattr(raster_result, "removable_region_mask", None),
                )
                _save_mask_array(
                    "preserved_axis_text_mask",
                    getattr(raster_result, "preserved_axis_mask", None),
                )
                _save_mask_array(
                    "text_glyph_mask",
                    getattr(raster_result, "removable_glyph_mask", None),
                )

                preprocessed_meta: dict[str, Any] = {"created": False}
                if create_preprocessed and not raster_result.failed:
                    try:
                        removable_dets = [
                            d
                            for d in raster_detections
                            if str(d.get("role") or "").startswith("removable_")
                        ]
                        prep = preprocess_removable_text(
                            image_bgr,
                            glyph_mask=getattr(
                                raster_result, "removable_glyph_mask", None
                            ),
                            region_mask=getattr(
                                raster_result, "removable_region_mask", None
                            ),
                            preserved_axis_mask=getattr(
                                raster_result, "preserved_axis_mask", None
                            ),
                            detections=removable_dets,
                            plot_bbox=(
                                tuple(int(v) for v in raster_result.plot_bbox)
                                if raster_result.plot_bbox
                                else None
                            ),
                            inner_plot_bbox=(
                                tuple(int(v) for v in raster_result.inner_plot_bbox)
                                if getattr(raster_result, "inner_plot_bbox", None)
                                else None
                            ),
                            removal_method=removal_method,
                        )
                        pre_path = output_directory / f"{stem}_preprocessed.png"
                        glyph_path = (
                            output_directory / f"{stem}_preprocessed_glyph.png"
                        )
                        region_path = (
                            output_directory / f"{stem}_preprocessed_region.png"
                        )
                        overlay_path = (
                            output_directory / f"{stem}_preprocessed_overlay.png"
                        )
                        cv2.imwrite(str(pre_path), prep.preprocessed_bgr)
                        cv2.imwrite(str(glyph_path), prep.glyph_bgr)
                        cv2.imwrite(str(region_path), prep.region_bgr)
                        cv2.imwrite(str(overlay_path), prep.overlay_bgr)
                        page_outputs["preprocessed"] = str(pre_path)
                        page_outputs["preprocessed_glyph"] = str(glyph_path)
                        page_outputs["preprocessed_region"] = str(region_path)
                        page_outputs["preprocessed_overlay"] = str(overlay_path)
                        if prep.residual_debug_bgr is not None:
                            residual_path = (
                                output_directory / f"{stem}_debug_residual_text.png"
                            )
                            cv2.imwrite(str(residual_path), prep.residual_debug_bgr)
                            page_outputs["debug_residual_text"] = str(residual_path)
                        if prep.curve_damage_debug_bgr is not None:
                            damage_path = (
                                output_directory / f"{stem}_debug_curve_damage.png"
                            )
                            cv2.imwrite(str(damage_path), prep.curve_damage_debug_bgr)
                            page_outputs["debug_curve_damage"] = str(damage_path)
                        if prep.expanded_glyph_mask is not None:
                            exp_path = (
                                output_directory / f"{stem}_expanded_glyph_mask.png"
                            )
                            cv2.imwrite(str(exp_path), prep.expanded_glyph_mask)
                            page_outputs["expanded_glyph_mask"] = str(exp_path)
                        if prep.protected_curve_mask is not None:
                            prot_path = (
                                output_directory / f"{stem}_protected_curve_mask.png"
                            )
                            # Full-image protected mask was cropped with plot_bbox.
                            cv2.imwrite(str(prot_path), prep.protected_curve_mask)
                            page_outputs["protected_curve_mask"] = str(prot_path)

                        preprocessed_meta = {
                            "created": True,
                            "status": prep.status,
                            "glyph_output": str(glyph_path),
                            "region_output": str(region_path),
                            "preprocessed_output": str(pre_path),
                            "overlay_output": str(overlay_path),
                            "removal_method": prep.removal_method,
                            "removed_pixel_count": prep.removed_pixel_count,
                            "protected_curve_pixel_count": (
                                prep.protected_curve_pixel_count
                            ),
                            "residual_text_groups": prep.residual_text_groups,
                            "curve_damage_groups": prep.curve_damage_groups,
                            "notes": list(prep.notes),
                            **(prep.meta or {}),
                        }
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception("Preprocessed image creation failed: %s", exc)
                        preprocessed_meta = {
                            "created": False,
                            "status": "failed",
                            "error": str(exc),
                            "removal_method": removal_method,
                        }
                raster_meta["preprocessed_output"] = preprocessed_meta

                if raster_result.debug_accepted_bgr is not None:
                    accepted_dbg = output_directory / f"{stem}_debug_accepted.png"
                    cv2.imwrite(str(accepted_dbg), raster_result.debug_accepted_bgr)
                    page_outputs["debug_accepted"] = str(accepted_dbg)
                    # Keep legacy filename for older consumers.
                    legacy_accepted = (
                        output_directory / f"{stem}_raster_text_debug_accepted.png"
                    )
                    cv2.imwrite(str(legacy_accepted), raster_result.debug_accepted_bgr)
                    page_outputs["raster_text_debug_accepted"] = str(legacy_accepted)
                preserved_dbg = getattr(raster_result, "debug_preserved_bgr", None)
                if preserved_dbg is not None:
                    path = output_directory / f"{stem}_debug_preserved_axes.png"
                    cv2.imwrite(str(path), preserved_dbg)
                    page_outputs["debug_preserved_axes"] = str(path)
                if raster_result.debug_rejected_bgr is not None:
                    rejected_dbg = output_directory / f"{stem}_debug_rejected.png"
                    cv2.imwrite(str(rejected_dbg), raster_result.debug_rejected_bgr)
                    page_outputs["debug_rejected"] = str(rejected_dbg)
                    legacy_rejected = (
                        output_directory / f"{stem}_raster_text_debug_rejected.png"
                    )
                    cv2.imwrite(str(legacy_rejected), raster_result.debug_rejected_bgr)
                    page_outputs["raster_text_debug_rejected"] = str(legacy_rejected)

                # Digitization default: removable glyph mask.
                raster_mask_array = getattr(
                    raster_result, "removable_glyph_mask", None
                )
                if raster_mask_array is None:
                    raster_mask_array = raster_result.mask
                if raster_mask_array is None:
                    raster_mask_array = np.zeros(
                        (original_pixmap.height, original_pixmap.width),
                        dtype=np.uint8,
                    )
            else:
                raster_mask_array = np.zeros(
                    (original_pixmap.height, original_pixmap.width),
                    dtype=np.uint8,
                )

            # Refine outlined/flattened classification using OCR tokens that
            # match diagnostic labels but are absent from PDF words.
            label_set = {str(label).upper() for label in diagnostic_labels}
            raster_labels = []
            for det in raster_detections:
                token = str(det.get("text") or "").strip().upper()
                if not token:
                    continue
                if token in label_set:
                    raster_labels.append(token)
                else:
                    for label in label_set:
                        if label in token.split("-") or label in token.split("_"):
                            raster_labels.append(label)
            outlined = classify_outlined_or_flattened_text(
                label_diagnostics=inspection.get("diagnostic_labels") or {},
                has_pdf_text=bool(inspection.get("has_pdf_text")),
                has_vector_drawings=bool(inspection.get("has_vector_drawings")),
                has_raster_images=bool(inspection.get("has_raster_images")),
                visible_labels_from_raster=raster_labels,
            )
            inspection["has_outlined_or_flattened_text"] = outlined[
                "has_outlined_or_flattened_text"
            ]
            inspection["text_encoding"] = outlined
            inspection["classification"] = classify_figure_structure(
                has_pdf_text=bool(inspection.get("has_pdf_text")),
                has_vector_drawings=bool(inspection.get("has_vector_drawings")),
                has_raster_images=bool(inspection.get("has_raster_images")),
                vector_drawing_count=int(inspection.get("vector_drawing_count") or 0),
                has_outlined_or_flattened_text=outlined[
                    "has_outlined_or_flattened_text"
                ],
            )
            inspection["text_separable_from_graphics"] = text_separable_from_graphics(
                inspection
            )

            raster_mask = build_raster_text_mask(
                raster_mask_array,
                width=original_pixmap.width,
                height=original_pixmap.height,
                padding_px=text_mask_padding_px,
            )
            raster_mask_path = output_directory / f"{stem}_raster_text_mask.png"
            raster_mask.save(raster_mask_path)
            page_outputs["raster_text_mask"] = str(raster_mask_path)

            # Validate both masks before unioning. Keep PDF text mask even when
            # raster detection fails; never blindly OR an invalid raster mask.
            plot_region = (
                tuple(int(v) for v in raster_meta["plot_bbox"])
                if raster_meta.get("plot_bbox")
                else None
            )
            pdf_arr = _pixmap_to_gray_array(pdf_mask)
            pdf_ok, pdf_cov, pdf_reason = validate_mask_for_combine(
                pdf_arr,
                max_coverage=0.35,
                region=None,
            )
            raster_arr = _pixmap_to_gray_array(raster_mask)
            raster_ok, raster_cov, raster_reason = validate_mask_for_combine(
                raster_arr,
                max_coverage=MAX_RASTER_MASK_COVERAGE,
                region=plot_region,
            )
            if not pdf_ok:
                LOGGER.warning(
                    "PDF text mask failed validation (%s); excluding from combine",
                    pdf_reason,
                )
                pdf_for_combine = np.zeros_like(pdf_arr)
            else:
                pdf_for_combine = pdf_arr
            if not raster_ok or raster_meta.get("failed"):
                LOGGER.info(
                    "Raster text mask not combined (%s); keeping PDF-only union base",
                    raster_reason or raster_meta.get("failure_reason"),
                )
                raster_for_combine = np.zeros_like(raster_arr)
            else:
                raster_for_combine = raster_arr

            combined_mask = combine_text_masks(pdf_for_combine, raster_for_combine)
            combined_mask_path = (
                output_directory / f"{stem}_combined_text_mask.png"
            )
            combined_mask.save(combined_mask_path)
            page_outputs["combined_text_mask"] = str(combined_mask_path)
            raster_meta["pdf_mask_validation"] = {
                "ok": pdf_ok,
                "coverage": pdf_cov,
                "reason": pdf_reason,
            }
            raster_meta["raster_mask_validation"] = {
                "ok": raster_ok,
                "coverage": raster_cov,
                "reason": raster_reason,
            }

            run_experimental = mode == "vectors_only" or experimental_vector_reconstruction
            if run_experimental:
                force = mode == "vectors_only"
                text_free_pixmap, text_free_meta = render_without_pdf_text(
                    page,
                    clip,
                    dpi=resolved_dpi,
                    include_images=(mode != "vectors_only"),
                    inspection=inspection,
                    force=force,
                )
                if text_free_pixmap is not None:
                    text_free_path = (
                        output_directory / f"{stem}_without_pdf_text.png"
                    )
                    text_free_pixmap.save(text_free_path)
                    page_outputs["without_pdf_text"] = str(text_free_path)
                    text_free_meta = {
                        **text_free_meta,
                        "experimental": True,
                        "filename_note": (
                            "Experimental artifact; prefer original.png + "
                            "combined_text_mask.png for digitization."
                        ),
                    }
                elif mode == "vectors_only" and not inspection.get(
                    "has_vector_drawings"
                ):
                    text_free_meta = {
                        "created": False,
                        "method": None,
                        "experimental": True,
                        "limitations": [
                            "vectors_only mode found no vector drawings "
                            "to reconstruct.",
                            *text_free_meta.get("limitations", []),
                        ],
                    }

            structure_payload = {
                **inspection,
                "raster_text_detections": raster_detections,
                "raster_text_rejected": raster_rejected,
                "raster_text_detection_count": len(raster_detections),
                "raster_text_by_role": raster_meta.get("by_role") or {},
                "raster_text_meta": raster_meta,
                "preprocessed_output": raster_meta.get("preprocessed_output")
                or {"created": False},
                "text_free_render": text_free_meta,
                "dpi": resolved_dpi,
                "zoom": dpi_to_zoom(resolved_dpi),
                "original_size": page_result["original_size"],
                "output_paths": page_outputs,
                "diagnostics": {
                    "text_span_count": len(inspection["text_spans"]),
                    "pdf_word_count": len(inspection.get("pdf_words") or []),
                    "vector_drawing_count": inspection["vector_drawing_count"],
                    "image_block_count": inspection.get("image_block_count")
                    or len(inspection.get("image_blocks") or []),
                    "diagnostic_labels": inspection.get("diagnostic_labels"),
                    "has_outlined_or_flattened_text": inspection.get(
                        "has_outlined_or_flattened_text"
                    ),
                    "text_separable_from_graphics": inspection.get(
                        "text_separable_from_graphics"
                    ),
                },
                "notes": [
                    *inspection.get("notes", []),
                    PDF_TEXT_MASK_COVERAGE_NOTE,
                    CANONICAL_IMAGE_NOTE,
                    "plot_only.png is an untouched crop. "
                    "Cleaned digitization images are *_preprocessed*.png "
                    "(glyph default; region is diagnostic/fallback).",
                    "Canonical original.png is never erased; masks remain "
                    "available as soft penalties during curve extraction.",
                    EXPERIMENTAL_RECONSTRUCTION_NOTE,
                ],
                "vector_reconstruction_known_issues": list(
                    VECTOR_RECONSTRUCTION_KNOWN_ISSUES
                ),
            }
            structure_path = output_directory / f"{stem}_pdf_structure.json"
            structure_path.write_text(
                json.dumps(structure_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            page_outputs["pdf_structure"] = str(structure_path)

            page_result["inspection"] = {
                "classification": inspection["classification"],
                "has_pdf_text": inspection["has_pdf_text"],
                "has_vector_drawings": inspection["has_vector_drawings"],
                "has_raster_images": inspection["has_raster_images"],
                "vector_drawing_count": inspection["vector_drawing_count"],
                "text_span_count": len(inspection["text_spans"]),
                "image_block_count": inspection.get("image_block_count")
                or len(inspection.get("image_blocks") or []),
                "has_outlined_or_flattened_text": inspection.get(
                    "has_outlined_or_flattened_text"
                ),
                "diagnostic_labels": inspection.get("diagnostic_labels"),
                "raster_text_detection_count": len(raster_detections),
            }
            page_result["text_free_render"] = text_free_meta
            results["pages"].append(page_result)
            results["output_paths"][stem] = page_outputs

            LOGGER.info(
                "Extracted figure %s page %s (%s); experimental_text_free=%s; "
                "outlined_text=%s; raster_dets=%s",
                figure_id,
                page_number,
                inspection["classification"],
                text_free_meta.get("created"),
                inspection.get("has_outlined_or_flattened_text"),
                len(raster_detections),
            )
    finally:
        document.close()

    return results
