from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import pymupdf

LOGGER = logging.getLogger(__name__)

MIN_GRAPHIC_HEIGHT = 30.0
MIN_GRAPHIC_WIDTH = 50.0
MIN_IMAGE_BAND_AREA = 5000.0
MIN_IMAGE_BAND_HEIGHT = 30.0
MAX_CAPTION_BAND_GAP = 150.0
IMAGE_BAND_VERTICAL_GAP = 20.0
AXIS_LABEL_LEFT_ANCHOR = 18.0
AXIS_LABEL_BOTTOM_MARGIN = 40.0
AXIS_LABEL_TOP_MARGIN = 20.0
AXIS_LABEL_RIGHT_MARGIN = 15.0
DEFAULT_CROP_PADDING_LEFT = 12.0
DEFAULT_CROP_PADDING_TOP = 8.0
DEFAULT_CROP_PADDING_RIGHT = 8.0
DEFAULT_CROP_PADDING_BOTTOM = 12.0
# Keep a tiny gap so padded crops do not swallow caption baselines. Axis-label
# image strips often end only a few points above the caption, so this must stay
# small — much smaller than the text-expansion caption_gap.
CAPTION_CROP_CLEARANCE = 1.0


class FigureLike(Protocol):
    figure_id: str | None
    coords: str | None
    graphic_coords: str | None
    image_paths: list[str]


def parse_grobid_coords(
    coords_string: str,
) -> list[tuple[int, float, float, float, float]]:
    """
    Parse a GROBID coordinate string.

    Input format:
        "page,x,y,width,height;page,x,y,width,height"

    Returns:
        [(page, x, y, width, height), ...]
    """
    boxes = []

    for raw_box in coords_string.split(";"):
        raw_box = raw_box.strip()
        if not raw_box:
            continue

        fields = raw_box.split(",")

        if len(fields) != 5:
            raise ValueError(f"Invalid GROBID coordinate: {raw_box!r}")

        page_number = int(fields[0])
        x = float(fields[1])
        y = float(fields[2])
        width = float(fields[3])
        height = float(fields[4])

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid box size in coordinate: {raw_box!r}")

        boxes.append((page_number, x, y, width, height))

    return boxes


def format_grobid_coords(
    boxes: list[tuple[int, float, float, float, float]],
) -> str:
    return ";".join(
        f"{page},{x},{y},{width},{height}"
        for page, x, y, width, height in boxes
    )


def _is_graphic_sized_box(width: float, height: float) -> bool:
    """True for plot-sized GROBID boxes (not thin caption text lines)."""
    return width >= MIN_GRAPHIC_WIDTH and height >= MIN_GRAPHIC_HEIGHT


def _caption_boxes_by_page(
    figure_coords: str,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """
    Caption-sized boxes from a figure's GROBID ``coords`` string.

    GROBID often packs both caption text lines and the ``<graphic>`` bbox into
    the same ``coords`` attribute. Only the short text lines are captions;
    graphic-sized boxes must be ignored so caption clamping / axis expansion
    uses the real caption top rather than the plot top.
    """
    boxes_by_page: dict[int, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )

    for page_number, x, y, width, height in parse_grobid_coords(figure_coords):
        if _is_graphic_sized_box(width, height):
            continue
        boxes_by_page[page_number].append(
            (x, y, x + width, y + height)
        )

    return boxes_by_page


def _unique_image_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects: list[pymupdf.Rect] = []
    seen: set[tuple[float, float, float, float]] = set()

    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            key = (
                round(rect.x0, 1),
                round(rect.y0, 1),
                round(rect.x1, 1),
                round(rect.y1, 1),
            )

            if key in seen:
                continue

            seen.add(key)
            rects.append(rect)

    return rects


def _group_rects_into_bands(
    rects: list[pymupdf.Rect],
    *,
    vertical_gap: float = IMAGE_BAND_VERTICAL_GAP,
) -> list[pymupdf.Rect]:
    if not rects:
        return []

    sorted_rects = sorted(rects, key=lambda rect: rect.y0)
    bands: list[pymupdf.Rect] = []
    current_group: list[pymupdf.Rect] = []
    current_bottom = float("-inf")

    for rect in sorted_rects:
        if current_group and rect.y0 > current_bottom + vertical_gap:
            band = pymupdf.Rect(current_group[0])
            for group_rect in current_group[1:]:
                band.include_rect(group_rect)
            bands.append(band)
            current_group = [rect]
        else:
            current_group.append(rect)

        current_bottom = max(current_bottom, rect.y1)

    if current_group:
        band = pymupdf.Rect(current_group[0])
        for group_rect in current_group[1:]:
            band.include_rect(group_rect)
        bands.append(band)

    return bands


def _expand_rect_for_axis_labels(
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    caption_y0: float,
    *,
    caption_gap: float = 12.0,
) -> pymupdf.Rect:
    """
    Expand a figure bounding box to include nearby axis labels and tick text.

    Plot data is often stored as vector paths whose bounds exclude the axes.
    """
    max_y = caption_y0 - caption_gap
    expanded = pymupdf.Rect(rect)

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue

                x0, y0, x1, y1 = span["bbox"]
                if y0 >= max_y:
                    continue

                vertically_near = (
                    y1 >= rect.y0 - AXIS_LABEL_TOP_MARGIN
                    and y0 <= rect.y1 + AXIS_LABEL_BOTTOM_MARGIN
                )
                if not vertically_near:
                    continue

                if (
                    x1 <= rect.x0 + 2.0
                    and x0 >= rect.x0 - AXIS_LABEL_LEFT_ANCHOR
                ):
                    expanded.x0 = min(expanded.x0, x0)

                if (
                    y0 >= rect.y1 - 2.0
                    and y0 <= max_y
                    and x1 >= rect.x0 - 5.0
                    and x0 <= rect.x1 + AXIS_LABEL_RIGHT_MARGIN
                ):
                    expanded.y1 = max(expanded.y1, y1)

                if (
                    y1 <= rect.y0 + 2.0
                    and y0 >= rect.y0 - AXIS_LABEL_TOP_MARGIN
                    and x1 >= rect.x0 - 5.0
                    and x0 <= rect.x1 + AXIS_LABEL_RIGHT_MARGIN
                ):
                    expanded.y0 = min(expanded.y0, y0)

    return expanded


def _union_drawing_rects_for_caption(
    page: pymupdf.Page,
    caption_x0: float,
    caption_y0: float,
    caption_x1: float,
    *,
    caption_gap: float = 12.0,
    min_overlap_ratio: float = 0.2,
) -> pymupdf.Rect | None:
    caption_width = max(caption_x1 - caption_x0, 1.0)
    effective_overlap = min_overlap_ratio
    if caption_width >= 300.0:
        effective_overlap = min(min_overlap_ratio, 0.12)
    candidate_rects: list[pymupdf.Rect] = []

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue

        # Keep drawings that end just above the caption (axis ticks / frames).
        # Only skip paths that start at or below the caption top.
        if rect.y0 >= caption_y0:
            continue

        overlap_x0 = max(rect.x0, caption_x0)
        overlap_x1 = min(rect.x1, caption_x1)
        overlap_width = overlap_x1 - overlap_x0

        if overlap_width / caption_width < effective_overlap:
            continue

        candidate_rects.append(rect)

    if not candidate_rects:
        return None

    union_rect = pymupdf.Rect(candidate_rects[0])
    for rect in candidate_rects[1:]:
        union_rect.include_rect(rect)

    # Do not let path unions spill into the caption itself.
    max_y1 = caption_y0 - CAPTION_CROP_CLEARANCE
    if union_rect.y1 > max_y1:
        if union_rect.y0 >= max_y1:
            return None
        union_rect.y1 = max_y1
    if union_rect.height < MIN_GRAPHIC_HEIGHT or union_rect.width < MIN_GRAPHIC_WIDTH:
        return None

    return _expand_rect_for_axis_labels(
        page,
        union_rect,
        caption_y0,
        caption_gap=caption_gap,
    )


def _expand_crop_coords_for_axis_labels(
    pdf_path: str | Path,
    crop_coords: str,
    figure_coords: str,
    *,
    caption_gap: float = 12.0,
) -> str:
    caption_boxes = _caption_boxes_by_page(figure_coords)
    crop_boxes = parse_grobid_coords(crop_coords)
    document = pymupdf.open(pdf_path)

    try:
        expanded_boxes: list[tuple[int, float, float, float, float]] = []

        for page_number, x, y, width, height in crop_boxes:
            page_captions = caption_boxes.get(page_number)
            if not page_captions or not 1 <= page_number <= document.page_count:
                expanded_boxes.append((page_number, x, y, width, height))
                continue

            page = document.load_page(page_number - 1)
            caption_y0 = min(box[1] for box in page_captions)
            expanded = _expand_rect_for_axis_labels(
                page,
                pymupdf.Rect(x, y, x + width, y + height),
                caption_y0,
                caption_gap=caption_gap,
            )
            expanded_boxes.append(
                (
                    page_number,
                    expanded.x0,
                    expanded.y0,
                    expanded.width,
                    expanded.height,
                )
            )
    finally:
        document.close()

    return format_grobid_coords(expanded_boxes)


def _select_image_band_for_caption(
    image_rects: list[pymupdf.Rect],
    caption_x0: float,
    caption_y0: float,
    caption_x1: float,
    *,
    caption_gap: float = 12.0,
    min_overlap_ratio: float = 0.2,
) -> pymupdf.Rect | None:
    """
    Pick the image band sitting just above a caption.

    ``caption_gap`` is retained for call-site compatibility; strip inclusion is
    based on whether the strip starts above the caption. Publishers often
    rasterize x-axis tick labels into a final strip that ends only a few points
    above the caption, so requiring a large bottom clearance cuts those values.

    For multi-panel figures rendered as stacked image strips, nearby qualifying
    bands are unioned so the crop keeps all panels rather than only the lowest.
    """
    _ = caption_gap  # inclusion uses caption top; see docstring
    caption_width = max(caption_x1 - caption_x0, 1.0)
    # Wide Nature-style captions often underlap side panels; relax slightly.
    effective_overlap = min_overlap_ratio
    if caption_width >= 300.0:
        effective_overlap = min(min_overlap_ratio, 0.12)

    candidate_rects: list[pymupdf.Rect] = []

    for rect in image_rects:
        # Include strips that end near the caption (axis ticks). Skip only
        # images that begin at/below the caption top.
        if rect.y0 >= caption_y0:
            continue

        overlap_x0 = max(rect.x0, caption_x0)
        overlap_x1 = min(rect.x1, caption_x1)
        overlap_width = overlap_x1 - overlap_x0

        if overlap_width / caption_width < effective_overlap:
            if rect.width < MIN_GRAPHIC_WIDTH:
                continue

        candidate_rects.append(rect)

    if not candidate_rects:
        return None

    qualifying_bands: list[pymupdf.Rect] = []
    max_y1 = caption_y0 - CAPTION_CROP_CLEARANCE

    for band in _group_rects_into_bands(candidate_rects):
        if band.y1 > max_y1:
            if band.y0 >= max_y1:
                continue
            band = pymupdf.Rect(band)
            band.y1 = max_y1

        area = band.width * band.height

        if area < MIN_IMAGE_BAND_AREA or band.height < MIN_IMAGE_BAND_HEIGHT:
            continue

        caption_gap_distance = caption_y0 - band.y1

        # Allow taller multi-panel stacks: gap limit applies to the lowest band
        # only; upper panels are merged below when vertically adjacent.
        if caption_gap_distance > MAX_CAPTION_BAND_GAP * 3:
            continue

        qualifying_bands.append(band)

    if not qualifying_bands:
        return None

    # Prefer bands near the caption, then union upward while gaps stay small.
    qualifying_bands.sort(key=lambda rect: rect.y1, reverse=True)
    union = pymupdf.Rect(qualifying_bands[0])
    for band in qualifying_bands[1:]:
        vertical_gap = union.y0 - band.y1
        if vertical_gap > IMAGE_BAND_VERTICAL_GAP * 4:
            continue
        # Require some horizontal affinity with the growing union / caption.
        overlap_x0 = max(band.x0, min(union.x0, caption_x0))
        overlap_x1 = min(band.x1, max(union.x1, caption_x1))
        if overlap_x1 - overlap_x0 < min(band.width, union.width) * 0.15:
            continue
        union.include_rect(band)

    if union.height < MIN_IMAGE_BAND_HEIGHT or union.width < MIN_GRAPHIC_WIDTH:
        return None

    return union


def infer_graphic_coords_from_embedded_images(
    pdf_path: str | Path,
    figure_coords: str,
    *,
    caption_gap: float = 12.0,
    min_overlap_ratio: float = 0.2,
) -> str | None:
    """
    Infer a figure crop from embedded image strips when GROBID lacks graphics.

    Some publishers rasterize plots as many thin horizontal image strips rather
    than vector paths or a single bitmap object.
    """
    boxes_by_page = _caption_boxes_by_page(figure_coords)
    if not boxes_by_page:
        return None

    inferred_boxes: list[tuple[int, float, float, float, float]] = []
    document = pymupdf.open(pdf_path)

    try:
        for page_number, page_captions in boxes_by_page.items():
            if not 1 <= page_number <= document.page_count:
                continue

            page = document.load_page(page_number - 1)
            caption_x0 = min(box[0] for box in page_captions)
            caption_y0 = min(box[1] for box in page_captions)
            caption_x1 = max(box[2] for box in page_captions)

            best_band = _select_image_band_for_caption(
                _unique_image_rects(page),
                caption_x0,
                caption_y0,
                caption_x1,
                caption_gap=caption_gap,
                min_overlap_ratio=min_overlap_ratio,
            )

            if best_band is not None:
                expanded_band = _expand_rect_for_axis_labels(
                    page,
                    best_band,
                    caption_y0,
                    caption_gap=caption_gap,
                )
                inferred_boxes.append(
                    (
                        page_number,
                        expanded_band.x0,
                        expanded_band.y0,
                        expanded_band.width,
                        expanded_band.height,
                    )
                )
    finally:
        document.close()

    if not inferred_boxes:
        return None

    return format_grobid_coords(inferred_boxes)


def infer_graphic_coords_from_pdf(
    pdf_path: str | Path,
    figure_coords: str,
    *,
    caption_gap: float = 12.0,
    min_overlap_ratio: float = 0.2,
) -> str | None:
    """
    Infer a figure crop from vector drawings when GROBID only reports captions.

    Useful for PDFs where plots are drawn as paths rather than embedded images.
    """
    boxes_by_page = _caption_boxes_by_page(figure_coords)
    if not boxes_by_page:
        return None

    inferred_boxes: list[tuple[int, float, float, float, float]] = []
    document = pymupdf.open(pdf_path)

    try:
        for page_number, page_captions in boxes_by_page.items():
            if not 1 <= page_number <= document.page_count:
                continue

            page = document.load_page(page_number - 1)
            caption_x0 = min(box[0] for box in page_captions)
            caption_y0 = min(box[1] for box in page_captions)
            caption_x1 = max(box[2] for box in page_captions)

            figure_rect = _union_drawing_rects_for_caption(
                page,
                caption_x0,
                caption_y0,
                caption_x1,
                caption_gap=caption_gap,
                min_overlap_ratio=min_overlap_ratio,
            )

            if figure_rect is not None:
                inferred_boxes.append(
                    (
                        page_number,
                        figure_rect.x0,
                        figure_rect.y0,
                        figure_rect.width,
                        figure_rect.height,
                    )
                )
    finally:
        document.close()

    if not inferred_boxes:
        return None

    return format_grobid_coords(inferred_boxes)


def _caption_top_by_page(
    figure_coords: str | None,
    *,
    crop_boxes_by_page: dict[int, list[tuple[float, float, float, float]]]
    | None = None,
) -> dict[int, float]:
    """
    Map 1-based page number → top of caption text boxes on that page.

    When ``crop_boxes_by_page`` is provided, ignore side-panel / in-figure text
    that sits beside the graphic (common GROBID mis-tag). Only boxes below the
    crop, or overlapping it horizontally in the lower portion, clamp the crop.
    """
    if not figure_coords or not figure_coords.strip():
        return {}

    tops: dict[int, float] = {}
    for page_number, boxes in _caption_boxes_by_page(figure_coords).items():
        if not boxes:
            continue

        crop_boxes = (crop_boxes_by_page or {}).get(page_number)
        if not crop_boxes:
            tops[page_number] = min(box[1] for box in boxes)
            continue

        crop_x0 = min(box[0] for box in crop_boxes)
        crop_y0 = min(box[1] for box in crop_boxes)
        crop_x1 = max(box[2] for box in crop_boxes)
        crop_y1 = max(box[3] for box in crop_boxes)
        crop_width = max(crop_x1 - crop_x0, 1.0)
        crop_height = max(crop_y1 - crop_y0, 1.0)

        below_or_footer: list[tuple[float, float, float, float]] = []
        for box in boxes:
            x0, y0, x1, y1 = box
            # True captions sit under the graphic.
            if y0 >= crop_y1 - 8.0:
                below_or_footer.append(box)
                continue
            # Ignore text that starts near the top of the crop (side labels /
            # panel letters) unless it heavily overlaps and lies in the bottom
            # third — still prefer below-crop boxes when any exist.
            overlap = max(0.0, min(x1, crop_x1) - max(x0, crop_x0))
            if (
                overlap / crop_width >= 0.35
                and y0 >= crop_y0 + 0.65 * crop_height
            ):
                below_or_footer.append(box)

        if below_or_footer:
            tops[page_number] = min(box[1] for box in below_or_footer)
        # If every "caption" box is side text, do not clamp.
    return tops


def _coords_area_sum(coords_string: str | None) -> float:
    if not coords_string or not str(coords_string).strip():
        return 0.0
    total = 0.0
    for page, x, y, width, height in parse_grobid_coords(coords_string):
        _ = (page, x, y)
        total += max(width, 0.0) * max(height, 0.0)
    return total


def _infer_crop_from_figure_coords(
    pdf_path: str | Path,
    figure_coords: str,
) -> str | None:
    crop_coords = infer_graphic_coords_from_pdf(pdf_path, figure_coords)
    if not crop_coords:
        crop_coords = infer_graphic_coords_from_embedded_images(
            pdf_path,
            figure_coords,
        )
    return crop_coords


def _prefer_larger_related_crop(
    primary: str,
    alternative: str | None,
    *,
    min_area_ratio: float = 1.35,
) -> str:
    """
    Prefer ``alternative`` when it covers substantially more area than
    ``primary`` on the same page(s) — typical of partial GROBID graphic boxes
    on multi-panel figures.
    """
    if not alternative:
        return primary

    primary_boxes = parse_grobid_coords(primary)
    alt_boxes = parse_grobid_coords(alternative)
    if not primary_boxes or not alt_boxes:
        return primary

    primary_pages = {box[0] for box in primary_boxes}
    alt_pages = {box[0] for box in alt_boxes}
    if primary_pages.isdisjoint(alt_pages):
        return primary

    primary_area = _coords_area_sum(primary)
    alt_area = _coords_area_sum(alternative)
    if primary_area <= 0:
        return alternative
    if alt_area < primary_area * min_area_ratio:
        return primary

    # Require rough spatial relatedness on at least one shared page.
    for page in primary_pages & alt_pages:
        p_boxes = [box for box in primary_boxes if box[0] == page]
        a_boxes = [box for box in alt_boxes if box[0] == page]
        px0 = min(box[1] for box in p_boxes)
        py0 = min(box[2] for box in p_boxes)
        px1 = max(box[1] + box[3] for box in p_boxes)
        py1 = max(box[2] + box[4] for box in p_boxes)
        ax0 = min(box[1] for box in a_boxes)
        ay0 = min(box[2] for box in a_boxes)
        ax1 = max(box[1] + box[3] for box in a_boxes)
        ay1 = max(box[2] + box[4] for box in a_boxes)
        overlap_x = max(0.0, min(px1, ax1) - max(px0, ax0))
        overlap_y = max(0.0, min(py1, ay1) - max(py0, ay0))
        primary_area_page = max((px1 - px0) * (py1 - py0), 1.0)
        if overlap_x * overlap_y >= primary_area_page * 0.25:
            LOGGER.info(
                "Replacing undersized GROBID graphic crop "
                "(area %.0f) with inferred crop (area %.0f)",
                primary_area,
                alt_area,
            )
            return alternative

    return primary


def select_figure_crop_coords(
    figure_coords: str | None,
    graphic_coords: str | None,
    pdf_path: str | Path | None = None,
) -> str | None:
    """
    Choose the best GROBID coordinate string for cropping a figure image.

    Prefer explicit ``<graphic coords="...">`` values from GROBID, but replace
    them when PDF inference finds a substantially larger related region (partial
    graphic boxes on multi-panel figures). When graphics are missing, fall back
    to large element boxes and vector/image inference.
    """
    crop_coords: str | None = None
    inferred: str | None = None

    if pdf_path is not None and figure_coords and figure_coords.strip():
        inferred = _infer_crop_from_figure_coords(pdf_path, figure_coords)

    if graphic_coords and graphic_coords.strip():
        crop_coords = _prefer_larger_related_crop(
            graphic_coords.strip(),
            inferred,
        )
    elif not figure_coords or not figure_coords.strip():
        return None
    else:
        image_boxes = [
            box
            for box in parse_grobid_coords(figure_coords)
            if _is_graphic_sized_box(box[3], box[4])
        ]

        if image_boxes:
            from_element = format_grobid_coords(image_boxes)
            crop_coords = _prefer_larger_related_crop(from_element, inferred)
        else:
            crop_coords = inferred

    if not crop_coords:
        return None

    if (
        pdf_path is not None
        and figure_coords
        and figure_coords.strip()
    ):
        crop_coords = _expand_crop_coords_for_axis_labels(
            pdf_path,
            crop_coords,
            figure_coords,
        )

    return crop_coords


def resolve_figure_page_clips(
    document: pymupdf.Document,
    coords_string: str,
    *,
    padding: float = 8.0,
    padding_left: float | None = None,
    padding_top: float | None = None,
    padding_right: float | None = None,
    padding_bottom: float | None = None,
    caption_coords: str | None = None,
    max_y1_by_page: dict[int, float] | None = None,
) -> list[tuple[int, pymupdf.Page, pymupdf.Rect]]:
    """
    Resolve GROBID coordinate boxes to padded, page-clamped clip rects.

    GROBID page numbers are 1-based; PyMuPDF pages are 0-based. Coordinates are
    already in PDF user space and need no DPI conversion.

    When ``caption_coords`` (or ``max_y1_by_page``) is provided, padded bottoms
    are clamped so crops do not swallow captions. This matters when the plot's
    own bottom strip already ends close to the caption and default bottom
    padding would otherwise overlap it.
    """
    pad_left = (
        padding_left
        if padding_left is not None
        else max(padding, DEFAULT_CROP_PADDING_LEFT)
    )
    pad_top = (
        padding_top
        if padding_top is not None
        else max(padding, DEFAULT_CROP_PADDING_TOP)
    )
    pad_right = (
        padding_right
        if padding_right is not None
        else max(padding, DEFAULT_CROP_PADDING_RIGHT)
    )
    pad_bottom = (
        padding_bottom
        if padding_bottom is not None
        else max(padding, DEFAULT_CROP_PADDING_BOTTOM)
    )

    boxes_by_page: dict[int, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )
    for page_number, x, y, width, height in parse_grobid_coords(coords_string):
        boxes_by_page[page_number].append((x, y, x + width, y + height))

    if max_y1_by_page is None and caption_coords:
        max_y1_by_page = {
            page: top - CAPTION_CROP_CLEARANCE
            for page, top in _caption_top_by_page(
                caption_coords,
                crop_boxes_by_page=boxes_by_page,
            ).items()
        }

    resolved: list[tuple[int, pymupdf.Page, pymupdf.Rect]] = []
    for page_number, page_boxes in sorted(boxes_by_page.items()):
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"GROBID page {page_number} is outside PDF range "
                f"1–{document.page_count}"
            )

        # GROBID pages are 1-based; PyMuPDF pages are 0-based.
        page = document.load_page(page_number - 1)

        x0 = min(box[0] for box in page_boxes) - pad_left
        y0 = min(box[1] for box in page_boxes) - pad_top
        x1 = max(box[2] for box in page_boxes) + pad_right
        y1 = max(box[3] for box in page_boxes) + pad_bottom

        if max_y1_by_page and page_number in max_y1_by_page:
            clamped = max_y1_by_page[page_number]
            # Never clamp away most of the graphic — side-text false captions
            # previously collapsed multi-panel figures to a few pixels tall.
            min_keep = y0 + 0.45 * max(y1 - y0, 1.0)
            if clamped >= min_keep:
                y1 = min(y1, clamped)

        # Keep the crop inside the visible page.
        x0 = max(page.rect.x0, x0)
        y0 = max(page.rect.y0, y0)
        x1 = min(page.rect.x1, x1)
        y1 = min(page.rect.y1, y1)

        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"Invalid crop on page {page_number}: "
                f"({x0}, {y0}, {x1}, {y1})"
            )

        resolved.append((page_number, page, pymupdf.Rect(x0, y0, x1, y1)))

    return resolved


def crop_figure_from_grobid_coords(
    pdf_path: str | Path,
    coords_string: str,
    output_dir: str | Path,
    figure_id: str,
    dpi: int = 300,
    padding: float = 8.0,
    padding_left: float | None = None,
    padding_top: float | None = None,
    padding_right: float | None = None,
    padding_bottom: float | None = None,
    caption_coords: str | None = None,
) -> list[Path]:
    """
    Render a figure region identified by GROBID coordinates.

    One image is produced per page if the logical figure spans pages.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    document = pymupdf.open(pdf_path)

    try:
        clips = resolve_figure_page_clips(
            document,
            coords_string,
            padding=padding,
            padding_left=padding_left,
            padding_top=padding_top,
            padding_right=padding_right,
            padding_bottom=padding_bottom,
            caption_coords=caption_coords,
        )

        for page_number, page, clip in clips:
            pixmap = page.get_pixmap(
                clip=clip,
                dpi=dpi,
                alpha=False,
                annots=False,
            )

            suffix = (
                f"_page_{page_number}"
                if len(clips) > 1
                else ""
            )

            output_path = output_dir / f"{figure_id}{suffix}.png"
            pixmap.save(output_path)
            output_paths.append(output_path)

    finally:
        document.close()

    return output_paths


_FIGURE_CAPTION_LEAD_PATTERN = re.compile(
    r"^\s*(?:extended\s+data\s+|supplementary\s+(?:information\s+)?"
    r"|supporting\s+(?:information\s+)?|data\s+)?fig(?:ure)?s?\.?\b",
    flags=re.IGNORECASE,
)


def _looks_like_figure_caption(caption: str | None) -> bool:
    text = (caption or "").strip()
    if not text:
        # Empty captions are allowed for graphic-only Extended Data pages.
        return False
    return bool(_FIGURE_CAPTION_LEAD_PATTERN.match(text))


def extract_document_figures(
    pdf_path: str | Path,
    figures: list[FigureLike],
    output_dir: str | Path,
    *,
    dpi: int = 300,
    padding: float = 8.0,
    xrd_only: bool = False,
) -> int:
    """
    Crop figure images from a PDF using GROBID coordinates.

    Updates each figure's ``image_paths`` in place and returns the number of
    figures successfully extracted.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    extracted_count = 0

    for figure in figures:
        figure.image_paths = []

        if xrd_only and not getattr(figure, "is_caption_xrd", False):
            continue

        figure_id = figure.figure_id or "unknown_figure"
        graphic_coords = getattr(figure, "graphic_coords", None)
        caption = getattr(figure, "caption", None)

        # Body-text phantoms (no <graphic>, caption is prose) produce junk crops.
        if not (graphic_coords and str(graphic_coords).strip()):
            if caption and not _looks_like_figure_caption(caption):
                coords = getattr(figure, "coords", None)
                has_graphic_box = False
                if coords and str(coords).strip():
                    has_graphic_box = any(
                        _is_graphic_sized_box(box[3], box[4])
                        for box in parse_grobid_coords(coords)
                    )
                if not has_graphic_box:
                    LOGGER.info(
                        "Skipping non-figure fragment %s (no graphic, "
                        "caption is not a figure lead-in).",
                        figure_id,
                    )
                    continue

        crop_coords = select_figure_crop_coords(
            figure.coords,
            graphic_coords,
            pdf_path=pdf_path,
        )

        if not crop_coords:
            LOGGER.warning(
                "Skipping figure %s: no usable GROBID graphic coordinates.",
                figure_id,
            )
            continue

        try:
            image_paths = crop_figure_from_grobid_coords(
                pdf_path=pdf_path,
                coords_string=crop_coords,
                output_dir=output_dir,
                figure_id=figure_id,
                dpi=dpi,
                padding=padding,
                caption_coords=figure.coords,
            )
        except (OSError, ValueError, pymupdf.FileDataError) as exc:
            LOGGER.warning(
                "Failed to extract figure %s: %s",
                figure_id,
                exc,
            )
            continue

        figure.image_paths = [str(path) for path in image_paths]
        extracted_count += 1
        LOGGER.info(
            "Extracted figure %s to %s",
            figure_id,
            ", ".join(figure.image_paths),
        )

    return extracted_count


def __getattr__(name: str):
    """Lazy re-exports for PDF structure helpers (avoids import cycles)."""
    if name in {
        "extract_figure",
        "inspect_figure_region",
        "render_without_pdf_text",
        "build_pdf_text_mask",
    }:
        import pdf_figure_structure as _pdf_figure_structure

        return getattr(_pdf_figure_structure, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
