from __future__ import annotations

import logging
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


def _caption_boxes_by_page(
    figure_coords: str,
) -> dict[int, list[tuple[float, float, float, float]]]:
    boxes_by_page: dict[int, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )

    for page_number, x, y, width, height in parse_grobid_coords(figure_coords):
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
    candidate_rects: list[pymupdf.Rect] = []

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue

        if rect.y1 > caption_y0 - caption_gap:
            continue

        overlap_x0 = max(rect.x0, caption_x0)
        overlap_x1 = min(rect.x1, caption_x1)
        overlap_width = overlap_x1 - overlap_x0

        if overlap_width / caption_width < min_overlap_ratio:
            continue

        candidate_rects.append(rect)

    if not candidate_rects:
        return None

    union_rect = pymupdf.Rect(candidate_rects[0])
    for rect in candidate_rects[1:]:
        union_rect.include_rect(rect)

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
    caption_width = max(caption_x1 - caption_x0, 1.0)
    max_image_bottom = caption_y0 - caption_gap

    candidate_rects: list[pymupdf.Rect] = []

    for rect in image_rects:
        if rect.y1 > max_image_bottom:
            continue

        overlap_x0 = max(rect.x0, caption_x0)
        overlap_x1 = min(rect.x1, caption_x1)
        overlap_width = overlap_x1 - overlap_x0

        if overlap_width / caption_width < min_overlap_ratio:
            if rect.width < MIN_GRAPHIC_WIDTH:
                continue

        candidate_rects.append(rect)

    if not candidate_rects:
        return None

    best_band: pymupdf.Rect | None = None

    for band in _group_rects_into_bands(candidate_rects):
        area = band.width * band.height

        if area < MIN_IMAGE_BAND_AREA or band.height < MIN_IMAGE_BAND_HEIGHT:
            continue

        caption_gap_distance = caption_y0 - band.y1

        if caption_gap_distance > MAX_CAPTION_BAND_GAP:
            continue

        if best_band is None or band.y1 > best_band.y1:
            best_band = band

    return best_band


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


def select_figure_crop_coords(
    figure_coords: str | None,
    graphic_coords: str | None,
    pdf_path: str | Path | None = None,
) -> str | None:
    """
    Choose the best GROBID coordinate string for cropping a figure image.

    Prefer explicit ``<graphic coords="...">`` values from GROBID. When those
    are missing, fall back to large bounding boxes on the figure element and
    ignore caption-sized text boxes.
    """
    crop_coords: str | None

    if graphic_coords and graphic_coords.strip():
        crop_coords = graphic_coords.strip()
    elif not figure_coords or not figure_coords.strip():
        return None
    else:
        image_boxes = [
            box
            for box in parse_grobid_coords(figure_coords)
            if box[3] >= MIN_GRAPHIC_WIDTH and box[4] >= MIN_GRAPHIC_HEIGHT
        ]

        if not image_boxes:
            if pdf_path is not None and figure_coords:
                crop_coords = infer_graphic_coords_from_pdf(
                    pdf_path,
                    figure_coords,
                )
                if not crop_coords:
                    crop_coords = infer_graphic_coords_from_embedded_images(
                        pdf_path,
                        figure_coords,
                    )
            else:
                return None
        else:
            crop_coords = format_grobid_coords(image_boxes)

    if (
        crop_coords
        and pdf_path is not None
        and figure_coords
        and figure_coords.strip()
    ):
        crop_coords = _expand_crop_coords_for_axis_labels(
            pdf_path,
            crop_coords,
            figure_coords,
        )

    return crop_coords


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
) -> list[Path]:
    """
    Render a figure region identified by GROBID coordinates.

    One image is produced per page if the logical figure spans pages.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    boxes = parse_grobid_coords(coords_string)

    boxes_by_page: dict[int, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )

    for page_number, x, y, width, height in boxes:
        boxes_by_page[page_number].append(
            (x, y, x + width, y + height)
        )

    output_paths = []
    document = pymupdf.open(pdf_path)

    try:
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

            # Keep the crop inside the visible page.
            x0 = max(page.rect.x0, x0)
            y0 = max(page.rect.y0, y0)
            x1 = min(page.rect.x1, x1)
            y1 = min(page.rect.y1, y1)

            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    f"Invalid crop for figure {figure_id} on page {page_number}"
                )

            clip = pymupdf.Rect(x0, y0, x1, y1)

            pixmap = page.get_pixmap(
                clip=clip,
                dpi=dpi,
                alpha=False,
                annots=False,
            )

            suffix = (
                f"_page_{page_number}"
                if len(boxes_by_page) > 1
                else ""
            )

            output_path = output_dir / f"{figure_id}{suffix}.png"
            pixmap.save(output_path)
            output_paths.append(output_path)

    finally:
        document.close()

    return output_paths


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
        crop_coords = select_figure_crop_coords(
            figure.coords,
            figure.graphic_coords,
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
