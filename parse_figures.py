from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import pymupdf

LOGGER = logging.getLogger(__name__)

MIN_GRAPHIC_HEIGHT = 30.0
MIN_GRAPHIC_WIDTH = 50.0


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
    caption_boxes = parse_grobid_coords(figure_coords)
    if not caption_boxes:
        return None

    boxes_by_page: dict[int, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )
    for page_number, x, y, width, height in caption_boxes:
        boxes_by_page[page_number].append(
            (x, y, x + width, y + height)
        )

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
            caption_width = max(caption_x1 - caption_x0, 1.0)

            best_rect: pymupdf.Rect | None = None
            best_area = 0.0

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

                area = rect.width * rect.height
                if area > best_area:
                    best_area = area
                    best_rect = rect

            if best_rect is not None:
                inferred_boxes.append(
                    (
                        page_number,
                        best_rect.x0,
                        best_rect.y0,
                        best_rect.width,
                        best_rect.height,
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
    if graphic_coords and graphic_coords.strip():
        return graphic_coords.strip()

    if not figure_coords or not figure_coords.strip():
        return None

    image_boxes = [
        box
        for box in parse_grobid_coords(figure_coords)
        if box[3] >= MIN_GRAPHIC_WIDTH and box[4] >= MIN_GRAPHIC_HEIGHT
    ]

    if not image_boxes:
        if pdf_path is not None and figure_coords:
            return infer_graphic_coords_from_pdf(pdf_path, figure_coords)
        return None

    return format_grobid_coords(image_boxes)


def crop_figure_from_grobid_coords(
    pdf_path: str | Path,
    coords_string: str,
    output_dir: str | Path,
    figure_id: str,
    dpi: int = 300,
    padding: float = 8.0,
) -> list[Path]:
    """
    Render a figure region identified by GROBID coordinates.

    One image is produced per page if the logical figure spans pages.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

            x0 = min(box[0] for box in page_boxes) - padding
            y0 = min(box[1] for box in page_boxes) - padding
            x1 = max(box[2] for box in page_boxes) + padding
            y1 = max(box[3] for box in page_boxes) + padding

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

        if xrd_only and not getattr(figure, "is_likely_xrd", False):
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
