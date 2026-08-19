#!/usr/bin/env python3
"""
Inspect a PDF figure region and write structure / text-free outputs.

Page numbers are **1-based** (GROBID convention). The rectangle is given in
PDF user-space points as ``x0 y0 x1 y1``.

Example:

    python scripts/inspect_pdf_figure.py \\
      --pdf paper.pdf \\
      --page 4 \\
      --rect 100 200 500 600 \\
      --output debug/figure_test \\
      --zoom 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
for path in (str(LEGACY), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pdf_figure_structure import extract_figure  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect PDF objects in a figure region and optionally reconstruct "
            "a rendering without PDF text objects. "
            "--page is 1-based (GROBID convention)."
        )
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the PDF")
    parser.add_argument(
        "--page",
        required=True,
        type=int,
        help="1-based page number (GROBID convention; first page is 1)",
    )
    parser.add_argument(
        "--rect",
        required=True,
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Figure rectangle in PDF points: x0 y0 x1 y1",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for inspection artifacts",
    )
    parser.add_argument(
        "--figure-id",
        default="figure",
        help="Output filename stem (default: figure)",
    )
    parser.add_argument(
        "--mode",
        choices=("original", "inspect", "all", "vectors_only"),
        default="all",
        help=(
            "Extraction mode (default: all). "
            "'all'/'inspect' save the canonical pixmap plus PDF/raster/"
            "combined text masks and diagnostics. "
            "'vectors_only' is experimental reconstruction and is not the "
            "default text-free path."
        ),
    )
    zoom_dpi = parser.add_mutually_exclusive_group()
    zoom_dpi.add_argument(
        "--zoom",
        type=float,
        help="Render zoom factor (dpi = zoom * 72). Mutually exclusive with --dpi.",
    )
    zoom_dpi.add_argument(
        "--dpi",
        type=int,
        help="Render DPI (default: 300 if neither --dpi nor --zoom is set)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Padding in PDF points around the rectangle (default: 0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.page < 1:
        parser.error("--page must be a 1-based page number (>= 1)")

    x0, y0, x1, y1 = args.rect
    if x1 <= x0 or y1 <= y0:
        parser.error("--rect must satisfy x1 > x0 and y1 > y0")

    # GROBID coordinate string: page,x,y,width,height
    coords = f"{args.page},{x0},{y0},{x1 - x0},{y1 - y0}"

    result = extract_figure(
        pdf_path=args.pdf,
        figure_coordinates=coords,
        output_directory=args.output,
        figure_id=args.figure_id,
        mode=args.mode,
        dpi=args.dpi,
        zoom=args.zoom,
        padding=args.padding,
        padding_left=args.padding,
        padding_top=args.padding,
        padding_right=args.padding,
        padding_bottom=args.padding,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
