#!/usr/bin/env python3
"""
Generate preprocessed figure crops + text masks from pdf_parser / GROBID outputs.

Examples:

    # All papers under a parsed-output root
    python scripts/preprocess_parsed_figures.py grobid_output/sample_pdfs

    # One paper
    python scripts/preprocess_parsed_figures.py grobid_output/sample_pdfs/carbonStacking

    # Custom PDF search dir / output
    python scripts/preprocess_parsed_figures.py grobid_output/sample_pdfs \\
      --pdf-dir pdf_files/sample_pdfs \\
      --output grobid_output/sample_pdfs/figures_without_text \\
      --dpi 300
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

from preprocess_parsed_figures import (  # noqa: E402
    DEFAULT_DPI,
    DEFAULT_MODE,
    default_output_root,
    preprocess_parsed_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess figures from parsed PDF outputs "
            "(*.parsed.json under paper dirs). Writes originals and "
            "PDF/raster/combined text masks under figures_without_text/<stem>/."
        )
    )
    parser.add_argument(
        "parsed_input",
        type=Path,
        help=(
            "Parsed-output path: a paper directory, a root of paper "
            "directories, or a *.parsed.json file"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output root for <stem>/ folders "
            "(default: <batch_root>/figures_without_text)"
        ),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory to search for <stem>.pdf when source_pdf is missing",
    )
    parser.add_argument(
        "--mode",
        choices=("original", "inspect", "all", "vectors_only"),
        default=DEFAULT_MODE,
        help="extract_figure mode (default: all)",
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
        default=None,
        help=f"Render DPI (default: {DEFAULT_DPI} if neither --dpi nor --zoom is set)",
    )
    parser.add_argument(
        "--xrd-only",
        action="store_true",
        help="Only process figures marked is_likely_xrd in the parse",
    )
    parser.add_argument(
        "--figure-id",
        action="append",
        dest="figure_ids",
        default=None,
        help="Only process this figure_id (repeatable)",
    )
    parser.add_argument(
        "--paper",
        action="append",
        dest="paper_names",
        default=None,
        help="Only process this paper stem when input is a root (repeatable)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have a manifest.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete each paper's output directory before writing",
    )
    parser.add_argument(
        "--removal-method",
        choices=("white", "local_background", "inpaint", "mask_only"),
        default="local_background",
        help=(
            "How to remove in-plot text when writing *_preprocessed*.png "
            "(default: local_background)"
        ),
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

    parsed_input = args.parsed_input
    if not parsed_input.exists():
        parser.error(f"parsed_input not found: {parsed_input}")

    dpi = args.dpi
    zoom = args.zoom
    if dpi is None and zoom is None:
        dpi = DEFAULT_DPI

    output_root = args.output
    if output_root is None:
        output_root = default_output_root(parsed_input)

    summary = preprocess_parsed_outputs(
        parsed_input,
        output_root=output_root,
        pdf_dir=args.pdf_dir,
        mode=args.mode,
        dpi=dpi if zoom is None else DEFAULT_DPI,
        zoom=zoom,
        xrd_only=args.xrd_only,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
        figure_ids=args.figure_ids,
        paper_names=args.paper_names,
        removal_method=args.removal_method,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    errors = [p for p in summary.get("papers") or [] if p.get("status") == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
