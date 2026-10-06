#!/usr/bin/env python3
"""
Extract compact XRD scientific metadata from cropped figure images via OpenAI vision.

Does NOT digitize curves or extract XY spectra. Input is already-cropped figure images.
Output emphasizes peaks, phases, hkl labels, FWHM/broadening, and curve conditions.

Examples:

    export OPENAI_API_KEY=...
    python scripts/analyze_figures.py data/analysis/ output/figure_metadata/

    python scripts/analyze_figures.py data/analysis/figure_1.png output/figure_metadata/ \\
        --workers 2 --model gpt-4.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "autonomous_digitizer"
for path in (ROOT, AUTO):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from figure_analysis.batch import run_batch  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyze cropped scientific figure images into structured JSON "
            "(observed / estimated / interpretation). No curve digitization."
        )
    )
    p.add_argument(
        "input_path",
        type=Path,
        help="Figure image file or directory of images (recursive by default)",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for per-figure folders and global JSONL",
    )
    p.add_argument(
        "--model",
        default=None,
        help="OpenAI model (default: OPENAI_MODEL or gpt-4.1)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent workers (default: 1)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-analyze figures even if analysis JSON already exists",
    )
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write human-readable *_summary.txt files",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable OpenAI response disk cache",
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="When input is a directory, only scan the top level",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    stats = run_batch(
        args.input_path,
        args.output_dir,
        model=args.model,
        workers=args.workers,
        overwrite=args.overwrite,
        write_summary=not args.no_summary,
        use_cache=not args.no_cache,
        recursive=not args.no_recursive,
    )

    if stats.total == 0:
        return 1
    if stats.failed and stats.succeeded == 0 and stats.skipped == 0:
        return 1
    if stats.failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
