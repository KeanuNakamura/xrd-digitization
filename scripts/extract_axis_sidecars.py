#!/usr/bin/env python3
"""
Extract axis calibration sidecars from original (text-bearing) figure PNGs.

Typical use with ClipDrop pairs under data/figures_with_text/:

    python scripts/extract_axis_sidecars.py data/figures_with_text

Writes figure_N.axes.json next to figure_N.png (skips *_clean.png).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xrd_digitization.axis_sidecar import (  # noqa: E402
    AxisSidecarError,
    extract_axis_sidecar_for_path,
    x_calibration_is_usable,
)

LOGGER = logging.getLogger(__name__)


def _iter_original_pngs(directory: Path) -> list[Path]:
    pngs = sorted(directory.glob("*.png"))
    return [p for p in pngs if not p.stem.endswith("_clean")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing original figure PNGs (and optional *_clean.png)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .axes.json files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        LOGGER.error("Not a directory: %s", input_dir)
        return 1

    pngs = _iter_original_pngs(input_dir)
    if not pngs:
        LOGGER.error("No original PNGs found in %s", input_dir)
        return 1

    counts = {"wrote": 0, "skipped": 0, "failed": 0, "unusable": 0}
    for png_path in pngs:
        out_path = png_path.with_name(f"{png_path.stem}.axes.json")
        if out_path.is_file() and not args.overwrite:
            LOGGER.info("skip (exists): %s", out_path.name)
            counts["skipped"] += 1
            continue
        try:
            sidecar = extract_axis_sidecar_for_path(png_path, output_path=out_path)
        except AxisSidecarError as exc:
            LOGGER.error("%s: %s", png_path.name, exc)
            counts["failed"] += 1
            continue
        usable, reasons = x_calibration_is_usable(sidecar.calibration)
        if not usable:
            LOGGER.warning(
                "%s: X calibration weak (%s); sidecar written but digitize will reject it",
                png_path.name,
                ", ".join(reasons),
            )
            counts["unusable"] += 1
        else:
            if sidecar.warnings:
                LOGGER.info("%s warnings: %s", png_path.name, sidecar.warnings)
            counts["wrote"] += 1

    LOGGER.info(
        "done wrote=%d skipped=%d failed=%d unusable=%d",
        counts["wrote"],
        counts["skipped"],
        counts["failed"],
        counts["unusable"],
    )
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
