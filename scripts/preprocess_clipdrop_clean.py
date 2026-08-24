#!/usr/bin/env python3
"""
Clean figure PNGs with ClipDrop Remove Text on the plot interior only.

Axis spines, tick labels, and surrounding titles stay pixel-identical to the
original. Only an inset crop of the plotting region is sent to ClipDrop; the
cleaned crop is pasted back into a copy of the original.

Typical use with pairs under data/figures_with_text/:

    export CLIPDROP_API_KEY=...
    python scripts/preprocess_clipdrop_clean.py data/figures_with_text

    # Validate bbox / exterior identity without calling the API:
    python scripts/preprocess_clipdrop_clean.py data/figures_with_text --dry-run

    # Also write *.axes.json from the originals (for digitize reuse):
    python scripts/preprocess_clipdrop_clean.py data/figures_with_text --extract-axes

Writes figure_N_clean.png next to figure_N.png (never overwrites the original).
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
from xrd_digitization.clipdrop_remove_text import (  # noqa: E402
    ClipdropError,
    clean_figure_preserve_axes,
)
from xrd_digitization.plot_interior_crop import DEFAULT_INSET_FRAC  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _iter_original_pngs(directory: Path) -> list[Path]:
    pngs = sorted(directory.glob("*.png"))
    return [p for p in pngs if not p.stem.endswith("_clean")]


def _parse_bbox(raw: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in raw.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "bbox must be x0,y0,x1,y1 (four integers)"
        )
    try:
        x0, y0, x1, y1 = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox values must be integers") from exc
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("bbox requires x1>x0 and y1>y0")
    return x0, y0, x1, y1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Original figure PNG, or a directory of PNGs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for *_clean.png (default: beside each original)",
    )
    parser.add_argument(
        "--bbox",
        type=_parse_bbox,
        default=None,
        help="Manual axes/plot rectangle x0,y0,x1,y1 (inset still applied unless --inset 0)",
    )
    parser.add_argument(
        "--inset",
        type=int,
        default=None,
        help=(
            "Fixed inward inset in pixels from the axes frame "
            f"(default: max(4, {DEFAULT_INSET_FRAC:.3f}*min_side))"
        ),
    )
    parser.add_argument(
        "--no-inset-on-manual",
        action="store_true",
        help="When --bbox is set, use it as-is without further inset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip ClipDrop; paste the original crop back (validates exterior identity)",
    )
    parser.add_argument(
        "--extract-axes",
        action="store_true",
        help="Also write *.axes.json from each original before cleaning",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_clean.png / *.axes.json outputs",
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

    input_path = args.input.resolve()
    if input_path.is_dir():
        pngs = _iter_original_pngs(input_path)
    elif input_path.is_file():
        pngs = [input_path]
    else:
        LOGGER.error("Not found: %s", input_path)
        return 1

    if not pngs:
        LOGGER.error("No original PNGs found under %s", input_path)
        return 1

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"cleaned": 0, "skipped": 0, "failed": 0, "axes": 0}

    for png_path in pngs:
        if args.extract_axes:
            axes_path = png_path.with_name(f"{png_path.stem}.axes.json")
            if axes_path.is_file() and not args.overwrite:
                LOGGER.info("skip axes (exists): %s", axes_path.name)
            else:
                try:
                    sidecar = extract_axis_sidecar_for_path(
                        png_path, output_path=axes_path
                    )
                    usable, reasons = x_calibration_is_usable(sidecar.calibration)
                    if not usable:
                        LOGGER.warning(
                            "%s: weak X calib (%s)", png_path.name, ", ".join(reasons)
                        )
                    counts["axes"] += 1
                except AxisSidecarError as exc:
                    LOGGER.error("axes %s: %s", png_path.name, exc)
                    counts["failed"] += 1

        if args.output_dir is not None:
            suffix = "_clean_dryrun.png" if args.dry_run else "_clean.png"
            out_path = args.output_dir / f"{png_path.stem}{suffix}"
        elif args.dry_run:
            out_path = png_path.with_name(f"{png_path.stem}_clean_dryrun.png")
        else:
            out_path = png_path.with_name(f"{png_path.stem}_clean.png")

        if out_path.is_file() and not args.overwrite:
            LOGGER.info("skip clean (exists): %s", out_path.name)
            counts["skipped"] += 1
            continue

        try:
            result = clean_figure_preserve_axes(
                png_path,
                output_path=out_path,
                bbox=args.bbox,
                inset_px=args.inset,
                apply_inset_to_manual=not args.no_inset_on_manual,
                dry_run=args.dry_run,
            )
        except (ClipdropError, ValueError) as exc:
            LOGGER.error("%s: %s", png_path.name, exc)
            counts["failed"] += 1
            continue

        LOGGER.info(
            "%s → %s interior=%s frame=%s method=%s warnings=%s",
            png_path.name,
            out_path.name,
            result.interior.bbox,
            result.interior.frame_bbox,
            result.interior.method,
            result.warnings,
        )
        counts["cleaned"] += 1

    LOGGER.info(
        "done cleaned=%d skipped=%d failed=%d axes=%d",
        counts["cleaned"],
        counts["skipped"],
        counts["failed"],
        counts["axes"],
    )
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
