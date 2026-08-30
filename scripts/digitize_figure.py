#!/usr/bin/env python3
"""
Digitize XRD figure PNG(s) into figure directories (CSV + digitized preview).

Accepts a single PNG path or a directory of PNGs. By default digitizes the
original image (no ClipDrop API calls). Pass --clipdrop to remove in-plot text
via ClipDrop first, then digitize the cleaned PNG.

Examples:

    python scripts/digitize_figure.py examples/figure_3.png output/

    python scripts/digitize_figure.py examples/ output/

    export CLIPDROP_API_KEY=...
    python scripts/digitize_figure.py examples/figure_3.png output/ --clipdrop
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotdigitizer_pipeline import digitize_figure_image  # noqa: E402
from xrd_digitization.clipdrop_remove_text import (  # noqa: E402
    ClipdropError,
    clean_figure_preserve_axes,
)

LOGGER = logging.getLogger(__name__)

_CLEAN_SUFFIX_RE = re.compile(r"_clean(?:_dryrun)?$", re.IGNORECASE)
_SKIP_STEM_SUFFIXES = ("_clean", "_clean_dryrun", "_digitized")


def figure_id_from_stem(stem: str) -> str:
    """``figure_1_clean`` → ``figure_1``; otherwise return the stem unchanged."""
    return _CLEAN_SUFFIX_RE.sub("", stem) or stem


def _is_source_png(path: Path) -> bool:
    """True for source figure PNGs (skip cleaned / digitized derivatives)."""
    stem = path.stem.lower()
    return not any(stem.endswith(suffix) for suffix in _SKIP_STEM_SUFFIXES)


def collect_pngs(input_path: Path) -> list[Path]:
    """
    Resolve ``input_path`` to a list of PNG files to digitize.

    A file path yields that single PNG. A directory yields sorted ``*.png``
    sources in that directory (non-recursive), excluding ``*_clean.png`` and
    ``*_digitized.png``.
    """
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            raise ValueError(f"Expected a .png file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        pngs = sorted(p for p in input_path.glob("*.png") if _is_source_png(p))
        if not pngs:
            raise FileNotFoundError(f"No source PNGs found under {input_path}")
        return pngs
    raise FileNotFoundError(f"Not found: {input_path}")


def digitize_one_figure(
    png_path: Path,
    output_dir: Path,
    *,
    use_clipdrop: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Digitize ``png_path`` into ``output_dir/<figure_id>/``.

    Returns the figure output directory.
    """
    png_path = png_path.resolve()
    if not png_path.is_file():
        raise FileNotFoundError(f"PNG not found: {png_path}")
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"Expected a .png file, got: {png_path}")

    figure_id = figure_id_from_stem(png_path.stem)
    figure_dir = output_dir.resolve() / figure_id

    if figure_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {figure_dir} (pass --overwrite to replace)"
            )
        shutil.rmtree(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    original_copy = figure_dir / png_path.name
    if original_copy.resolve() != png_path.resolve():
        shutil.copy2(png_path, original_copy)

    digitize_path = png_path
    if use_clipdrop:
        clean_path = figure_dir / f"{figure_id}_clean.png"
        LOGGER.info("ClipDrop cleaning %s → %s", png_path.name, clean_path.name)
        clean_figure_preserve_axes(png_path, output_path=clean_path)
        digitize_path = clean_path
    else:
        LOGGER.info("Digitizing original (ClipDrop skipped): %s", png_path.name)

    result = digitize_figure_image(
        digitize_path,
        figure_dir,
        figure_id=figure_id,
    )
    if not result.bands:
        raise RuntimeError("digitize_figure_image returned no bands")

    succeeded = [b for b in result.bands if b.success and b.csv_path and b.csv_path.is_file()]
    if not succeeded:
        errors = "; ".join(b.error or "unknown" for b in result.bands)
        raise RuntimeError(f"digitization failed: {errors}")

    LOGGER.info(
        "Wrote %d band(s) under %s",
        len(succeeded),
        figure_dir,
    )
    return figure_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a figure PNG, or a directory of PNGs to digitize",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will contain <figure_id>/ with CSV and digitized PNG",
    )
    parser.add_argument(
        "--clipdrop",
        action="store_true",
        help=(
            "Run ClipDrop Remove Text on the plot interior before digitizing "
            "(requires CLIPDROP_API_KEY; spends API credits)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output/<figure_id>/ directory",
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

    try:
        pngs = collect_pngs(args.input)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for png_path in pngs:
        LOGGER.info("=== %s ===", png_path.name)
        try:
            figure_dir = digitize_one_figure(
                png_path,
                args.output_dir,
                use_clipdrop=args.clipdrop,
                overwrite=args.overwrite,
            )
        except (FileNotFoundError, FileExistsError, ValueError, ClipdropError, RuntimeError) as exc:
            LOGGER.error("%s: %s", png_path.name, exc)
            failed.append((png_path, str(exc)))
            continue
        except Exception as exc:
            LOGGER.exception("Digitization failed for %s", png_path.name)
            failed.append((png_path, str(exc)))
            continue
        ok.append(figure_dir)
        print(figure_dir)

    if len(pngs) > 1:
        LOGGER.info("Done: %d succeeded, %d failed (of %d)", len(ok), len(failed), len(pngs))
        for path, err in failed:
            LOGGER.error("  failed %s: %s", path.name, err)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
