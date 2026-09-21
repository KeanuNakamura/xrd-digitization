#!/usr/bin/env python3
"""
Digitize XRD figure PNG(s) into figure directories (CSV + digitized preview).

Accepts a single PNG path or a directory of PNGs. By default digitizes the
original image (no ClipDrop API calls). Pass --clipdrop to remove in-plot text
via ClipDrop first, then digitize the cleaned PNG.

For stacked multi-curve figures use ``--layout stacked`` and optionally
``--clean-image`` to supply a pre-cleaned PNG (no ClipDrop call).

Examples:

    python scripts/digitize_figure.py examples/figure_3.png output/

    python scripts/digitize_figure.py examples/ output/

    python scripts/digitize_figure.py data/multi_curve/figure_1/figure_1.png output/ \\
        --layout stacked --clean-image data/multi_curve/figure_1/figure_1_clean.png --debug

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
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotdigitizer_pipeline import digitize_figure_image  # noqa: E402
from xrd_digitization.axis_sidecar import (  # noqa: E402
    AxisSidecarError,
    extract_axis_sidecar_for_path,
    x_calibration_is_usable,
)
from xrd_digitization.cleaning_decision import decide_cleaning_strategy  # noqa: E402
from xrd_digitization.clipdrop_remove_text import (  # noqa: E402
    ClipdropError,
    clean_figure_full_image,
    clean_figure_preserve_axes,
)
from xrd_digitization.stacked_curves import digitize_stacked_figure  # noqa: E402
import cv2  # noqa: E402
import json  # noqa: E402

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
    layout: str = "single",
    clean_image: Path | None = None,
    debug: bool = False,
    curve_labels: Sequence[Any] | None = None,
    estimated_curve_count: int | None = None,
    extract_axes: bool = True,
    auto_cleaning: bool = True,
    require_usable_axes: bool = True,
    clipdrop_mode: str = "full",
) -> Path:
    """
    Digitize ``png_path`` into ``output_dir/<figure_id>/``.

    ``layout``:
      - ``single`` — existing single-curve PlotDigitizer path (forced one band)
      - ``stacked`` — stacked multi-curve path with shared x-axis
      - ``auto`` — try stacked band detection via digitize_figure_image defaults

    ``clean_image`` supplies a pre-cleaned PNG (skips ClipDrop). Useful for
    fixtures under ``data/multi_curve/``.

    When ``extract_axes`` is True, writes ``{figure_id}.axes.json`` from the
    original image before any ClipDrop cleaning. Unusable X calibration raises
    when ``require_usable_axes`` is True (no fabricated default ranges).

    When ``auto_cleaning`` is True, ClipDrop runs only if text materially
    interferes with colored-curve separation (``use_clipdrop`` becomes a soft
    hint). ``clipdrop_mode`` ``full`` (default) cleans the whole panel after
    axis extraction; ``preserve_axes`` uses the legacy inset path.

    Returns the figure output directory.
    """
    png_path = Path(png_path).resolve()
    if not png_path.is_file():
        raise FileNotFoundError(f"PNG not found: {png_path}")
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"Expected a .png file, got: {png_path}")

    layout = (layout or "single").strip().lower()
    if layout not in {"single", "stacked", "auto"}:
        raise ValueError(f"Unknown layout: {layout}")
    clipdrop_mode = (clipdrop_mode or "full").strip().lower()
    if clipdrop_mode not in {"full", "preserve_axes"}:
        raise ValueError(f"Unknown clipdrop_mode: {clipdrop_mode}")

    figure_id = figure_id_from_stem(png_path.stem)
    figure_dir = output_dir.resolve() / figure_id

    if figure_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {figure_dir} (pass --overwrite to replace)"
            )
        shutil.rmtree(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    original_copy = figure_dir / f"{figure_id}.png"
    if png_path.resolve() != original_copy.resolve():
        shutil.copy2(png_path, original_copy)
    elif png_path.name != original_copy.name:
        shutil.copy2(png_path, original_copy)

    axes_path: Path | None = figure_dir / f"{figure_id}.axes.json"
    sidecar = None
    if extract_axes:
        try:
            sidecar = extract_axis_sidecar_for_path(original_copy, output_path=axes_path)
        except AxisSidecarError as exc:
            LOGGER.warning("Axis sidecar extraction failed for %s: %s", figure_id, exc)
            axes_path = None
            if require_usable_axes:
                raise RuntimeError(f"calibration_failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Axis sidecar extraction failed for %s: %s", figure_id, exc)
            axes_path = None
            if require_usable_axes:
                raise RuntimeError(f"calibration_failed: {exc}") from exc
        if sidecar is not None and require_usable_axes:
            usable, reasons = x_calibration_is_usable(sidecar.calibration)
            if not usable:
                raise RuntimeError(
                    "calibration_failed: unusable X calibration on original panel "
                    f"({', '.join(reasons)}); expected >=3 OCR tick pairs "
                    f"(got method={sidecar.calibration.method} "
                    f"x=[{sidecar.calibration.x_min}, {sidecar.calibration.x_max}])"
                )
    else:
        axes_path = None

    original_bgr = cv2.imread(str(original_copy))
    if original_bgr is None:
        raise ValueError(f"Could not load image: {original_copy}")

    decision = decide_cleaning_strategy(
        original_bgr,
        triage_needs_clipdrop=bool(use_clipdrop),
        force_clean=bool(use_clipdrop) and not auto_cleaning,
        force_none=(not use_clipdrop) and not auto_cleaning,
    )
    if auto_cleaning:
        # Soft-merge triage hint: only force clean when interference already high
        # on colored/mixed curves. Gray/black XRD digitizes from the original.
        if use_clipdrop and decision.cleaning_strategy == "none":
            if decision.curve_color_mode in {"mixed", "unknown"}:
                decision = decide_cleaning_strategy(
                    original_bgr,
                    triage_needs_clipdrop=True,
                    force_clean=True,
                )

    digitize_path = original_copy
    cleaned_path: Path | None = None

    if clean_image is not None:
        clean_src = Path(clean_image).resolve()
        if not clean_src.is_file():
            raise FileNotFoundError(f"Clean image not found: {clean_src}")
        cleaned_path = figure_dir / f"{figure_id}_clean.png"
        if clean_src.resolve() != cleaned_path.resolve():
            shutil.copy2(clean_src, cleaned_path)
        digitize_path = cleaned_path
        decision.cleaning_strategy = "targeted"
        decision.cleaning_reason = "pre_cleaned_image_supplied"
        decision.needs_cleaning = True
        decision.digitization_source = "cleaned"
        LOGGER.info("Using pre-cleaned image (ClipDrop skipped): %s", clean_src.name)
    elif decision.needs_cleaning:
        cleaned_path = figure_dir / f"{figure_id}_clean.png"
        LOGGER.info(
            "Cleaning %s via %s (%s)",
            png_path.name,
            decision.cleaning_strategy,
            decision.cleaning_reason,
        )
        if clipdrop_mode == "preserve_axes" or decision.cleaning_strategy == "targeted":
            clean_figure_preserve_axes(original_copy, output_path=cleaned_path)
            decision.cleaning_strategy = "targeted"
        else:
            clean_figure_full_image(original_copy, output_path=cleaned_path)
            decision.cleaning_strategy = "clipdrop_full"
        digitize_path = cleaned_path
        decision.digitization_source = "cleaned"
    else:
        LOGGER.info(
            "Digitizing original (no ClipDrop): %s [%s]",
            png_path.name,
            decision.cleaning_reason,
        )
        decision.digitization_source = "original"

    meta_path = figure_dir / f"{figure_id}.cleaning.json"
    meta_path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")

    sidecar_arg = axes_path if axes_path is not None and Path(axes_path).is_file() else None

    if layout == "stacked":
        stacked = digitize_stacked_figure(
            original_copy if original_copy.is_file() else png_path,
            figure_dir,
            figure_id=figure_id,
            cleaned_image=digitize_path,
            estimated_curve_count=estimated_curve_count,
            curve_labels=curve_labels,
            debug=debug,
            axes_sidecar_path=sidecar_arg,
            prefer_colored_ink=(
                decision.curve_color_mode == "colored"
                and decision.digitization_source == "original"
            ),
            require_usable_axes=require_usable_axes,
        )
        succeeded = [c for c in stacked.curves if c.success]
        if not stacked.curves:
            raise RuntimeError(
                stacked.reason or "stacked digitization failed (no bands detected)"
            )
        if not succeeded:
            LOGGER.warning(
                "Stacked digitize produced 0 successful curves for %s (%s); "
                "writing outputs anyway",
                figure_id,
                stacked.reason or stacked.multi_curve_status,
            )
        else:
            LOGGER.info(
                "Wrote %d/%d stacked curve(s) under %s (status=%s)",
                len(succeeded),
                len(stacked.curves),
                figure_dir,
                stacked.multi_curve_status,
            )
        return figure_dir

    result = digitize_figure_image(
        digitize_path,
        figure_dir,
        figure_id=figure_id,
        force_single_band=(layout == "single"),
        axes_sidecar_path=sidecar_arg,
        require_axes_sidecar=require_usable_axes and sidecar_arg is not None,
    )
    if not result.bands:
        raise RuntimeError("digitize_figure_image returned no bands")

    succeeded_bands = [
        b for b in result.bands if b.success and b.csv_path and b.csv_path.is_file()
    ]
    if not succeeded_bands:
        errors = "; ".join(b.error or "unknown" for b in result.bands)
        raise RuntimeError(f"digitization failed: {errors}")

    LOGGER.info(
        "Wrote %d band(s) under %s",
        len(succeeded_bands),
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
        "--clean-image",
        type=Path,
        default=None,
        help="Pre-cleaned PNG to digitize (skips ClipDrop; for fixtures/debug)",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "single", "stacked"),
        default="single",
        help="Curve layout routing (default: single)",
    )
    parser.add_argument(
        "--estimated-curve-count",
        type=int,
        default=None,
        help="Soft hint for stacked band count validation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write stacked debug overlays under <figure_id>/debug/",
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

    if args.clipdrop and args.clean_image is not None:
        LOGGER.error("Pass only one of --clipdrop or --clean-image")
        return 1

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
                layout=args.layout,
                clean_image=args.clean_image,
                debug=args.debug,
                estimated_curve_count=args.estimated_curve_count,
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
