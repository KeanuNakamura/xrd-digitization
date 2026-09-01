#!/usr/bin/env python3
"""
Scrape PDF figures with GROBID, triage with OpenAI, digitize via PlotDigitizer.

For each PDF: parse XRD figures only (pdf_parser ``xrd_figures_only`` /
``--xrd-figures-only``), ask OpenAI whether each figure is a single-curve plot
(digitizable) and whether ClipDrop text removal is needed, then call
digitize_one_figure (same path as scripts/digitize_figure.py, including
--clipdrop behavior when requested). Non-digitizable figures are skipped.

Per-figure outputs land in ``figures/<figure_id>/`` (original PNG, triage JSON,
digitized CSV/PNG when available). Paper-level GROBID artifacts stay under
``extra/``.

Examples:

    export OPENAI_API_KEY=...
    export CLIPDROP_API_KEY=...   # only if triage requests ClipDrop

    python scripts/scrape_and_digitize.py paper.pdf output/

    python scripts/scrape_and_digitize.py pdf_files/ output/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(LEGACY), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from digitize_figure import digitize_one_figure, figure_id_from_stem  # noqa: E402
from figure_triage import (  # noqa: E402
    is_digitizable,
    save_triage_result,
    triage_figure_image,
)
from pdf_parser import collect_pdf_paths, parse_pdf  # noqa: E402
from xrd_digitization.clipdrop_remove_text import ClipdropError  # noqa: E402

LOGGER = logging.getLogger(__name__)


def paper_output_dir(output_root: Path, pdf_path: Path) -> Path:
    """``output_root/<pdf_stem>/`` for both single-PDF and directory inputs."""
    return output_root.resolve() / pdf_path.stem


def collect_figure_pngs(figures_dir: Path) -> list[Path]:
    """Sorted source PNGs directly under ``figures/`` (flat extract from parse_pdf)."""
    if not figures_dir.is_dir():
        return []
    return sorted(
        p
        for p in figures_dir.glob("*.png")
        if p.is_file() and not p.stem.lower().endswith(("_clean", "_digitized"))
    )


def stage_figure_directory(png_path: Path, figures_dir: Path) -> tuple[str, Path, Path]:
    """
    Move a flat extracted PNG into ``figures/<figure_id>/<figure_id>.png``.

    Returns ``(figure_id, figure_dir, staged_png)``.
    """
    figure_id = figure_id_from_stem(png_path.stem)
    figure_dir = figures_dir / figure_id
    figure_dir.mkdir(parents=True, exist_ok=True)
    staged_png = figure_dir / f"{figure_id}.png"
    if png_path.resolve() != staged_png.resolve():
        if staged_png.exists():
            staged_png.unlink()
        shutil.move(str(png_path), str(staged_png))
    return figure_id, figure_dir, staged_png


def rearrange_digitize_outputs(
    figure_work_dir: Path,
    *,
    figure_id: str,
    figure_dir: Path,
) -> dict[str, str]:
    """
    Copy digitize_one_figure artifacts into ``figures/<figure_id>/``.

    - primary CSV → ``<figure_id>.csv``
    - ``*_digitized.png`` / ``*_clean.png`` when present
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    placed: dict[str, str] = {}

    primary_csv = figure_work_dir / f"{figure_id}.csv"
    if not primary_csv.is_file():
        # Band stems may differ slightly; take the first successful CSV.
        csv_candidates = sorted(figure_work_dir.glob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV produced under {figure_work_dir}")
        primary_csv = csv_candidates[0]

    dest_csv = figure_dir / f"{figure_id}.csv"
    shutil.copy2(primary_csv, dest_csv)
    placed["csv"] = str(dest_csv)

    digitized = figure_work_dir / f"{figure_id}_digitized.png"
    if not digitized.is_file():
        dig_candidates = sorted(figure_work_dir.glob("*_digitized.png"))
        digitized = dig_candidates[0] if dig_candidates else digitized
    if digitized.is_file():
        dest_dig = figure_dir / f"{figure_id}_digitized.png"
        shutil.copy2(digitized, dest_dig)
        placed["digitized_png"] = str(dest_dig)

    clean = figure_work_dir / f"{figure_id}_clean.png"
    if clean.is_file():
        dest_clean = figure_dir / f"{figure_id}_clean.png"
        shutil.copy2(clean, dest_clean)
        placed["clean_png"] = str(dest_clean)

    return placed


def process_figure(
    png_path: Path,
    *,
    figures_dir: Path,
    model: str | None,
    overwrite: bool,
    http_post: Any | None = None,
) -> dict[str, Any]:
    """Triage one figure; digitize via digitize_one_figure when eligible."""
    figure_id, figure_dir, staged_png = stage_figure_directory(png_path, figures_dir)
    entry: dict[str, Any] = {
        "figure_id": figure_id,
        "figure_dir": str(figure_dir),
        "source_png": str(staged_png),
        "status": "pending",
    }

    try:
        triage = triage_figure_image(staged_png, model=model, http_post=http_post)
    except Exception as exc:
        LOGGER.exception("Triage failed for %s", staged_png.name)
        entry["status"] = "triage_failed"
        entry["error"] = str(exc)
        return entry

    triage_path = save_triage_result(triage, figure_dir / f"{figure_id}.triage.json")
    entry["triage"] = {
        "digitizable": triage.digitizable,
        "curve_count": triage.curve_count,
        "curve_layout": triage.curve_layout,
        "needs_clipdrop": triage.needs_clipdrop,
        "reason": triage.reason,
        "path": str(triage_path),
    }

    if not is_digitizable(triage):
        LOGGER.info(
            "Skipping %s (not digitizable): %s",
            figure_id,
            triage.reason or triage.curve_layout,
        )
        entry["status"] = "skipped_not_digitizable"
        return entry

    use_clipdrop = bool(triage.needs_clipdrop)
    entry["use_clipdrop"] = use_clipdrop

    with tempfile.TemporaryDirectory(prefix=f"dig_{figure_id}_") as tmp:
        work_root = Path(tmp)
        try:
            figure_work_dir = digitize_one_figure(
                staged_png,
                work_root,
                use_clipdrop=use_clipdrop,
                overwrite=overwrite,
            )
            placed = rearrange_digitize_outputs(
                figure_work_dir,
                figure_id=figure_id,
                figure_dir=figure_dir,
            )
        except (ClipdropError, FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
            LOGGER.error("Digitization failed for %s: %s", figure_id, exc)
            entry["status"] = "digitize_failed"
            entry["error"] = str(exc)
            return entry
        except Exception as exc:
            LOGGER.exception("Digitization failed for %s", figure_id)
            entry["status"] = "digitize_failed"
            entry["error"] = str(exc)
            return entry

    entry["status"] = "digitized"
    entry["outputs"] = placed
    LOGGER.info(
        "Digitized %s (clipdrop=%s) → %s",
        figure_id,
        use_clipdrop,
        placed.get("csv"),
    )
    return entry


def process_pdf(
    pdf_path: Path,
    output_root: Path,
    *,
    grobid_url: str,
    figure_dpi: int,
    model: str | None,
    overwrite: bool,
    http_post: Any | None = None,
) -> dict[str, Any]:
    """Scrape one PDF then triage/digitize its figures."""
    paper_dir = paper_output_dir(output_root, pdf_path)
    if paper_dir.exists() and overwrite:
        LOGGER.info("Removing existing output: %s", paper_dir)
        shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Scraping %s → %s", pdf_path.name, paper_dir)
    parse_pdf(
        pdf_path=pdf_path,
        output_directory=paper_dir,
        grobid_url=grobid_url,
        extract_figures=True,
        figure_dpi=figure_dpi,
        xrd_figures_only=True,
    )

    figures_dir = paper_dir / "figures"
    pngs = collect_figure_pngs(figures_dir)
    figure_entries: list[dict[str, Any]] = []

    for png_path in pngs:
        LOGGER.info("=== %s ===", png_path.name)
        figure_entries.append(
            process_figure(
                png_path,
                figures_dir=figures_dir,
                model=model,
                overwrite=True,
                http_post=http_post,
            )
        )

    manifest = {
        "source_pdf": str(pdf_path.resolve()),
        "output_directory": str(paper_dir.resolve()),
        "figures_total": len(pngs),
        "digitized": sum(1 for e in figure_entries if e.get("status") == "digitized"),
        "skipped": sum(
            1 for e in figure_entries if e.get("status") == "skipped_not_digitizable"
        ),
        "failed": sum(
            1
            for e in figure_entries
            if str(e.get("status", "")).endswith("_failed")
        ),
        "figures": figure_entries,
    }
    manifest_path = paper_dir / "digitization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest: %s", manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a PDF file or a directory containing PDFs (searched recursively)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output root directory (each PDF writes to <output_dir>/<pdf_stem>/)",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="Base URL of the GROBID server",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI used when rendering cropped figure images",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI vision model for triage (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing <output_dir>/<pdf_stem>/ directory",
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
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        pdf_paths = collect_pdf_paths(args.input_path.resolve())
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    failures: list[tuple[Path, str]] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        LOGGER.info("Processing PDF %d/%d: %s", index, len(pdf_paths), pdf_path)
        try:
            summary = process_pdf(
                pdf_path,
                output_root,
                grobid_url=args.grobid_url,
                figure_dpi=args.figure_dpi,
                model=args.model,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            LOGGER.exception("Failed on %s", pdf_path)
            failures.append((pdf_path, str(exc)))
            continue
        summaries.append(summary)
        print(
            f"{pdf_path.name}: digitized={summary['digitized']} "
            f"skipped={summary['skipped']} failed={summary['failed']} "
            f"→ {summary['output_directory']}"
        )

    if len(pdf_paths) > 1:
        LOGGER.info(
            "Batch done: %d succeeded, %d failed (of %d PDFs)",
            len(summaries),
            len(failures),
            len(pdf_paths),
        )
        for path, err in failures:
            LOGGER.error("  failed %s: %s", path.name, err)

    if failures and not summaries:
        return 1
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
