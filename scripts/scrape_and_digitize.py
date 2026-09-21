#!/usr/bin/env python3
"""
Scrape PDF figures with GROBID, triage with OpenAI, digitize via PlotDigitizer.

For each PDF: parse XRD figures only (pdf_parser ``xrd_figures_only`` /
``--xrd-figures-only``), ask OpenAI whether each figure is digitizable and
what its curve layout is (single / stacked / overlapping / multipanel), then
call digitize_one_figure. Single-curve and stacked shared-x figures use the
existing paths; multi-subplot figures are split into panel crops and digitized
independently. Overlapping multi-curve figures are skipped as unsupported.
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
    digitization_route,
    save_triage_result,
    triage_figure_image,
)
from pdf_parser import collect_pdf_paths, parse_pdf  # noqa: E402
from xrd_digitization.clipdrop_remove_text import ClipdropError  # noqa: E402
from xrd_digitization.extract_annotations import extract_plot_annotations  # noqa: E402
from xrd_digitization.split_figure_panels import (  # noqa: E402
    load_figure_text_context,
    split_figure_into_panels,
)

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
    debug_subdir: str | None = None,
) -> dict[str, Any]:
    """
    Copy digitize_one_figure artifacts into ``figures/<figure_id>/``.

    Top-level keeps only the deliverables:
      - ``<figure_id>.csv`` (primary / first curve)
      - ``<figure_id>_digitized.png`` (reconstructed preview when available)

    Everything else (per-curve CSVs/PNGs, stacked sidecars, band crops, clean
    PNG, debug overlays) goes under ``figures/<figure_id>/debug/`` (or
    ``debug/<debug_subdir>/`` for panel jobs).
    The original ``<figure_id>.png`` and triage JSON are already staged in
    ``figure_dir`` by the caller.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = figure_dir / "debug"
    if debug_subdir:
        debug_dir = debug_dir / debug_subdir
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    placed: dict[str, Any] = {"debug_dir": str(debug_dir)}

    def _copy_to_debug(src: Path, relative: Path) -> Path:
        dest = debug_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest

    # Primary CSV: combined stacked CSV, else first curve CSV, else single-curve CSV.
    stacked_csv = figure_work_dir / f"{figure_id}_stacked.csv"
    curve_csvs = sorted(figure_work_dir.glob(f"{figure_id}_curve_*.csv"))
    primary_csv: Path | None
    if stacked_csv.is_file():
        primary_csv = stacked_csv
    elif curve_csvs:
        primary_csv = curve_csvs[0]
    else:
        primary_csv = figure_work_dir / f"{figure_id}.csv"
        if not primary_csv.is_file():
            csv_candidates = sorted(figure_work_dir.glob("*.csv"))
            primary_csv = csv_candidates[0] if csv_candidates else None

    stacked_json = figure_work_dir / f"{figure_id}_stacked.json"
    if not stacked_json.is_file():
        # Backward-compatible legacy name.
        stacked_json = figure_work_dir / f"{figure_id}.stacked.json"
    has_stacked_json = stacked_json.is_file()

    if primary_csv is not None and primary_csv.is_file():
        dest_csv = figure_dir / f"{figure_id}.csv"
        if primary_csv.resolve() != dest_csv.resolve():
            shutil.copy2(primary_csv, dest_csv)
        placed["csv"] = str(dest_csv)
    elif not has_stacked_json:
        raise FileNotFoundError(f"No CSV produced under {figure_work_dir}")

    # Prefer clean digitized_stacked.png, then reconstructed overlay, then PD plot.
    digitized_stacked = figure_work_dir / "digitized_stacked.png"
    reconstructed = figure_work_dir / "reconstructed_overlay.png"
    if not reconstructed.is_file():
        reconstructed = figure_work_dir / "debug" / "reconstructed_overlay.png"
    digitized = figure_work_dir / f"{figure_id}_digitized.png"
    if not digitized.is_file():
        dig_candidates = sorted(figure_work_dir.glob("*_digitized.png"))
        digitized = dig_candidates[0] if dig_candidates else digitized
    if digitized_stacked.is_file():
        primary_dig_src = digitized_stacked
    elif reconstructed.is_file():
        primary_dig_src = reconstructed
    else:
        primary_dig_src = digitized
    if primary_dig_src.is_file():
        dest_dig = figure_dir / f"{figure_id}_digitized.png"
        shutil.copy2(primary_dig_src, dest_dig)
        placed["digitized_png"] = str(dest_dig)

    # Promote cleaned PNG and axes sidecar to figure dir (esp. panel stems).
    clean_src = figure_work_dir / f"{figure_id}_clean.png"
    if clean_src.is_file():
        dest_clean = figure_dir / f"{figure_id}_clean.png"
        shutil.copy2(clean_src, dest_clean)
        placed["clean_png"] = str(dest_clean)
    axes_src = figure_work_dir / f"{figure_id}.axes.json"
    if axes_src.is_file():
        dest_axes = figure_dir / f"{figure_id}.axes.json"
        shutil.copy2(axes_src, dest_axes)
        placed["axes_json"] = str(dest_axes)
    cleaning_src = figure_work_dir / f"{figure_id}.cleaning.json"
    if cleaning_src.is_file():
        dest_cleaning = figure_dir / f"{figure_id}.cleaning.json"
        shutil.copy2(cleaning_src, dest_cleaning)
        placed["cleaning_json"] = str(dest_cleaning)

    # Skip originals already staged at top-level by the caller.
    skip_top_level = {
        f"{figure_id}.png",
        f"{figure_id}.triage.json",
        f"{figure_id}_clean.png",
        f"{figure_id}.axes.json",
        f"{figure_id}.cleaning.json",
        f"{figure_id}.csv",
        f"{figure_id}_digitized.png",
    }

    for src in sorted(figure_work_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.name in skip_top_level and src.parent.resolve() == figure_work_dir.resolve():
            continue
        try:
            relative = src.relative_to(figure_work_dir)
        except ValueError:
            relative = Path(src.name)
        # Flatten work/debug/* into figure_dir/debug/*
        if relative.parts and relative.parts[0] == "debug":
            relative = Path(*relative.parts[1:]) if len(relative.parts) > 1 else Path(src.name)
        dest = _copy_to_debug(src, relative)
        name = src.name
        if name in {f"{figure_id}_stacked.json", f"{figure_id}.stacked.json"}:
            placed["stacked_json"] = str(dest)
        elif name == f"{figure_id}_stacked.csv":
            placed["stacked_csv"] = str(dest)
        elif name == f"{figure_id}_stacked.xy":
            placed["stacked_xy"] = str(dest)
        elif name == "digitized_stacked.png":
            placed["digitized_stacked_png"] = str(dest)
        elif name == "reconstructed_overlay.png":
            placed["reconstructed_overlay"] = str(dest)
        elif name == f"{figure_id}_clean.png":
            placed.setdefault("clean_png", str(dest))
        elif name.startswith(f"{figure_id}_curve_") and name.endswith(".csv"):
            placed.setdefault("curve_csvs", []).append(str(dest))
        elif name.startswith(f"{figure_id}_curve_") and name.endswith("_digitized.png"):
            placed.setdefault("curve_digitized_pngs", []).append(str(dest))

    return placed


def _triage_entry_dict(triage: Any, triage_path: Path, route: str) -> dict[str, Any]:
    return {
        "digitizable": triage.digitizable,
        "curve_count": triage.curve_count,
        "curve_layout": triage.curve_layout,
        "shared_x_axis": triage.shared_x_axis,
        "vertically_offset": triage.vertically_offset,
        "curves_cross": triage.curves_cross,
        "needs_clipdrop": triage.needs_clipdrop,
        "preprocessing_reasons": list(getattr(triage, "preprocessing_reasons", []) or []),
        "curve_labels": [
            {"text": label.text, "vertical_order": label.vertical_order}
            for label in triage.curve_labels
        ],
        "reason": triage.reason,
        "path": str(triage_path),
        "route": route,
    }


def _digitize_routed_figure(
    png_path: Path,
    *,
    figure_id: str,
    figure_dir: Path,
    triage: Any,
    route: str,
    overwrite: bool,
    debug_subdir: str | None = None,
) -> dict[str, Any]:
    # Triage needs_clipdrop is a soft hint; digitize_one_figure decides via
    # color/text interference whether ClipDrop actually runs.
    use_clipdrop_hint = bool(triage.needs_clipdrop)
    layout = "stacked" if route == "stacked" else "single"
    with tempfile.TemporaryDirectory(prefix=f"dig_{figure_id}_") as tmp:
        work_root = Path(tmp)
        figure_work_dir = digitize_one_figure(
            png_path,
            work_root,
            use_clipdrop=use_clipdrop_hint,
            overwrite=overwrite,
            layout=layout,
            curve_labels=triage.curve_labels,
            estimated_curve_count=triage.curve_count,
            debug=True,
            auto_cleaning=True,
            require_usable_axes=True,
            clipdrop_mode="full",
        )
        placed = rearrange_digitize_outputs(
            figure_work_dir,
            figure_id=figure_id,
            figure_dir=figure_dir,
            debug_subdir=debug_subdir,
        )
    cleaning_meta = None
    cleaning_path = figure_dir / f"{figure_id}.cleaning.json"
    if cleaning_path.is_file():
        try:
            cleaning_meta = json.loads(cleaning_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cleaning_meta = None
    return {
        "use_clipdrop": bool(cleaning_meta.get("needs_cleaning"))
        if isinstance(cleaning_meta, dict)
        else use_clipdrop_hint,
        "layout": layout,
        "outputs": placed,
        "cleaning": cleaning_meta,
    }


def process_multipanel_figure(
    staged_png: Path,
    *,
    figure_id: str,
    figure_dir: Path,
    paper_dir: Path,
    figure_triage: Any,
    model: str | None,
    overwrite: bool,
    http_post: Any | None = None,
) -> dict[str, Any]:
    """Split a multipanel figure and digitize each panel independently."""
    caption, body_text = load_figure_text_context(paper_dir, figure_id=figure_id)
    split = split_figure_into_panels(
        staged_png,
        figure_id=figure_id,
        figure_caption=caption,
        figure_body_text=body_text,
        output_dir=figure_dir,
        model=model,
        http_post=http_post,
    )

    panel_entries: list[dict[str, Any]] = []
    for record in split.panels:
        label = record.panel.label or str(record.panel.index)
        panel_id = f"{figure_id}_{label}"
        panel_entry: dict[str, Any] = {
            "panel_label": label,
            "panel_id": panel_id,
            "panel_png": str(record.image_path),
            "bbox": list(record.panel.bbox),
            "plot_type": record.panel.plot_type,
            "digitizable_after_split": record.panel.digitizable_after_split,
            "figure_caption": record.figure_caption,
            "panel_body_context": record.panel_body_context,
            "status": "pending",
        }
        if not record.panel.digitizable_after_split:
            panel_entry["status"] = "skipped_not_digitizable"
            panel_entries.append(panel_entry)
            continue

        # Optional metadata — never blocks digitization.
        try:
            ann_path = figure_dir / f"{panel_id}.annotations.json"
            anns = extract_plot_annotations(record.image_path, output_path=ann_path)
            if anns:
                panel_entry["annotations_json"] = str(ann_path)
                panel_entry["annotation_count"] = len(anns)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Annotation extraction skipped for %s: %s", panel_id, exc)

        try:
            panel_triage = triage_figure_image(
                record.image_path,
                model=model,
                http_post=http_post,
            )
        except Exception as exc:
            LOGGER.exception("Panel triage failed for %s", panel_id)
            panel_entry["status"] = "triage_failed"
            panel_entry["error"] = str(exc)
            panel_entries.append(panel_entry)
            continue

        triage_path = save_triage_result(
            panel_triage,
            figure_dir / f"{panel_id}.triage.json",
        )
        panel_route = digitization_route(panel_triage)
        # Multipanel crops should never re-enter multipanel routing.
        if panel_route == "multipanel":
            if panel_triage.curve_count >= 2 and panel_triage.shared_x_axis:
                panel_route = "stacked"
            elif panel_triage.curve_count == 1:
                panel_route = "single"
            else:
                panel_route = "unsupported"
        panel_entry["triage"] = _triage_entry_dict(panel_triage, triage_path, panel_route)

        if panel_route == "reject":
            panel_entry["status"] = "skipped_not_digitizable"
            panel_entries.append(panel_entry)
            continue
        if panel_route == "unsupported":
            panel_entry["status"] = f"skipped_unsupported_{panel_triage.curve_layout}"
            panel_entries.append(panel_entry)
            continue

        try:
            dig = _digitize_routed_figure(
                record.image_path,
                figure_id=panel_id,
                figure_dir=figure_dir,
                triage=panel_triage,
                route=panel_route,
                overwrite=overwrite,
                debug_subdir=label,
            )
        except (ClipdropError, FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
            LOGGER.error("Digitization failed for panel %s: %s", panel_id, exc)
            panel_entry["status"] = "digitize_failed"
            panel_entry["error"] = str(exc)
            panel_entries.append(panel_entry)
            continue
        except Exception as exc:
            LOGGER.exception("Digitization failed for panel %s", panel_id)
            panel_entry["status"] = "digitize_failed"
            panel_entry["error"] = str(exc)
            panel_entries.append(panel_entry)
            continue

        panel_entry["status"] = "digitized"
        panel_entry["use_clipdrop"] = dig["use_clipdrop"]
        panel_entry["layout"] = dig["layout"]
        panel_entry["outputs"] = dig["outputs"]
        if dig.get("cleaning"):
            panel_entry["cleaning"] = dig["cleaning"]
        panel_entries.append(panel_entry)
        LOGGER.info(
            "Digitized panel %s (route=%s) → %s",
            panel_id,
            panel_route,
            (dig["outputs"] or {}).get("csv") or (dig["outputs"] or {}).get("stacked_json"),
        )

    digitized_n = sum(1 for p in panel_entries if p.get("status") == "digitized")
    failed_n = sum(1 for p in panel_entries if str(p.get("status", "")).endswith("_failed"))
    if digitized_n == 0 and failed_n > 0:
        status = "digitize_failed"
    elif digitized_n == 0:
        status = "skipped_not_digitizable"
    elif failed_n > 0 or digitized_n < len(panel_entries):
        status = "digitized_partial"
    else:
        status = "digitized"

    return {
        "status": status,
        "preprocessing_reasons": list(
            getattr(figure_triage, "preprocessing_reasons", []) or ["multiple_panels"]
        ),
        "panels_json": str(split.panels_json_path) if split.panels_json_path else None,
        "panel_split_method": split.method,
        "panels": panel_entries,
        "figure_caption": caption,
    }


def process_figure(
    png_path: Path,
    *,
    figures_dir: Path,
    model: str | None,
    overwrite: bool,
    http_post: Any | None = None,
    paper_dir: Path | None = None,
) -> dict[str, Any]:
    """Triage one figure; digitize via digitize_one_figure when eligible."""
    figure_id, figure_dir, staged_png = stage_figure_directory(png_path, figures_dir)
    entry: dict[str, Any] = {
        "figure_id": figure_id,
        "figure_dir": str(figure_dir),
        "source_png": str(staged_png),
        "status": "pending",
    }
    resolved_paper_dir = Path(paper_dir) if paper_dir is not None else figures_dir.parent

    try:
        triage = triage_figure_image(staged_png, model=model, http_post=http_post)
    except Exception as exc:
        LOGGER.exception("Triage failed for %s", staged_png.name)
        entry["status"] = "triage_failed"
        entry["error"] = str(exc)
        return entry

    triage_path = save_triage_result(triage, figure_dir / f"{figure_id}.triage.json")
    route = digitization_route(triage)
    entry["triage"] = _triage_entry_dict(triage, triage_path, route)

    if route == "reject":
        LOGGER.info(
            "Skipping %s (not digitizable): %s",
            figure_id,
            triage.reason or triage.curve_layout,
        )
        entry["status"] = "skipped_not_digitizable"
        return entry

    if route == "unsupported":
        LOGGER.info(
            "Skipping %s (unsupported layout %s): %s",
            figure_id,
            triage.curve_layout,
            triage.reason or "",
        )
        entry["status"] = f"skipped_unsupported_{triage.curve_layout}"
        return entry

    if route == "multipanel":
        LOGGER.info(
            "Multipanel figure %s → split and digitize panels (%s)",
            figure_id,
            ", ".join(triage.preprocessing_reasons) or "multiple_panels",
        )
        entry["status"] = "requires_preprocessing"
        try:
            multi = process_multipanel_figure(
                staged_png,
                figure_id=figure_id,
                figure_dir=figure_dir,
                paper_dir=resolved_paper_dir,
                figure_triage=triage,
                model=model,
                overwrite=overwrite,
                http_post=http_post,
            )
        except Exception as exc:
            LOGGER.exception("Multipanel processing failed for %s", figure_id)
            entry["status"] = "digitize_failed"
            entry["error"] = str(exc)
            return entry
        entry.update(multi)
        return entry

    try:
        dig = _digitize_routed_figure(
            staged_png,
            figure_id=figure_id,
            figure_dir=figure_dir,
            triage=triage,
            route=route,
            overwrite=overwrite,
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
    entry["use_clipdrop"] = dig["use_clipdrop"]
    entry["outputs"] = dig["outputs"]
    LOGGER.info(
        "Digitized %s (route=%s clipdrop=%s) → %s",
        figure_id,
        route,
        dig["use_clipdrop"],
        dig["outputs"].get("csv") or dig["outputs"].get("stacked_json"),
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
                paper_dir=paper_dir,
            )
        )

    def _is_digitized(status: str) -> bool:
        return status in {"digitized", "digitized_partial"}

    manifest = {
        "source_pdf": str(pdf_path.resolve()),
        "output_directory": str(paper_dir.resolve()),
        "figures_total": len(pngs),
        "digitized": sum(1 for e in figure_entries if _is_digitized(str(e.get("status")))),
        "skipped": sum(
            1
            for e in figure_entries
            if str(e.get("status", "")).startswith("skipped_")
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
