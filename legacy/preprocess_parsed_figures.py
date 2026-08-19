"""
Batch-preprocess figures from GROBID / pdf_parser parsed outputs.

Given a parsed-paper directory (or a root containing many of them), resolve each
figure crop from ``*.parsed.json``, call ``extract_figure``, and write
originals + text masks under ``figures_without_text/<stem>/``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from parse_figures import select_figure_crop_coords
from pdf_figure_structure import ExtractMode, extract_figure

LOGGER = logging.getLogger(__name__)

DEFAULT_DPI = 300
DEFAULT_MODE: ExtractMode = "all"
SKIP_DIR_NAMES = frozenset({"figures_without_text", "__pycache__"})


def find_parsed_json(paper_dir: Path) -> Path | None:
    """Return the primary ``*.parsed.json`` for a paper directory, if any."""
    paper_dir = Path(paper_dir)
    candidates = sorted((paper_dir / "extra").glob("*.parsed.json"))
    if not candidates:
        candidates = sorted(paper_dir.glob("*.parsed.json"))
    if not candidates:
        return None
    preferred = paper_dir / "extra" / f"{paper_dir.name}.parsed.json"
    if preferred in candidates:
        return preferred
    return candidates[0]


def is_paper_dir(path: Path) -> bool:
    """True when ``path`` looks like one parsed-paper output directory."""
    return find_parsed_json(path) is not None


def discover_paper_dirs(parsed_root: Path) -> list[Path]:
    """
    Discover paper directories under ``parsed_root``.

    Accepts either:
      - a single paper directory containing ``extra/*.parsed.json``
      - a root whose immediate child directories are papers
    """
    parsed_root = Path(parsed_root).resolve()
    if not parsed_root.exists():
        raise FileNotFoundError(f"Parsed output path not found: {parsed_root}")
    if parsed_root.is_file():
        if parsed_root.name.endswith(".parsed.json"):
            # ``.../extra/<stem>.parsed.json`` → paper dir is parent of extra
            if parsed_root.parent.name == "extra":
                return [parsed_root.parent.parent]
            return [parsed_root.parent]
        raise ValueError(f"Expected a directory or *.parsed.json, got {parsed_root}")

    if is_paper_dir(parsed_root):
        return [parsed_root]

    papers = [
        child
        for child in sorted(parsed_root.iterdir())
        if child.is_dir()
        and child.name not in SKIP_DIR_NAMES
        and not child.name.startswith(".")
        and is_paper_dir(child)
    ]
    return papers


def default_output_root(parsed_input: Path) -> Path:
    """
    Default output root: ``<batch_root>/figures_without_text``.

    For a single paper dir, batch root is the parent. For a multi-paper root,
    batch root is that root.
    """
    parsed_input = Path(parsed_input).resolve()
    if parsed_input.is_file():
        papers = discover_paper_dirs(parsed_input)
        batch_root = papers[0].parent if papers else parsed_input.parent
    elif is_paper_dir(parsed_input):
        batch_root = parsed_input.parent
    else:
        batch_root = parsed_input
    return batch_root / "figures_without_text"


def resolve_pdf_path(
    document: dict[str, Any],
    paper_dir: Path,
    *,
    pdf_dir: Path | None = None,
    pdf_path: Path | None = None,
) -> Path | None:
    """Resolve the source PDF for a parsed document."""
    if pdf_path is not None:
        candidate = Path(pdf_path)
        return candidate if candidate.is_file() else None

    source = document.get("source_pdf")
    if source:
        candidate = Path(source)
        if candidate.is_file():
            return candidate

    stem = paper_dir.name
    search_dirs: list[Path] = []
    if pdf_dir is not None:
        search_dirs.append(Path(pdf_dir))
    search_dirs.extend(
        [
            paper_dir,
            paper_dir.parent,
            paper_dir.parent.parent / "pdf_files" / paper_dir.parent.name,
            Path.cwd() / "pdf_files" / "sample_pdfs",
        ]
    )
    for directory in search_dirs:
        candidate = Path(directory) / f"{stem}.pdf"
        if candidate.is_file():
            return candidate
    return None


def _page_for_stem(result: dict[str, Any], out_stem: str, paths: dict[str, Any]) -> dict[str, Any]:
    for page in result.get("pages") or []:
        original = paths.get("original") or ""
        if original.endswith(f"{out_stem}_original.png"):
            return page
    pages = result.get("pages") or []
    return pages[0] if pages else {}


def preprocess_paper(
    paper_dir: Path,
    output_dir: Path,
    *,
    pdf_dir: Path | None = None,
    pdf_path: Path | None = None,
    mode: ExtractMode = DEFAULT_MODE,
    dpi: int = DEFAULT_DPI,
    zoom: float | None = None,
    xrd_only: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
    figure_ids: Sequence[str] | None = None,
    removal_method: str = "local_background",
) -> dict[str, Any]:
    """
    Preprocess all usable figures for one parsed paper.

    Writes ``extract_figure`` artifacts into ``output_dir`` and a
    ``manifest.json`` summarizing results.
    """
    paper_dir = Path(paper_dir).resolve()
    output_dir = Path(output_dir)
    parsed_path = find_parsed_json(paper_dir)
    if parsed_path is None:
        raise FileNotFoundError(f"No *.parsed.json under {paper_dir}")

    if skip_existing and not overwrite and (output_dir / "manifest.json").is_file():
        LOGGER.info("Skipping %s: manifest already exists", paper_dir.name)
        return {
            "pdf": None,
            "paper_dir": str(paper_dir),
            "parsed_json": str(parsed_path),
            "output_directory": str(output_dir),
            "status": "skipped_existing",
            "counts": {
                "figures_in_parse": 0,
                "ok": 0,
                "text_free": 0,
                "skipped_no_coords": 0,
                "skipped_filter": 0,
                "errors": 0,
            },
            "figures": [],
        }

    document = json.loads(parsed_path.read_text(encoding="utf-8"))
    resolved_pdf = resolve_pdf_path(
        document,
        paper_dir,
        pdf_dir=pdf_dir,
        pdf_path=pdf_path,
    )
    if resolved_pdf is None:
        raise FileNotFoundError(
            f"Could not resolve PDF for {paper_dir.name}. "
            "Set source_pdf in the parse or pass --pdf-dir / --pdf."
        )

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = list(document.get("figures") or [])
    wanted = set(figure_ids) if figure_ids else None
    manifest_figures: list[dict[str, Any]] = []
    n_ok = n_text_free = n_skip = n_filter = n_err = 0

    LOGGER.info("=== %s (%d figures) ===", paper_dir.name, len(figures))

    for figure in figures:
        figure_id = str(figure.get("figure_id") or "unknown_figure")
        if wanted is not None and figure_id not in wanted:
            continue
        if xrd_only and not figure.get("is_likely_xrd"):
            manifest_figures.append(
                {
                    "figure_id": figure_id,
                    "status": "skipped_not_xrd",
                }
            )
            n_filter += 1
            continue

        crop_coords = select_figure_crop_coords(
            figure.get("coords"),
            figure.get("graphic_coords"),
            pdf_path=resolved_pdf,
        )
        if not crop_coords:
            LOGGER.warning("Skipping %s/%s: no usable coords", paper_dir.name, figure_id)
            manifest_figures.append(
                {
                    "figure_id": figure_id,
                    "status": "skipped_no_coords",
                    "coords": figure.get("coords"),
                    "graphic_coords": figure.get("graphic_coords"),
                }
            )
            n_skip += 1
            continue

        try:
            result = extract_figure(
                pdf_path=resolved_pdf,
                figure_coordinates=crop_coords,
                output_directory=output_dir,
                figure_id=figure_id,
                mode=mode,
                dpi=None if zoom is not None else dpi,
                zoom=zoom,
                removal_method=removal_method,  # type: ignore[arg-type]
            )
        except Exception as exc:
            LOGGER.exception("Failed %s/%s: %s", paper_dir.name, figure_id, exc)
            manifest_figures.append(
                {
                    "figure_id": figure_id,
                    "status": "error",
                    "error": str(exc),
                    "crop_coords": crop_coords,
                }
            )
            n_err += 1
            continue

        n_ok += 1
        for out_stem, paths in (result.get("output_paths") or {}).items():
            page = _page_for_stem(result, out_stem, paths)
            tfr = page.get("text_free_render") or {}
            if paths.get("without_pdf_text") or tfr.get("created"):
                n_text_free += 1
            classification = page.get("classification")
            if not isinstance(classification, dict):
                diagnostics = page.get("diagnostics") or {}
                classification = (
                    diagnostics.get("classification")
                    if isinstance(diagnostics, dict)
                    else None
                )
            manifest_figures.append(
                {
                    "figure_id": figure_id,
                    "output_stem": out_stem,
                    "status": "ok",
                    "crop_coords": crop_coords,
                    "paths": paths,
                    "text_free_render": tfr,
                    "page_number": page.get("page_number"),
                    "figure_rect": page.get("figure_rect"),
                    "classification": classification,
                }
            )

    manifest = {
        "pdf": str(resolved_pdf),
        "paper_dir": str(paper_dir),
        "parsed_json": str(parsed_path),
        "output_directory": str(output_dir),
        "dpi": dpi if zoom is None else None,
        "zoom": zoom,
        "mode": mode,
        "status": "ok",
        "counts": {
            "figures_in_parse": len(figures),
            "ok": n_ok,
            "text_free": n_text_free,
            "skipped_no_coords": n_skip,
            "skipped_filter": n_filter,
            "errors": n_err,
        },
        "figures": manifest_figures,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "%s done: ok=%d text_free=%d skipped=%d filtered=%d errors=%d",
        paper_dir.name,
        n_ok,
        n_text_free,
        n_skip,
        n_filter,
        n_err,
    )
    return manifest


def preprocess_parsed_outputs(
    parsed_input: Path,
    *,
    output_root: Path | None = None,
    pdf_dir: Path | None = None,
    mode: ExtractMode = DEFAULT_MODE,
    dpi: int = DEFAULT_DPI,
    zoom: float | None = None,
    xrd_only: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
    figure_ids: Sequence[str] | None = None,
    paper_names: Iterable[str] | None = None,
    removal_method: str = "local_background",
) -> dict[str, Any]:
    """
    Preprocess figures for one paper or every paper under a parsed-output root.

    Parameters
    ----------
    parsed_input
        A paper directory, a root of paper directories, or a ``*.parsed.json`` path.
    output_root
        Directory that will contain ``<stem>/`` subfolders. Defaults to
        ``<batch_root>/figures_without_text``.
    """
    parsed_input = Path(parsed_input)
    papers = discover_paper_dirs(parsed_input)
    if paper_names is not None:
        wanted_papers = set(paper_names)
        papers = [p for p in papers if p.name in wanted_papers]

    if not papers:
        raise FileNotFoundError(f"No parsed paper directories found under {parsed_input}")

    out_root = Path(output_root) if output_root is not None else default_output_root(parsed_input)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_papers: list[dict[str, Any]] = []
    for paper_dir in papers:
        try:
            manifest = preprocess_paper(
                paper_dir,
                out_root / paper_dir.name,
                pdf_dir=pdf_dir,
                mode=mode,
                dpi=dpi,
                zoom=zoom,
                xrd_only=xrd_only,
                skip_existing=skip_existing,
                overwrite=overwrite,
                figure_ids=figure_ids,
                removal_method=removal_method,
            )
            summary_papers.append(
                {
                    "paper": paper_dir.name,
                    "status": manifest.get("status", "ok"),
                    **(manifest.get("counts") or {}),
                    "output_directory": manifest.get("output_directory"),
                }
            )
        except Exception as exc:
            LOGGER.exception("Paper failed %s: %s", paper_dir.name, exc)
            summary_papers.append(
                {
                    "paper": paper_dir.name,
                    "status": "error",
                    "error": str(exc),
                }
            )

    summary = {
        "parsed_input": str(Path(parsed_input).resolve()),
        "output_root": str(out_root.resolve()),
        "mode": mode,
        "dpi": dpi if zoom is None else None,
        "zoom": zoom,
        "papers": summary_papers,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
