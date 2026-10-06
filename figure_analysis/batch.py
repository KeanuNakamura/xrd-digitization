"""Batch processing for scientific figure metadata extraction."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from figure_analysis.analyzer import analyze_figure_image, _ensure_autodigitizer_path
from figure_analysis.io_util import (
    IMAGE_EXTENSIONS,
    analysis_json_path,
    append_jsonl,
    compact_jsonl_record,
    copied_image_path,
    copy_source_image,
    figure_stem,
    format_summary,
    load_analysis_if_valid,
    summary_txt_path,
    write_json,
)
from figure_analysis.schemas import FigureAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)


def discover_images(input_path: Path, *, recursive: bool = True) -> list[Path]:
    """Find supported figure images under ``input_path``."""
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern_iter = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        p
        for p in pattern_iter
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


def _process_one(
    image_path: Path,
    output_root: Path,
    *,
    client,
    overwrite: bool,
    write_summary: bool,
    analyze_fn: Callable[..., FigureAnalysisResult],
    rate_lock: threading.Semaphore,
) -> tuple[str, Optional[FigureAnalysisResult], Optional[dict[str, Any]]]:
    """Process a single image. Returns (status, result, failure_record)."""
    stem = figure_stem(image_path)
    out_json = analysis_json_path(output_root, stem)

    if not overwrite:
        existing = load_analysis_if_valid(out_json)
        if existing is not None:
            logger.info("Skip (resume): %s", image_path.name)
            return "skipped", existing, None

    try:
        with rate_lock:
            result = analyze_fn(
                image_path,
                client=client,
                cache_key=f"figure_analysis/{stem}",
                source_image_name=image_path.name,
            )

        # Write outputs
        dest_img = copied_image_path(output_root, stem, image_path)
        copy_source_image(image_path, dest_img)
        write_json(out_json, result)
        if write_summary:
            summary_txt_path(output_root, stem).write_text(
                format_summary(result), encoding="utf-8"
            )
        return "succeeded", result, None
    except Exception as exc:
        logger.exception("Failed analyzing %s", image_path)
        failure = {
            "source_image": image_path.name,
            "source_path": str(image_path),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }
        return "failed", None, failure


def run_batch(
    input_path: Path,
    output_dir: Path,
    *,
    model: Optional[str] = None,
    workers: int = 1,
    overwrite: bool = False,
    write_summary: bool = True,
    use_cache: bool = True,
    recursive: bool = True,
    analyze_fn: Optional[Callable[..., FigureAnalysisResult]] = None,
    client=None,
) -> BatchStats:
    """Analyze all images under ``input_path`` into ``output_dir``."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(input_path, recursive=recursive)
    stats = BatchStats(total=len(images))
    if not images:
        logger.warning("No images found under %s", input_path)
        return stats

    analyze = analyze_fn or analyze_figure_image
    _ensure_autodigitizer_path()
    from autodigitizer.config import Config

    if client is None and analyze_fn is None:
        cfg = Config.from_env(model=model) if model else Config.from_env()
        cfg.use_cache = use_cache
        if model:
            cfg.model = model
        from autodigitizer.vision.openai_client import OpenAIVisionClient

        client = OpenAIVisionClient(cfg)
    elif client is None:
        client = object()  # placeholder; analyze_fn must not need a real client

    jsonl_path = output_dir / "all_figure_metadata.jsonl"
    failed_path = output_dir / "failed_images.jsonl"
    rate_lock = threading.Semaphore(max(1, min(workers, 8)))

    t0 = time.time()
    workers = max(1, int(workers))
    logger.info(
        "Starting batch: %d images → %s (workers=%d overwrite=%s)",
        len(images),
        output_dir,
        workers,
        overwrite,
    )

    def _job(path: Path):
        return path, _process_one(
            path,
            output_dir,
            client=client,
            overwrite=overwrite,
            write_summary=write_summary,
            analyze_fn=analyze,
            rate_lock=rate_lock,
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, p): p for p in images}
        for fut in as_completed(futures):
            path, (status, result, failure) = fut.result()
            completed += 1
            if status == "succeeded":
                stats.succeeded += 1
                assert result is not None
                append_jsonl(jsonl_path, compact_jsonl_record(result))
            elif status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
                if failure:
                    stats.failures.append(failure)
                    append_jsonl(failed_path, failure)

            logger.info(
                "Progress %d/%d — last=%s status=%s",
                completed,
                stats.total,
                path.name,
                status,
            )

    stats.elapsed_seconds = round(time.time() - t0, 3)

    if hasattr(client, "api_calls"):
        client_api = getattr(client, "api_calls", 0)
        client_cache = getattr(client, "cache_hits", 0)
        stats.api_calls = client_api
        stats.cache_hits = client_cache
    else:
        client_api = stats.api_calls
        client_cache = stats.cache_hits

    usage_payload = {
        "total_images": stats.total,
        "succeeded": stats.succeeded,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "api_calls_sum": stats.api_calls,
        "cache_hits_sum": stats.cache_hits,
        "client_api_calls": client_api,
        "client_cache_hits": client_cache,
        "elapsed_seconds": stats.elapsed_seconds,
        "model": model or getattr(getattr(client, "config", None), "model", None),
        "workers": workers,
    }
    write_json(output_dir / "run_usage.json", usage_payload)
    logger.info(
        "Batch done: ok=%d skip=%d fail=%d in %.1fs",
        stats.succeeded,
        stats.skipped,
        stats.failed,
        stats.elapsed_seconds,
    )
    return stats
