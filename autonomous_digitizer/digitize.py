#!/usr/bin/env python3
"""CLI entrypoint for autonomous scientific figure digitization."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autodigitizer.config import Config
from autodigitizer.pipeline import DigitizationPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous scientific figure digitizer "
        "(OpenAI for interpretation, local CV for extraction)."
    )
    p.add_argument("image", type=Path, help="Path to figure image (PNG/JPG/...)")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: outputs/<image_stem>)",
    )
    p.add_argument("--model", type=str, default=None, help="OpenAI model name")
    p.add_argument("--max-retries", type=int, default=2, help="QA retry limit per subplot")
    p.add_argument("--samples", type=int, default=1000, help="Samples per curve")
    p.add_argument("--debug", action="store_true", help="Write pipeline.log + DEBUG file logs")
    p.add_argument("--no-cache", action="store_true", help="Disable OpenAI response cache")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Cache directory for OpenAI responses",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip OpenAI calls (deterministic fallbacks; for testing)",
    )
    return p


def _configure_logging(debug: bool, log_path: Path | None = None) -> None:
    """Quiet console by default; detailed logs only with --debug."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if debug else logging.WARNING)

    # Silence noisy HTTP clients even in debug (they drown progress)
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if debug and log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
        # Keep our package info+ in the file
        logging.getLogger("autodigitizer").setLevel(logging.DEBUG)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 2

    output = args.output or Path("outputs") / args.image.stem
    _configure_logging(args.debug, output / "pipeline.log" if args.debug else None)

    overrides = {
        "output_dir": output,
        "cache_dir": args.cache_dir,
        "use_cache": not args.no_cache,
        "max_retries": args.max_retries,
        "samples_per_curve": args.samples,
        "debug": args.debug,
        "verbose": True,
    }
    if args.model:
        overrides["model"] = args.model

    config = Config.from_env(**overrides)
    pipeline = DigitizationPipeline(config, offline=args.offline)
    report = pipeline.run(args.image)
    print(f"{report.status.value} → {output}")
    return 0 if report.status.value in {"success", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
