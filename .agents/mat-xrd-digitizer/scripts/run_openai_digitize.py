#!/usr/bin/env python3
"""Autonomous OpenAI vision digitizer for single-curve XRD plots.

Flow (matches mat-xrd-digitizer skill):
  figure PNG
    -> OpenAI vision extracts every visible peak as a JSON list
    -> digitize_plot.py builds pseudo-Voigt .xy + preview PNG

CNRS batch (--png-dir) packages PlotDigitizer-matched outputs:
  figure_N_digitized/{pattern_N.png, peaks.json / pattern_N.json,
                      figure_N.csv, figure_N_digitized.xy/.png,
                      figure_N_overlay.png}

# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/run_openai_digitize.py path/to/figure.png
python .agents/mat-xrd-digitizer/scripts/run_openai_digitize.py data/CNRS_figures --png-dir \\
  --output-dir data/CNRS_digitized_agent --json-dir data/CNRS
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1"

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DIGITIZE_PLOT_PATH = Path(__file__).resolve().parent / "digitize_plot.py"
DEFAULT_CNRS_DIGITIZED_AGENT_DIR = REPO_ROOT / "data" / "CNRS_digitized_agent"
DEFAULT_CNRS_JSON_DIR = REPO_ROOT / "data" / "CNRS"
PATTERN_STEM_RE = re.compile(r"^pattern_(\d+)$", re.IGNORECASE)
FIGURE_STEM_RE = re.compile(r"^figure_(\d+)$", re.IGNORECASE)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYSTEM_PROMPT = """\
You digitize single-curve XRD plots. Return JSON only.
Extract EVERY visible peak on the single plotted curve, including tiny minor peaks.
Intensity is normalized 0–1 with the tallest peak = 1.0. fwhm defaults to 0.3 degrees
unless a peak is clearly broader/narrower.
If the figure is not a single-curve XRD plot, return {"is_xrd": false, "reason": "..."}.
"""

USER_PROMPT = """\
Digitize this single-curve XRD figure.

source_image must be exactly: {filename}

If not a single-curve XRD / diffraction plot, return ONLY:
{{"is_xrd": false, "reason": "..."}}

Otherwise return ONLY:
{{
  "is_xrd": true,
  "source_image": "{filename}",
  "min_x": <leftmost axis tick as float>,
  "max_x": <rightmost axis tick as float>,
  "peaks": [
    {{"2theta": <float>, "intensity": <0-1>, "fwhm": <float>}},
    ...
  ]
}}

Rules:
- Include every visible peak (major and minor).
- intensity: relative height of each peak tip, tallest = 1.0.
- fwhm: peak width in degrees (default 0.3).
- Read 2θ from axis tick labels.
- Return valid JSON only (no markdown).
"""


def _require_requests():
    try:
        import requests  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'requests'. Install in base-agent: pip install requests"
        ) from exc
    return __import__("requests")


def encode_image_b64(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    return base64.b64encode(data).decode("ascii"), mime or "image/png"


def parse_json_content(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Vision response must be a JSON object")
    return payload


def _retry_wait_seconds(response: Any, attempt: int) -> float:
    retry_after = None
    try:
        retry_after = response.headers.get("Retry-After")
    except Exception:
        pass
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return min(90.0, (2**attempt) + random.uniform(0.25, 1.5))


def call_openai_vision(
    image_path: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_retries: int = 8,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    requests = _require_requests()
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("XRD_AGENT_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY (or XRD_AGENT_API_KEY)")

    url_base = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_name = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    b64, mime = encode_image_b64(image_path)
    body = {
        "model": model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(filename=image_path.name)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{url_base}/chat/completions"
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout_s)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            wait = min(60.0, (2**attempt) + random.uniform(0.25, 1.5))
            print(f"OpenAI request error ({exc}); retry {attempt + 1}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            wait = _retry_wait_seconds(response, attempt)
            print(
                f"OpenAI HTTP {response.status_code}; "
                f"retry {attempt + 1}/{max_retries} in {wait:.1f}s"
            )
            time.sleep(wait)
            last_error = RuntimeError(f"{response.status_code} for url: {url}")
            continue

        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("OpenAI message content must be a string")
        return parse_json_content(content)

    raise RuntimeError(f"OpenAI failed after {max_retries} retries: {last_error}") from last_error


def normalize_peaks(payload: dict[str, Any]) -> tuple[list[dict[str, float]], float, float]:
    """Return cleaned peaks list plus axis range."""
    raw_peaks = payload.get("peaks")
    if not isinstance(raw_peaks, list):
        curves = payload.get("curves")
        if isinstance(curves, list) and curves and isinstance(curves[0], dict):
            raw_peaks = curves[0].get("peaks")
    if not isinstance(raw_peaks, list) or not raw_peaks:
        raise ValueError("Vision JSON missing non-empty 'peaks' list")

    peaks: list[dict[str, float]] = []
    for peak in raw_peaks:
        if not isinstance(peak, dict):
            continue
        tt = peak.get("2theta", peak.get("two_theta"))
        inten = peak.get("intensity")
        if tt is None or inten is None:
            continue
        fwhm = peak.get("fwhm", 0.3)
        try:
            peaks.append(
                {
                    "2theta": float(tt),
                    "intensity": max(0.0, min(1.0, float(inten))),
                    "fwhm": float(fwhm) if float(fwhm) > 0 else 0.3,
                }
            )
        except (TypeError, ValueError):
            continue

    if not peaks:
        raise ValueError("No valid peaks after normalization")

    # Renormalize so tallest peak is 1.0
    max_i = max(p["intensity"] for p in peaks)
    if max_i > 0:
        for peak in peaks:
            peak["intensity"] = peak["intensity"] / max_i

    peaks.sort(key=lambda p: p["2theta"])
    min_x = float(payload.get("min_x", payload.get("x_min", 5.0)))
    max_x = float(payload.get("max_x", payload.get("x_max", 80.0)))
    x_axis = payload.get("x_axis")
    if isinstance(x_axis, dict):
        min_x = float(x_axis.get("min", min_x))
        max_x = float(x_axis.get("max", max_x))
    if max_x <= min_x:
        min_x, max_x = 5.0, 80.0
    return peaks, min_x, max_x


def run_digitize_plot(
    peaks_json: Path,
    output_xy: Path,
    *,
    min_x: float,
    max_x: float,
    points: int,
    noise: float,
    background: float,
) -> None:
    cmd = [
        sys.executable,
        str(DIGITIZE_PLOT_PATH),
        str(peaks_json),
        "--output",
        str(output_xy),
        "--min-x",
        str(min_x),
        "--max-x",
        str(max_x),
        "--points",
        str(points),
        "--noise",
        str(noise),
        "--background",
        str(background),
    ]
    subprocess.run(cmd, check=True)


def xy_to_csv(xy_path: Path, csv_path: Path) -> None:
    rows: list[tuple[float, float]] = []
    for raw in xy_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric rows in {xy_path}")
    csv_path.write_text(
        "two_theta,intensity\n"
        + "\n".join(f"{x:.3f},{y:.6g}" for x, y in rows)
        + "\n",
        encoding="utf-8",
    )


def pattern_index_from_stem(stem: str) -> int | None:
    for regex in (PATTERN_STEM_RE, FIGURE_STEM_RE):
        match = regex.match(stem)
        if match:
            return int(match.group(1))
    return None


def save_sid_overlay(
    truth_json: Path,
    csv_path: Path,
    overlay_path: Path,
    *,
    title: str | None = None,
    scores_path: Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np
    from compute_sid import compare_spectra, format_peak_match_debug, write_score_files

    comparison = compare_spectra(truth_json, csv_path)
    print(format_peak_match_debug(comparison))

    def _norm(values: Any) -> Any:
        arr = np.asarray(values, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    true_y = _norm(comparison["true_y"])
    approx_y = _norm(comparison["approx_y"])
    raw_sid = comparison["raw_sid"]
    modified_sid = comparison["modified_sid"]
    final_score = comparison["final_xrd_score"]

    fig, axis = plt.subplots(figsize=(10, 4.5), dpi=150)
    axis.plot(comparison["true_x"], true_y, color="#0072B2", linewidth=1.4, label="Original (JSON)")
    axis.plot(
        comparison["approx_x"],
        approx_y,
        color="#D55E00",
        linewidth=1.2,
        alpha=0.9,
        label="Digitized (CSV)",
    )
    axis.set_xlabel("2θ (degrees)")
    axis.set_ylabel("Normalized intensity")
    axis.set_ylim(-0.02, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", fontsize=9)
    plot_title = title or overlay_path.stem
    axis.set_title(
        f"{plot_title}  |  Raw={raw_sid:.4g}  Mod={modified_sid:.4g}  "
        f"XRD={final_score:.4g}"
    )
    axis.text(
        0.02,
        0.98,
        (
            f"Raw SID = {raw_sid:.6g}\n"
            f"Modified SID = {modified_sid:.6g}\n"
            f"Peak Recall = {comparison['peak_recall']:.4f}\n"
            f"Peak Precision = {comparison['peak_precision']:.4f}\n"
            f"Peak F1 = {comparison['peak_f1']:.4f}\n"
            f"Final XRD Score = {final_score:.6g}"
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)

    figure_id = title or overlay_path.stem.replace("_overlay", "")
    write_score_files(
        comparison,
        figure_dir=overlay_path.parent,
        figure_id=figure_id,
        summary_path=scores_path,
    )
    return comparison


def digitize_image(
    image_path: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    points: int = 4000,
    noise: float = 0.01,
    background: float = 0.05,
    out_dir: Path | None = None,
    output_stem: str | None = None,
    skip_move: bool = False,
    dry_run_json: Path | None = None,
    max_retries: int = 8,
) -> dict[str, Any]:
    """Digitize one single-curve XRD image."""
    image_path = image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    if dry_run_json is not None:
        payload = json.loads(dry_run_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payload = {"is_xrd": True, "peaks": payload, "min_x": 5.0, "max_x": 80.0}
    else:
        payload = call_openai_vision(
            image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_retries=max_retries,
        )

    if payload.get("is_xrd") is False:
        return {
            "status": "skipped",
            "image": str(image_path),
            "reason": payload.get("reason") or "not_xrd",
        }

    if out_dir is not None:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        image_in_dir = out_dir / image_path.name
        if image_path.resolve() != image_in_dir.resolve():
            shutil.copy2(image_path, image_in_dir)
    elif skip_move:
        out_dir = image_path.parent / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        image_in_dir = out_dir / image_path.name
        if not image_in_dir.exists():
            shutil.copy2(image_path, image_in_dir)
    else:
        out_dir = image_path.parent / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        image_in_dir = out_dir / image_path.name
        if image_path.resolve() != image_in_dir.resolve() and not image_in_dir.exists():
            shutil.move(str(image_path), str(image_in_dir))
            image_path = image_in_dir

    peaks, min_x, max_x = normalize_peaks(payload)
    dig_stem = output_stem or image_in_dir.stem
    peaks_path = out_dir / f"{image_in_dir.stem}.json"
    peaks_path.write_text(json.dumps(peaks, indent=2), encoding="utf-8")

    output_xy = out_dir / f"{dig_stem}_digitized.xy"
    run_digitize_plot(
        peaks_path,
        output_xy,
        min_x=min_x,
        max_x=max_x,
        points=points,
        noise=noise,
        background=background,
    )
    return {
        "status": "digitized",
        "image": str(image_in_dir),
        "json": str(peaks_path),
        "output_dir": str(out_dir),
        "xy": str(output_xy),
        "digitized_png": str(output_xy.with_suffix(".png")),
        "n_peaks": len(peaks),
        "min_x": min_x,
        "max_x": max_x,
    }


def iter_png_directory(png_dir: Path) -> list[Path]:
    png_dir = png_dir.resolve()
    files = sorted(png_dir.glob("pattern_*.png"))
    if not files:
        files = sorted(png_dir.glob("figure_*.png"))
    if not files:
        files = sorted(p for p in png_dir.glob("*.png") if p.is_file())
    return files


def digitize_png_directory(
    png_dir: Path,
    *,
    output_dir: Path = DEFAULT_CNRS_DIGITIZED_AGENT_DIR,
    json_dir: Path = DEFAULT_CNRS_JSON_DIR,
    skip_existing: bool = True,
    overwrite: bool = False,
    limit: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    points: int = 4000,
    noise: float = 0.01,
    background: float = 0.05,
    from_json_dir: Path | None = None,
    max_retries: int = 8,
    request_delay: float = 1.5,
) -> dict[str, Any]:
    """Batch-digitize flat PNG dirs into figure_N_digitized/ with SID overlays."""
    png_dir = png_dir.resolve()
    output_dir = output_dir.resolve()
    json_dir = json_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = iter_png_directory(png_dir)
    if limit is not None:
        png_files = png_files[: max(0, limit)]
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {png_dir}")

    counts = {"succeeded": 0, "failed": 0, "skipped": 0, "total": len(png_files)}
    sid_rows: list[dict[str, Any]] = []
    summary_path = output_dir / "sid_summary.json"

    for index, png_path in enumerate(png_files, start=1):
        pattern_index = pattern_index_from_stem(png_path.stem)
        figure_id = (
            f"figure_{pattern_index}" if pattern_index is not None else png_path.stem
        )
        figure_dir = output_dir / f"{figure_id}_digitized"
        csv_path = figure_dir / f"{figure_id}.csv"
        digitized_png = figure_dir / f"{figure_id}_digitized.png"
        digitized_xy = figure_dir / f"{figure_id}_digitized.xy"
        overlay_path = figure_dir / f"{figure_id}_overlay.png"
        original_copy = figure_dir / png_path.name
        truth_json = (
            json_dir / f"pattern_{pattern_index}.json"
            if PATTERN_STEM_RE.match(png_path.stem) and pattern_index is not None
            else None
        )
        has_truth = truth_json is not None and truth_json.is_file()

        already_done = (
            csv_path.is_file()
            and digitized_png.is_file()
            and original_copy.is_file()
            and (not has_truth or overlay_path.is_file())
        )
        if already_done and skip_existing and not overwrite:
            counts["skipped"] += 1
            print(f"[{index}/{counts['total']}] skip (exists): {figure_dir.name}")
            continue

        try:
            if overwrite and figure_dir.exists():
                shutil.rmtree(figure_dir)
            figure_dir.mkdir(parents=True, exist_ok=True)

            dry_json = None
            if from_json_dir is not None:
                candidate = from_json_dir / f"{png_path.stem}.json"
                if candidate.is_file():
                    dry_json = candidate

            if dry_json is None and request_delay > 0 and index > 1:
                time.sleep(request_delay)

            result = digitize_image(
                png_path,
                api_key=api_key,
                base_url=base_url,
                model=model,
                points=points,
                noise=noise,
                background=background,
                out_dir=figure_dir,
                output_stem=figure_id,
                dry_run_json=dry_json,
                max_retries=max_retries,
            )
            if result.get("status") != "digitized":
                raise RuntimeError(result.get("reason") or "skipped/failed")

            xy_path = Path(result["xy"])
            if xy_path.resolve() != digitized_xy.resolve():
                shutil.copy2(xy_path, digitized_xy)
            preview = Path(result["digitized_png"])
            if preview.is_file() and preview.resolve() != digitized_png.resolve():
                shutil.copy2(preview, digitized_png)
            if not digitized_png.is_file():
                raise RuntimeError(f"Missing digitized PNG for {figure_id}")

            xy_to_csv(digitized_xy, csv_path)

            sid_note = ""
            if has_truth:
                comparison = save_sid_overlay(
                    truth_json,
                    csv_path,
                    overlay_path,
                    title=figure_id,
                    scores_path=summary_path,
                )
                sid_note = (
                    f" raw={comparison['raw_sid']:.4g}"
                    f" mod={comparison['modified_sid']:.4g}"
                    f" xrd={comparison['final_xrd_score']:.4g}"
                )
                from compute_sid import score_record

                row = score_record(comparison, figure_id=figure_id)
                row["n_peaks"] = int(result.get("n_peaks") or 0)
                row["out_dir"] = str(figure_dir)
                sid_rows.append(row)

            counts["succeeded"] += 1
            print(
                f"[{index}/{counts['total']}] ok {figure_dir.name} "
                f"(peaks={result.get('n_peaks')}){sid_note}"
            )
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            print(f"[{index}/{counts['total']}] FAILED {png_path.name}: {exc}")

    if summary_path.is_file():
        print(f"Wrote SID summary: {summary_path}")

    return {"counts": counts, "sid_rows": sid_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous single-curve XRD digitizer (OpenAI vision + pseudo-Voigt). "
            "Use --png-dir for CNRS-style batch packaging + SID overlays."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Single figure image, or PNG directory with --png-dir",
    )
    parser.add_argument(
        "--png-dir",
        action="store_true",
        help="Treat positional path as a flat PNG directory (CNRS batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CNRS_DIGITIZED_AGENT_DIR,
        help="Output root for --png-dir (default: data/CNRS_digitized_agent)",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_CNRS_JSON_DIR,
        help="Ground-truth JSON dir for SID overlays (default: data/CNRS)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip complete figure folders (default: on)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redo existing folders")
    parser.add_argument("--limit", type=int, default=None, help="Max PNGs in --png-dir")
    parser.add_argument(
        "--from-json-dir",
        type=Path,
        default=None,
        help="Skip vision; load {stem}.json peaks from this directory",
    )
    parser.add_argument("--from-json", type=Path, help="Skip vision for a single image")
    parser.add_argument("--skip-move", action="store_true", help="Copy instead of move")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--background", type=float, default=0.05)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.5,
        help="Seconds between OpenAI calls in batch mode",
    )
    args = parser.parse_args(argv)

    if args.png_dir:
        if args.image is None:
            parser.error("--png-dir requires a PNG directory path")
        if not args.image.exists():
            raise SystemExit(f"Directory not found: {args.image}")
        result = digitize_png_directory(
            args.image,
            output_dir=args.output_dir,
            json_dir=args.json_dir,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
            limit=args.limit,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            points=args.points,
            noise=args.noise,
            background=args.background,
            from_json_dir=args.from_json_dir,
            max_retries=args.max_retries,
            request_delay=args.request_delay,
        )
        counts = result["counts"]
        print(
            "PNG batch complete: "
            f"succeeded={counts['succeeded']}, skipped={counts['skipped']}, "
            f"failed={counts['failed']}, total={counts['total']} -> {args.output_dir}"
        )
        return 1 if counts["failed"] else 0

    if args.image is None:
        parser.error("Provide an image path or use --png-dir")

    result = digitize_image(
        args.image,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        points=args.points,
        noise=args.noise,
        background=args.background,
        skip_move=args.skip_move,
        dry_run_json=args.from_json,
        max_retries=args.max_retries,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"digitized", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
