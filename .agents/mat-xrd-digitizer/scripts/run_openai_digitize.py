#!/usr/bin/env python3
"""OpenAI vision pipeline for mat-xrd-digitizer (separate from PlotDigitizer).

Flow:
  XRD image
    -> OpenAI vision extracts peaks JSON (SKILL.md schema)
    -> digitize_plot.py builds pseudo-Voigt .xy + preview PNG

This does not call PlotDigitizer or the hybrid xrd_digitization path.

# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/run_openai_digitize.py path/to/figure.png
python .agents/mat-xrd-digitizer/scripts/run_openai_digitize.py --batch grobid_output/sample_pdfs
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1"

SKILL_DIR = Path(__file__).resolve().parents[1]
SKILL_MD_PATH = SKILL_DIR / "SKILL.md"
DIGITIZE_PLOT_PATH = SKILL_DIR / "scripts" / "digitize_plot.py"

# Minimal single-image ask — mirrors a typical Cursor skill invocation.
# Extra coaching is intentionally omitted so API vs Cursor get the same skill text.
TASK_PROMPT = """\
Digitize this figure using the mat-xrd-digitizer skill (single-image workflow).

- If the figure is not an XRD / diffraction plot, return ONLY \
{{"is_xrd": false, "reason": "..."}}.
- If it is XRD, return ONLY the skill JSON schema (set "is_xrd": true if useful).
- source_image must be exactly: {filename}
- Follow the skill document and digitize_plot.py reference in the system message.
- Return valid JSON only (no markdown fences, no commentary).
"""


def load_skill_markdown() -> str:
    """Load SKILL.md so the API receives the same skill text Cursor uses."""
    if not SKILL_MD_PATH.exists():
        raise FileNotFoundError(f"Missing skill file: {SKILL_MD_PATH}")
    return SKILL_MD_PATH.read_text(encoding="utf-8")


def load_digitize_plot_reference() -> str:
    """Load digitize_plot.py — Cursor skill agents can open this file."""
    if not DIGITIZE_PLOT_PATH.exists():
        raise FileNotFoundError(f"Missing digitize script: {DIGITIZE_PLOT_PATH}")
    return DIGITIZE_PLOT_PATH.read_text(encoding="utf-8")


def build_system_prompt() -> str:
    """Same information surface as Cursor: full SKILL.md + digitize_plot.py."""
    return (
        "You are running the mat-xrd-digitizer skill. The two files below are the "
        "same documents a Cursor agent loads for this skill. Treat them as "
        "authoritative.\n\n"
        "Your job in this API call is ONLY vision extraction of the peak JSON "
        "described by the skill. Do not invent shell commands; another process "
        "will run digitize_plot.py on your JSON.\n\n"
        "===== FILE: .agents/mat-xrd-digitizer/SKILL.md =====\n"
        f"{load_skill_markdown()}\n\n"
        "===== FILE: .agents/mat-xrd-digitizer/scripts/digitize_plot.py =====\n"
        f"{load_digitize_plot_reference()}\n"
    )


def build_user_prompt(filename: str) -> str:
    return TASK_PROMPT.format(filename=filename)

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
    if mime is None:
        mime = "image/png"
    return base64.b64encode(data).decode("ascii"), mime


def parse_json_content(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Vision response JSON must be an object")
    return payload


def call_openai_vision(
    image_path: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Call OpenAI-compatible chat completions with the figure image.

    System prompt = full SKILL.md + digitize_plot.py (same files Cursor can load).
    User prompt = minimal single-image skill ask (no extra API-only coaching).
    """
    requests = _require_requests()
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("XRD_AGENT_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY (or XRD_AGENT_API_KEY) to run vision digitization")

    url_base = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_name = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    b64, mime = encode_image_b64(image_path)
    system_prompt = build_system_prompt()
    user_text = build_user_prompt(image_path.name)

    body = {
        "model": model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
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
    response = requests.post(
        f"{url_base}/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected OpenAI response shape: {data!r}") from exc
    if not isinstance(content, str):
        raise ValueError("OpenAI message content must be a string")
    return parse_json_content(content)


def normalize_digitization_json(payload: dict[str, Any], source_image: str) -> dict[str, Any]:
    """Ensure SKILL.md-compatible fields for digitize_plot.py."""
    out = dict(payload)
    out.pop("is_xrd", None)
    out["source_image"] = source_image
    out.setdefault("figure_type", "xrd")

    if "plots" in out and isinstance(out["plots"], list):
        out.setdefault("figure_layout", "multi_panel")
        for i, plot in enumerate(out["plots"], start=1):
            if not isinstance(plot, dict):
                continue
            plot.setdefault("plot_id", f"plot_{i}")
            plot.setdefault("curve_layout", "overlay")
            plot.setdefault("source_image", source_image)
            _normalize_axes(plot)
            _normalize_curves(plot.get("curves") or [])
            plot.setdefault("noise", out.get("noise", 0.01))
            plot.setdefault("background", out.get("background", 0.03))
        out.setdefault("noise", 0.01)
        out.setdefault("background", 0.03)
        return out

    out.setdefault("curve_layout", "overlay")
    _normalize_axes(out)
    curves = out.get("curves")
    if not curves:
        # Allow accidental legacy peak list under "peaks".
        peaks = out.get("peaks")
        if isinstance(peaks, list):
            out["curves"] = [
                {
                    "curve_id": "curve_1",
                    "intensity_normalization": "normalized_within_curve",
                    "peaks": peaks,
                }
            ]
    _normalize_curves(out.get("curves") or [])
    # Match prior CNRS Cursor-agent simulation defaults when unspecified.
    out.setdefault("noise", 0.01)
    out.setdefault("background", 0.03)
    return out


def _normalize_axes(obj: dict[str, Any]) -> None:
    x_axis = obj.get("x_axis")
    if not isinstance(x_axis, dict):
        x_axis = {}
        obj["x_axis"] = x_axis
    x_axis.setdefault("label", "2theta")
    x_axis.setdefault("unit", "degrees")
    x_axis.setdefault("min", 5.0)
    x_axis.setdefault("max", 80.0)

    y_axis = obj.get("y_axis")
    if not isinstance(y_axis, dict):
        y_axis = {}
        obj["y_axis"] = y_axis
    y_axis.setdefault("label", "intensity")
    y_axis.setdefault("unit", "normalized")
    y_axis.setdefault("min", 0.0)
    y_axis.setdefault("max", 1.0)


def _normalize_curves(curves: list[Any]) -> None:
    for i, curve in enumerate(curves, start=1):
        if not isinstance(curve, dict):
            continue
        curve.setdefault("curve_id", f"curve_{i}")
        curve.setdefault("intensity_normalization", "normalized_within_curve")
        peaks = curve.get("peaks") or []
        cleaned = []
        for peak in peaks:
            if not isinstance(peak, dict):
                continue
            tt = peak.get("2theta", peak.get("two_theta"))
            if tt is None:
                continue
            inten = float(peak.get("intensity", 0.0))
            fwhm = float(peak.get("fwhm", 0.3) or 0.3)
            cleaned.append(
                {
                    "2theta": float(tt),
                    "intensity": max(0.0, min(1.0, inten)),
                    "fwhm": fwhm if fwhm > 0 else 0.3,
                }
            )
        curve["peaks"] = cleaned


def prepare_output_dir(image_path: Path) -> Path:
    """Create {parent}/{stem}/ and move/copy the image beside outputs."""
    stem = image_path.stem
    out_dir = image_path.parent / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / image_path.name
    if image_path.resolve() != dest.resolve():
        if not dest.exists():
            # Prefer move when the image still sits next to sibling figures.
            try:
                shutil.move(str(image_path), str(dest))
            except Exception:
                shutil.copy2(image_path, dest)
        image_path = dest
    return out_dir if dest.exists() else out_dir


def resolve_image_in_output_dir(image_path: Path, out_dir: Path) -> Path:
    candidate = out_dir / image_path.name
    if candidate.exists():
        return candidate
    if image_path.exists():
        return image_path
    raise FileNotFoundError(f"Image not found after output prep: {image_path}")


def run_digitize_plot(
    json_path: Path,
    output_xy: Path,
    *,
    min_x: float,
    max_x: float,
    points: int,
    noise: float,
    background: float,
    python_exe: str | None = None,
) -> None:
    script = Path(__file__).resolve().parent / "digitize_plot.py"
    cmd = [
        python_exe or sys.executable,
        str(script),
        str(json_path),
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


def digitize_image(
    image_path: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    points: int = 4000,
    noise: float = 0.01,
    background: float = 0.03,
    skip_move: bool = False,
    dry_run_json: Path | None = None,
) -> dict[str, Any]:
    """Digitize one image with OpenAI vision + digitize_plot.py."""
    image_path = image_path.expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    else:
        image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "Pass a real filesystem path to a .png/.jpg figure, e.g.\n"
            "  python .agents/mat-xrd-digitizer/scripts/run_openai_digitize.py "
            "path/to/figure.png\n"
            "Chat attachments like '[Image #1]' are not valid paths."
        )
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{image_path.suffix}'. "
            f"Expected one of: {sorted(IMAGE_EXTENSIONS)}"
        )

    if dry_run_json is not None:
        payload = json.loads(dry_run_json.read_text(encoding="utf-8"))
    else:
        payload = call_openai_vision(
            image_path, api_key=api_key, base_url=base_url, model=model
        )

    if payload.get("is_xrd") is False:
        return {
            "status": "skipped",
            "image": str(image_path),
            "reason": payload.get("reason") or "not_xrd",
        }

    if skip_move:
        out_dir = image_path.parent / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / image_path.name).exists():
            shutil.copy2(image_path, out_dir / image_path.name)
        image_in_dir = out_dir / image_path.name
    else:
        out_dir = prepare_output_dir(image_path)
        image_in_dir = resolve_image_in_output_dir(image_path, out_dir)

    stem = image_in_dir.stem
    dig_json = normalize_digitization_json(payload, source_image=image_in_dir.name)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(dig_json, indent=2), encoding="utf-8")

    x_axis = dig_json.get("x_axis") or {}
    if "plots" in dig_json and dig_json["plots"]:
        first = dig_json["plots"][0]
        x_axis = first.get("x_axis") or x_axis
    min_x = float(x_axis.get("min", 5.0))
    max_x = float(x_axis.get("max", 80.0))

    output_xy = out_dir / f"{stem}_digitized.xy"
    run_digitize_plot(
        json_path,
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
        "json": str(json_path),
        "output_dir": str(out_dir),
        "xy": str(output_xy),
        "curve_layout": dig_json.get("curve_layout") or dig_json.get("figure_layout"),
        "n_curves": (
            sum(len(p.get("curves") or []) for p in dig_json.get("plots") or [])
            if "plots" in dig_json
            else len(dig_json.get("curves") or [])
        ),
    }


def iter_batch_images(root: Path) -> list[Path]:
    """Find figure images under grobid_output/sample_pdfs/{example}/figures/."""
    images: list[Path] = []
    root = root.resolve()
    if not root.exists():
        return images

    # Prefer .../{example}/figures/* pattern; also accept a direct figures dir.
    candidates = []
    if root.name == "figures":
        candidates.append(root)
    else:
        candidates.extend(sorted(p for p in root.glob("*/figures") if p.is_dir()))
        candidates.extend(sorted(p for p in root.glob("**/figures") if p.is_dir()))

    seen: set[Path] = set()
    for figures_dir in candidates:
        figures_dir = figures_dir.resolve()
        if figures_dir in seen:
            continue
        seen.add(figures_dir)
        for path in sorted(figures_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                # Skip images already inside a stem subdirectory.
                if path.parent.name == path.stem:
                    continue
                images.append(path)
    return images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI vision XRD digitizer for mat-xrd-digitizer. "
            "Separate from PlotDigitizer / hybrid pixel tracing."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Single XRD figure image to digitize",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="Batch root (e.g. grobid_output/sample_pdfs) to scan */figures/",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key (else OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help=f"Vision model (default: {DEFAULT_MODEL})")
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--background", type=float, default=0.03)
    parser.add_argument(
        "--skip-move",
        action="store_true",
        help="Copy image into output dir instead of moving it",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        help="Skip vision call and digitize using an existing peaks JSON",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write a JSON processing summary",
    )
    args = parser.parse_args(argv)

    if not args.image and not args.batch:
        parser.error("Provide an image path or --batch root")

    results: list[dict[str, Any]] = []

    if args.image:
        if args.from_json and not args.image.exists() and args.skip_move:
            parser.error("image path must exist")
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
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    if args.batch:
        images = iter_batch_images(args.batch)
        print(f"Found {len(images)} candidate figure image(s) under {args.batch}")
        for image_path in images:
            print(f"\n=== {image_path} ===")
            try:
                result = digitize_image(
                    image_path,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    points=args.points,
                    noise=args.noise,
                    background=args.background,
                    skip_move=args.skip_move,
                )
            except Exception as exc:  # noqa: BLE001 - batch continues
                result = {
                    "status": "error",
                    "image": str(image_path),
                    "error": str(exc),
                }
            results.append(result)
            print(json.dumps(result, indent=2))

    summary = {
        "n_inspected": len(results),
        "n_digitized": sum(1 for r in results if r.get("status") == "digitized"),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "n_errors": sum(1 for r in results if r.get("status") == "error"),
        "results": results,
    }
    print("\n=== summary ===")
    print(json.dumps({k: summary[k] for k in ("n_inspected", "n_digitized", "n_skipped", "n_errors")}, indent=2))

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {args.summary}")

    return 1 if summary["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
