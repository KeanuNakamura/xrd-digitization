"""
Create a clean, high-resolution line plot from XRD-like JSON data.

Designed for easy curve extraction with PlotDigitizer:
- white background
- one solid, high-contrast curve
- no title, legend, or grid
- no y-axis ticks, numbers, or label
- relative intensity normalized to 0..1 by default
- high-resolution PNG output

Expected JSON keys by default:
    two_theta_values
    intensities

Single file:
    python plot_pattern.py pattern.json output.png

Batch directory (every *.json -> <stem>.png):
    python plot_pattern.py data/CNRS --output-dir data/CNRS_figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


DEFAULT_X_KEYS = (
    "two_theta_values",
    "two_theta",
    "2theta",
    "x",
    "x_values",
)

DEFAULT_Y_KEYS = (
    "intensities",
    "intensity",
    "counts",
    "y",
    "y_values",
)

DEFAULT_BATCH_OUTPUT_DIR = Path("data/CNRS_figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert JSON x/y arrays into a clean PNG optimized for "
            "digitization with PlotDigitizer. Pass a directory to batch "
            "every *.json file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Input JSON file, or a directory of JSON files for batch mode",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=None,
        help="Output image path (required for single-file mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BATCH_OUTPUT_DIR,
        help="Output directory for batch mode (json_path is a directory)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Force batch mode even if json_path is a file (treat parent as dir)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs in batch mode (default: skip)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N JSON files in batch mode",
    )
    parser.add_argument(
        "--x-key",
        help="JSON key containing x values; auto-detected when omitted",
    )
    parser.add_argument(
        "--y-key",
        help="JSON key containing y values; auto-detected when omitted",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Plot original y values instead of scaling them to 0..1",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.5,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.5,
        help="Curve width in points",
    )
    parser.add_argument(
        "--x-label",
        default=r"2$\theta$ (degrees)",
        help="X-axis label; pass an empty string to hide it",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        help="Optional lower x-axis limit",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        help="Optional upper x-axis limit",
    )
    return parser.parse_args()


def find_key(
    data: dict[str, Any],
    requested: str | None,
    candidates: Iterable[str],
    kind: str,
) -> str:
    if requested is not None:
        if requested not in data:
            raise KeyError(
                f"{kind} key {requested!r} was not found. "
                f"Available top-level keys: {', '.join(data.keys())}"
            )
        return requested

    for key in candidates:
        if key in data:
            return key

    raise KeyError(
        f"Could not auto-detect the {kind} array. "
        f"Use --{kind}-key. Available top-level keys: {', '.join(data.keys())}"
    )


def to_finite_floats(values: Any, key: str) -> list[float]:
    if not isinstance(values, list):
        raise TypeError(f"JSON value at {key!r} must be an array.")

    converted: list[float] = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Value {value!r} at {key}[{index}] is not numeric."
            ) from exc

        if not math.isfinite(number):
            raise ValueError(
                f"Value at {key}[{index}] is not finite: {number!r}"
            )
        converted.append(number)

    if not converted:
        raise ValueError(f"JSON array {key!r} is empty.")

    return converted


def normalize_zero_to_one(values: list[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [0.0 for _ in values]

    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


def plot_json_to_png(
    json_path: Path,
    output_path: Path,
    *,
    x_key: str | None = None,
    y_key: str | None = None,
    no_normalize: bool = False,
    width: float = 12.0,
    height: float = 4.5,
    dpi: int = 300,
    line_width: float = 1.5,
    x_label: str = r"2$\theta$ (degrees)",
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[str, str]:
    """Render one JSON spectrum to a digitizer-friendly PNG.

    Returns:
        (resolved_x_key, resolved_y_key)
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError("The JSON root must be an object containing x and y arrays.")

    resolved_x_key = find_key(data, x_key, DEFAULT_X_KEYS, "x")
    resolved_y_key = find_key(data, y_key, DEFAULT_Y_KEYS, "y")

    x_values = to_finite_floats(data[resolved_x_key], resolved_x_key)
    y_values = to_finite_floats(data[resolved_y_key], resolved_y_key)

    if len(x_values) != len(y_values):
        raise ValueError(
            f"Length mismatch: {resolved_x_key!r} has {len(x_values)} values, "
            f"but {resolved_y_key!r} has {len(y_values)}."
        )

    # Sort by x so the rendered curve never doubles back because of input order.
    points = sorted(zip(x_values, y_values), key=lambda point: point[0])
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    if not no_normalize:
        y_values = normalize_zero_to_one(y_values)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(width, height))

    # A saturated blue curve separates cleanly from black axes and white background.
    axis.plot(
        x_values,
        y_values,
        color="#0072B2",
        linewidth=line_width,
        antialiased=False,
        solid_capstyle="butt",
        solid_joinstyle="miter",
    )

    axis.set_title("")
    axis.grid(False)
    axis.set_xlabel(x_label)

    # Keep the axis boundary for calibration, but remove the complete y scale.
    axis.set_ylabel("")
    axis.set_yticks([])
    axis.tick_params(axis="y", left=False, right=False, labelleft=False)

    # Remove unnecessary top/right borders while retaining left/bottom calibration axes.
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)

    if x_min is not None or x_max is not None:
        current_min, current_max = axis.get_xlim()
        axis.set_xlim(
            x_min if x_min is not None else current_min,
            x_max if x_max is not None else current_max,
        )
    else:
        axis.set_xlim(min(x_values), max(x_values))

    if no_normalize:
        y_min = min(y_values)
        y_max = max(y_values)
        padding = 0.02 * (y_max - y_min) if y_max != y_min else 1.0
        axis.set_ylim(y_min - padding, y_max + padding)
    else:
        # Exact limits make PlotDigitizer calibration simple: bottom=0, top=1.
        axis.set_ylim(0.0, 1.0)

    axis.margins(x=0)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.tight_layout(pad=0.6)

    figure.savefig(
        output_path,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)

    return resolved_x_key, resolved_y_key


def _plot_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "x_key": args.x_key,
        "y_key": args.y_key,
        "no_normalize": args.no_normalize,
        "width": args.width,
        "height": args.height,
        "dpi": args.dpi,
        "line_width": args.line_width,
        "x_label": args.x_label,
        "x_min": args.x_min,
        "x_max": args.x_max,
    }


def run_batch(args: argparse.Namespace) -> None:
    input_dir = args.json_path
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Batch input is not a directory: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if args.limit is not None:
        json_files = json_files[: max(0, args.limit)]

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = _plot_kwargs_from_args(args)

    saved = 0
    skipped = 0
    failed = 0
    total = len(json_files)

    for index, json_path in enumerate(json_files, start=1):
        output_path = output_dir / f"{json_path.stem}.png"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{total}] skip (exists): {output_path.name}")
            continue

        try:
            x_key, y_key = plot_json_to_png(json_path, output_path, **kwargs)
            saved += 1
            print(
                f"[{index}/{total}] saved {output_path.name} "
                f"(x={x_key!r}, y={y_key!r})"
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            failed += 1
            print(f"[{index}/{total}] FAILED {json_path.name}: {error}")

    print(
        f"Batch complete: saved={saved}, skipped={skipped}, "
        f"failed={failed}, total={total} -> {output_dir}"
    )


def run_single(args: argparse.Namespace) -> None:
    if args.output_path is None:
        raise ValueError(
            "output_path is required for single-file mode. "
            "Example: python plot_pattern.py pattern.json output.png"
        )

    x_key, y_key = plot_json_to_png(
        args.json_path,
        args.output_path,
        **_plot_kwargs_from_args(args),
    )
    print(f"Saved digitizer-friendly plot to: {args.output_path}")
    print(f"Used x key: {x_key!r}; y key: {y_key!r}")
    if not args.no_normalize:
        print("Y values were normalized to relative intensity 0..1.")


def main() -> None:
    args = parse_args()
    batch_mode = args.batch or args.json_path.is_dir()

    if batch_mode:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"Error: {error}") from error
