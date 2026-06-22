from __future__ import annotations

"""
Generates simulated XRD .xy data from manually extracted peaks.

Supports two JSON formats:

1. Legacy single-curve format:
[
  {"2theta": 10.5, "intensity": 1.0, "fwhm": 0.3},
  {"2theta": 20.1, "intensity": 0.6, "fwhm": 0.3}
]

2. Multi-curve format:
{
  "source_image": "figure_1.png",
  "curve_layout": "stacked",
  "x_axis": {
    "label": "2theta",
    "unit": "degrees",
    "min": 5.0,
    "max": 80.0
  },
  "y_axis": {
    "label": "intensity",
    "unit": "counts_per_pixel",
    "min": 0.0,
    "max": 10000.0
  },
  "curves": [
    {
      "curve_id": "curve_1",
      "label": "Precursor",
      "color": "blue",
      "position": "bottom",
      "baseline_offset": 500.0,
      "intensity_scale": 1250.0,
      "peaks": [
        {"2theta": 10.5, "intensity": 1.0, "fwhm": 0.3}
      ]
    }
  ]
}

curve_layout values:
  - "stacked": vertically offset curves (common in multi-pattern XRD figures)
  - "overlay": curves share the same baseline (e.g. Rietveld observed/calc/diff)
  - "auto": infer from metadata (default)

Usage:
    python digitize_plot.py peaks.json --output digitized.xy --min-x 5 --max-x 80 --points 4000

Requirements:
    - Conda environment: base-agent
    - Required packages: numpy, matplotlib
"""

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

POSITION_ORDER = {"bottom": 0, "middle": 1, "top": 2}
COLOR_MAP = {
    "black": "black",
    "blue": "tab:blue",
    "red": "tab:red",
    "green": "tab:green",
    "orange": "tab:orange",
    "purple": "tab:purple",
    "brown": "tab:brown",
    "gray": "gray",
    "grey": "gray",
}


def pseudo_voigt(x, xc, A, w, eta=0.5):
    """
    Pseudo-Voigt profile generator.

    x: independent variable
    xc: peak center
    A: area/intensity
    w: FWHM
    eta: Lorentzian fraction, from 0 to 1
    """
    if w <= 0:
        w = 0.3

    w_g = w / np.sqrt(2 * np.log(2))
    w_l = w

    gaussian = (
        (2 / w_g)
        * np.sqrt(np.log(2) / np.pi)
        * np.exp(-4 * np.log(2) * ((x - xc) / w_g) ** 2)
    )

    lorentzian = (2 / np.pi) * (w_l / (4 * (x - xc) ** 2 + w_l**2))

    return A * (eta * lorentzian + (1 - eta) * gaussian) * 1.5


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        print(f"Error: Input file not found: {path}")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {path}: {e}")
        sys.exit(1)


def normalize_input(data: Any) -> dict[str, Any]:
    """
    Convert either supported JSON format into a common internal format.
    """
    if isinstance(data, list):
        return {
            "source_image": None,
            "curve_layout": "overlay",
            "x_axis": {},
            "y_axis": {},
            "curves": [
                {
                    "curve_id": "curve_1",
                    "label": None,
                    "color": None,
                    "position": None,
                    "baseline_offset": None,
                    "intensity_scale": None,
                    "peaks": data,
                }
            ],
        }

    if isinstance(data, dict):
        curves = data.get("curves")

        if isinstance(curves, list):
            normalized_curves = []

            for i, curve in enumerate(curves, start=1):
                if not isinstance(curve, dict):
                    print(f"Warning: Skipping invalid curve entry: {curve}")
                    continue

                peaks = curve.get("peaks", [])
                if not isinstance(peaks, list):
                    print(
                        f"Warning: Curve {curve.get('curve_id', i)} has invalid peaks field. Skipping."
                    )
                    continue

                normalized_curves.append(
                    {
                        "curve_id": curve.get("curve_id") or f"curve_{i}",
                        "label": curve.get("label"),
                        "color": curve.get("color"),
                        "position": curve.get("position"),
                        "baseline_offset": curve.get("baseline_offset"),
                        "intensity_scale": curve.get("intensity_scale"),
                        "peaks": peaks,
                    }
                )

            return {
                "source_image": data.get("source_image"),
                "curve_layout": data.get("curve_layout", "auto"),
                "x_axis": data.get("x_axis", {}),
                "y_axis": data.get("y_axis", {}),
                "curves": normalized_curves,
            }

        peaks = data.get("peaks")
        if isinstance(peaks, list):
            return {
                "source_image": data.get("source_image"),
                "curve_layout": data.get("curve_layout", "overlay"),
                "x_axis": data.get("x_axis", {}),
                "y_axis": data.get("y_axis", {}),
                "curves": [
                    {
                        "curve_id": data.get("curve_id", "curve_1"),
                        "label": data.get("label"),
                        "color": data.get("color"),
                        "position": data.get("position"),
                        "baseline_offset": data.get("baseline_offset"),
                        "intensity_scale": data.get("intensity_scale"),
                        "peaks": peaks,
                    }
                ],
            }

    print(
        "Error: Unsupported JSON format. Expected a list of peaks or a dict with a 'curves' field."
    )
    sys.exit(1)


def sort_curves_by_position(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(curve: dict[str, Any]) -> tuple[int, str]:
        position = curve.get("position")
        rank = POSITION_ORDER.get(position, 99)
        return (rank, curve.get("curve_id", ""))

    return sorted(curves, key=sort_key)


def detect_curve_layout(
    normalized: dict[str, Any],
    curves: list[dict[str, Any]],
    cli_layout: str | None = None,
) -> str:
    if cli_layout and cli_layout != "auto":
        return cli_layout

    layout = normalized.get("curve_layout", "auto")
    if layout in ("stacked", "overlay"):
        return layout

    if len(curves) <= 1:
        return "overlay"

    has_position = any(
        curve.get("position") in POSITION_ORDER for curve in curves
    )
    has_baseline = any(
        curve.get("baseline_offset") is not None for curve in curves
    )

    if has_baseline or has_position:
        return "stacked"

    return "overlay"


def default_intensity_scale(
    curve: dict[str, Any],
    y_axis: dict[str, Any],
    layout: str,
) -> float:
    if curve.get("intensity_scale") is not None:
        return float(curve["intensity_scale"])

    y_max = y_axis.get("max")
    if layout == "stacked" and isinstance(y_max, (int, float)) and y_max <= 2.0:
        return 1.0

    if layout == "stacked":
        return 1000.0

    return 1000.0


def get_x_range(args, normalized: dict[str, Any]) -> tuple[float, float]:
    min_x = args.min_x
    max_x = args.max_x

    x_axis = normalized.get("x_axis") or {}

    json_min = x_axis.get("min")
    json_max = x_axis.get("max")

    user_set_min = "--min-x" in sys.argv
    user_set_max = "--max-x" in sys.argv

    if not user_set_min and isinstance(json_min, (int, float)):
        min_x = float(json_min)

    if not user_set_max and isinstance(json_max, (int, float)):
        max_x = float(json_max)

    if min_x >= max_x:
        print(f"Error: Invalid x range: min_x={min_x}, max_x={max_x}")
        sys.exit(1)

    return min_x, max_x


def simulate_curve(
    x: np.ndarray,
    peaks: list[dict[str, Any]],
    background: float,
    noise: float,
    seed: int,
    intensity_scale: float,
) -> np.ndarray:
    y = np.zeros_like(x)

    for peak in peaks:
        if not isinstance(peak, dict):
            print(f"Warning: Skipping invalid peak entry: {peak}")
            continue

        xc = peak.get("2theta")
        A = peak.get("intensity")
        w = peak.get("fwhm", 0.3)
        eta = peak.get("eta", 0.5)

        if xc is None or A is None:
            print(
                f"Warning: Skipping invalid peak entry {peak}. Missing '2theta' or 'intensity'."
            )
            continue

        try:
            xc = float(xc)
            A = float(A)
            w = float(w)
            eta = float(eta)
        except ValueError:
            print(f"Warning: Skipping peak with non-numeric values: {peak}")
            continue

        y += pseudo_voigt(x, xc, A, w, eta)

    if background > 0.0:
        background_curve = background * np.exp(-(x - x[0]) / 10) + (background * 0.4)
        y += background_curve

    if noise > 0.0:
        rng = np.random.default_rng(seed)
        noise_curve = rng.normal(0, noise, len(x))
        y += noise_curve

    y = np.clip(y, 0, None)

    peak_max = float(np.max(y))
    if peak_max > 0:
        y = y * (intensity_scale / peak_max)

    return y


def apply_stacked_offsets(
    curves: list[dict[str, Any]],
    y_by_curve: dict[str, np.ndarray],
    y_axis: dict[str, Any],
    stack_gap: float,
) -> dict[str, np.ndarray]:
    ordered = sort_curves_by_position(curves)
    explicit_offsets = [
        curve.get("baseline_offset") for curve in ordered
    ]

    if all(offset is not None for offset in explicit_offsets):
        for curve in ordered:
            curve_id = curve["curve_id"]
            y_by_curve[curve_id] = y_by_curve[curve_id] + float(
                curve["baseline_offset"]
            )
        return y_by_curve

    if any(offset is not None for offset in explicit_offsets):
        print(
            "Warning: Only some curves define baseline_offset. "
            "Provide baseline_offset for every curve in stacked layouts."
        )

    current_offset = float(y_axis.get("min", 0) or 0)
    for curve in ordered:
        curve_id = curve["curve_id"]
        y = y_by_curve[curve_id]

        if curve.get("baseline_offset") is not None:
            current_offset = float(curve["baseline_offset"])

        y_by_curve[curve_id] = y + current_offset
        current_offset += float(np.max(y)) * (1.0 + stack_gap)

    return y_by_curve


def resolve_curve_color(color: str | None) -> str | None:
    if not color:
        return None
    return COLOR_MAP.get(color.lower(), color)


def write_single_curve_xy(output_path: str, x: np.ndarray, y: np.ndarray) -> None:
    np.savetxt(output_path, np.column_stack((x, y)), fmt="%.3f %.3f")


def write_multi_curve_xy(
    output_path: str,
    source_image: str | None,
    curves: list[dict[str, Any]],
    x: np.ndarray,
    y_by_curve: dict[str, np.ndarray],
    layout: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for curve in curves:
            curve_id = curve["curve_id"]
            label = curve.get("label")
            color = curve.get("color")
            position = curve.get("position")
            baseline_offset = curve.get("baseline_offset")
            y = y_by_curve[curve_id]

            f.write(f"# source_image: {source_image or ''}\n")
            f.write(f"# curve_id: {curve_id}\n")
            f.write(f"# curve_layout: {layout}\n")

            if label:
                f.write(f"# label: {label}\n")
            if color:
                f.write(f"# color: {color}\n")
            if position:
                f.write(f"# position: {position}\n")
            if baseline_offset is not None:
                f.write(f"# baseline_offset: {baseline_offset}\n")

            f.write("# columns: 2theta intensity\n")

            for xi, yi in zip(x, y):
                f.write(f"{xi:.3f} {yi:.3f}\n")

            f.write("\n")


def save_plot(
    plot_output: str,
    x: np.ndarray,
    curves: list[dict[str, Any]],
    y_by_curve: dict[str, np.ndarray],
    min_x: float,
    max_x: float,
    layout: str,
    y_axis: dict[str, Any],
) -> None:
    fig_height = 5 if layout == "overlay" else max(5, 2 + len(curves) * 1.5)
    plt.figure(figsize=(10, fig_height))

    global_min_y = 0.0
    global_max_y = 0.0

    for curve in curves:
        curve_id = curve["curve_id"]
        label = curve.get("label") or curve_id
        y = y_by_curve[curve_id]
        color = resolve_curve_color(curve.get("color"))

        global_min_y = min(global_min_y, float(np.min(y)))
        global_max_y = max(global_max_y, float(np.max(y)))

        plot_kwargs = {"linewidth": 1.5, "label": label}
        if color:
            plot_kwargs["color"] = color

        plt.plot(x, y, **plot_kwargs)

        for peak in curve.get("peaks", []):
            if not isinstance(peak, dict):
                continue

            xc = peak.get("2theta")
            name = peak.get("name")

            if xc is not None and name is not None:
                try:
                    xc = float(xc)
                except ValueError:
                    continue

                idx = np.abs(x - xc).argmin()
                peak_y = y[idx]
                text_offset = (global_max_y - global_min_y) * 0.02

                plt.text(
                    xc,
                    peak_y + text_offset,
                    name,
                    rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=8,
                )

    y_label = y_axis.get("label", "intensity")
    y_unit = y_axis.get("unit")
    if y_unit:
        y_label = f"{y_label} ({y_unit})"

    plt.xlabel("2 theta (deg)")
    plt.ylabel(y_label)
    plt.title("Digitized XRD Pattern")
    plt.xlim(min_x, max_x)

    if global_max_y > global_min_y:
        span = global_max_y - global_min_y
        plt.ylim(global_min_y - 0.05 * span, global_max_y + 0.10 * span)

    if len(curves) > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig(plot_output, dpi=300)
    plt.close()


def save_skill_inputs_safely(args, output_path: str) -> None:
    try:
        from src.utils.config_utils import save_skill_inputs

        save_skill_inputs(args, output_path)
    except ImportError:
        print("Note: src.utils.config_utils not found. Skipping skill input config save.")
    except Exception as e:
        print(f"Warning: Could not save skill input config: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Digitize XRD peaks to an .xy file. Supports single-curve and multi-curve JSON."
    )

    parser.add_argument(
        "input",
        help=(
            "JSON file containing either a list of peaks or a multi-curve object "
            "with {'curves': [{'curve_id': ..., 'peaks': [...]}]}"
        ),
    )

    parser.add_argument("--output", default="digitized.xy", help="Output .xy file path")

    parser.add_argument(
        "--min-x", type=float, default=5.0, help="Minimum 2-theta value"
    )

    parser.add_argument(
        "--max-x", type=float, default=90.0, help="Maximum 2-theta value"
    )

    parser.add_argument(
        "--points", type=int, default=4000, help="Number of data points"
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Amplitude of experimental noise to add",
    )

    parser.add_argument(
        "--background",
        type=float,
        default=0.05,
        help="Amplitude of exponential background to add",
    )

    parser.add_argument(
        "--layout",
        choices=["auto", "stacked", "overlay"],
        default="auto",
        help="Curve layout: stacked (vertical offsets), overlay (shared baseline), or auto",
    )

    parser.add_argument(
        "--stack-gap",
        type=float,
        default=0.15,
        help=(
            "Fractional gap between stacked curves when baseline_offset is not provided "
            "(default: 0.15)"
        ),
    )

    args = parser.parse_args()

    raw_data = load_json(args.input)
    normalized = normalize_input(raw_data)

    curves = normalized["curves"]

    if not curves:
        print("Error: No valid curves found in input JSON.")
        sys.exit(1)

    layout = detect_curve_layout(normalized, curves, cli_layout=args.layout)
    y_axis = normalized.get("y_axis") or {}

    min_x, max_x = get_x_range(args, normalized)

    if args.points <= 1:
        print("Error: --points must be greater than 1.")
        sys.exit(1)

    x = np.linspace(min_x, max_x, args.points)

    y_by_curve = {}

    for i, curve in enumerate(curves):
        curve_id = curve["curve_id"]
        peaks = curve.get("peaks", [])

        if not peaks:
            print(
                f"Warning: Curve {curve_id} has no peaks. Output will contain baseline/noise only."
            )

        intensity_scale = default_intensity_scale(curve, y_axis, layout)

        y = simulate_curve(
            x=x,
            peaks=peaks,
            background=args.background,
            noise=args.noise,
            seed=42 + i,
            intensity_scale=intensity_scale,
        )

        y_by_curve[curve_id] = y

    if layout == "stacked" and len(curves) > 1:
        y_by_curve = apply_stacked_offsets(
            curves=curves,
            y_by_curve=y_by_curve,
            y_axis=y_axis,
            stack_gap=args.stack_gap,
        )
        print(f"Applied stacked layout with vertical offsets for {len(curves)} curves.")
    elif len(curves) > 1:
        print(f"Applied overlay layout for {len(curves)} curves.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if len(curves) == 1:
        curve_id = curves[0]["curve_id"]
        write_single_curve_xy(args.output, x, y_by_curve[curve_id])
    else:
        write_multi_curve_xy(
            output_path=args.output,
            source_image=normalized.get("source_image"),
            curves=curves,
            x=x,
            y_by_curve=y_by_curve,
            layout=layout,
        )

    print(f"Successfully generated digitized XY data at: {args.output}")

    plot_output = os.path.splitext(args.output)[0] + ".png"

    save_plot(
        plot_output=plot_output,
        x=x,
        curves=curves,
        y_by_curve=y_by_curve,
        min_x=min_x,
        max_x=max_x,
        layout=layout,
        y_axis=y_axis,
    )

    print(f"Saved digitized plot image to: {plot_output}")

    save_skill_inputs_safely(args, args.output)


if __name__ == "__main__":
    main()
