from __future__ import annotations

"""
Generates simulated XRD .xy data from manually extracted peaks.

Supports three JSON formats:

1. Legacy single-curve format:
[
  {"2theta": 10.5, "intensity": 1.0, "fwhm": 0.3},
  {"2theta": 20.1, "intensity": 0.6, "fwhm": 0.3}
]

2. Single-plot multi-curve format:
{
  "source_image": "figure_1.png",
  "curve_layout": "stacked",
  "x_axis": {"label": "2theta", "unit": "degrees", "min": 5.0, "max": 80.0},
  "y_axis": {"label": "intensity", "unit": "counts_per_pixel", "min": 0.0, "max": 10000.0},
  "curves": [
    {
      "curve_id": "curve_1",
      "label": "Precursor",
      "color": "blue",
      "position": "bottom",
      "baseline_offset": 500.0,
      "intensity_scale": 1250.0,
      "peaks": [{"2theta": 10.5, "intensity": 1.0, "fwhm": 0.3}]
    }
  ]
}

3. Multi-panel format (separate subplots in one figure image):
{
  "source_image": "figure_7.png",
  "figure_layout": "multi_panel",
  "plots": [
    {
      "plot_id": "plot_1",
      "label": "ICSD icsd_156218",
      "position": "top",
      "curve_layout": "overlay",
      "x_axis": {"label": "2theta", "unit": "degrees", "min": 0.0, "max": 45.0},
      "y_axis": {"label": "intensity", "unit": "normalized", "min": 0.0, "max": 1.0},
      "curves": [{"curve_id": "curve_1", "label": "ICSD icsd_156218", "color": "orange", "peaks": [...]}]
    },
    {
      "plot_id": "plot_2",
      "label": "RRUFF R060558",
      "position": "bottom",
      "curve_layout": "overlay",
      "x_axis": {"label": "2theta", "unit": "degrees", "min": 0.0, "max": 45.0},
      "y_axis": {"label": "intensity", "unit": "normalized", "min": 0.0, "max": 1.0},
      "curves": [{"curve_id": "curve_1", "label": "RRUFF R060558", "color": "blue", "peaks": [...]}]
    }
  ]
}

curve_layout values:
  - "stacked": vertically offset curves within one plot
  - "overlay": curves share the same baseline within one plot
  - "auto": infer from metadata (default)

Multi-panel output naming (from --output figure_7/figure_7_digitized.xy):
  - figure_7_digitized_1.xy, figure_7_digitized_1.png
  - figure_7_digitized_2.xy, figure_7_digitized_2.png
  - figure_7_1.json, figure_7_2.json (single-plot JSON per panel)

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


def normalize_curves(curves: list[Any]) -> list[dict[str, Any]]:
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

    return normalized_curves


def normalize_single_plot(
    data: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = defaults or {}
    curves = data.get("curves")

    if not isinstance(curves, list):
        peaks = data.get("peaks")
        if isinstance(peaks, list):
            curves = [
                {
                    "curve_id": data.get("curve_id", "curve_1"),
                    "label": data.get("label"),
                    "color": data.get("color"),
                    "position": data.get("position"),
                    "baseline_offset": data.get("baseline_offset"),
                    "intensity_scale": data.get("intensity_scale"),
                    "peaks": peaks,
                }
            ]
        else:
            curves = []

    return {
        "plot_id": data.get("plot_id"),
        "label": data.get("label"),
        "position": data.get("position"),
        "source_image": data.get("source_image", defaults.get("source_image")),
        "figure_type": data.get("figure_type", defaults.get("figure_type")),
        "curve_layout": data.get("curve_layout", defaults.get("curve_layout", "auto")),
        "x_axis": data.get("x_axis", defaults.get("x_axis", {})),
        "y_axis": data.get("y_axis", defaults.get("y_axis", {})),
        "notes": data.get("notes", defaults.get("notes")),
        "noise": data.get("noise", defaults.get("noise")),
        "background": data.get("background", defaults.get("background")),
        "curves": normalize_curves(curves),
    }


def normalize_input(data: Any) -> dict[str, Any]:
    """
    Convert supported JSON formats into a common internal format.

    Returns either:
      {"mode": "single", ...plot fields...}
      {"mode": "multi_panel", "source_image": ..., "plots": [...]}
    """
    if isinstance(data, list):
        plot = normalize_single_plot({"curves": [{"curve_id": "curve_1", "peaks": data}]})
        plot["curve_layout"] = "overlay"
        return {"mode": "single", **plot}

    if not isinstance(data, dict):
        print(
            "Error: Unsupported JSON format. Expected a list of peaks or a JSON object."
        )
        sys.exit(1)

    plots = data.get("plots")
    if isinstance(plots, list) and plots:
        defaults = {
            "source_image": data.get("source_image"),
            "figure_type": data.get("figure_type"),
            "notes": data.get("notes"),
            "x_axis": data.get("x_axis", {}),
            "y_axis": data.get("y_axis", {}),
            "curve_layout": data.get("curve_layout", "auto"),
        }

        normalized_plots = []
        for i, plot in enumerate(plots, start=1):
            if not isinstance(plot, dict):
                print(f"Warning: Skipping invalid plot entry: {plot}")
                continue

            normalized = normalize_single_plot(plot, defaults=defaults)
            if not normalized.get("plot_id"):
                normalized["plot_id"] = f"plot_{i}"
            normalized_plots.append(normalized)

        if not normalized_plots:
            print("Error: No valid plots found in input JSON.")
            sys.exit(1)

        return {
            "mode": "multi_panel",
            "source_image": data.get("source_image"),
            "figure_type": data.get("figure_type"),
            "figure_layout": data.get("figure_layout", "multi_panel"),
            "notes": data.get("notes"),
            "plots": normalized_plots,
        }

    if isinstance(data.get("curves"), list) or isinstance(data.get("peaks"), list):
        plot = normalize_single_plot(data)
        return {"mode": "single", **plot}

    print(
        "Error: Unsupported JSON format. Expected a list of peaks, a dict with "
        "'curves', or a dict with 'plots'."
    )
    sys.exit(1)


def sort_curves_by_position(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(curve: dict[str, Any]) -> tuple[int, str]:
        position = curve.get("position")
        rank = POSITION_ORDER.get(position, 99)
        return (rank, curve.get("curve_id", ""))

    return sorted(curves, key=sort_key)


def detect_curve_layout(
    plot: dict[str, Any],
    curves: list[dict[str, Any]],
    cli_layout: str | None = None,
) -> str:
    if cli_layout and cli_layout != "auto":
        return cli_layout

    layout = plot.get("curve_layout", "auto")
    if layout in ("stacked", "overlay"):
        return layout

    if len(curves) <= 1:
        return "overlay"

    has_position = any(curve.get("position") in POSITION_ORDER for curve in curves)
    has_baseline = any(curve.get("baseline_offset") is not None for curve in curves)

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
    y_unit = (y_axis.get("unit") or "").lower()

    if isinstance(y_max, (int, float)) and y_max <= 2.0:
        return 1.0

    if y_unit in ("normalized", "arbitrary", "norm"):
        return 1.0

    if layout == "stacked":
        return 1000.0

    return 1000.0


def resolve_plot_simulation_params(
    plot: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, float]:
    noise = plot.get("noise")
    background = plot.get("background")

    if noise is None:
        noise = args.noise
    if background is None:
        background = args.background

    return float(noise), float(background)


def get_x_range(
    args: argparse.Namespace,
    plot: dict[str, Any],
    user_set_min: bool,
    user_set_max: bool,
) -> tuple[float, float]:
    min_x = args.min_x
    max_x = args.max_x

    x_axis = plot.get("x_axis") or {}
    json_min = x_axis.get("min")
    json_max = x_axis.get("max")

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
    explicit_offsets = [curve.get("baseline_offset") for curve in ordered]

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
    plot_id: str | None = None,
    plot_label: str | None = None,
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
            if plot_id:
                f.write(f"# plot_id: {plot_id}\n")
            if plot_label:
                f.write(f"# plot_label: {plot_label}\n")
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
    title: str | None = None,
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
    plt.title(title or "Digitized XRD Pattern")
    plt.xlim(min_x, max_x)

    if global_max_y > global_min_y:
        span = global_max_y - global_min_y
        plt.ylim(global_min_y - 0.05 * span, global_max_y + 0.10 * span)

    if len(curves) > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig(plot_output, dpi=300)
    plt.close()


def numbered_output_path(base_output: str, index: int) -> str:
    root, ext = os.path.splitext(base_output)
    return f"{root}_{index}{ext}"


def plot_json_output_path(input_json: str, index: int) -> str:
    directory = os.path.dirname(os.path.abspath(input_json))
    stem = os.path.splitext(os.path.basename(input_json))[0]
    return os.path.join(directory, f"{stem}_{index}.json")


def plot_to_export_json(plot: dict[str, Any]) -> dict[str, Any]:
    export = {
        "source_image": plot.get("source_image"),
        "figure_type": plot.get("figure_type", "xrd"),
        "curve_layout": plot.get("curve_layout", "auto"),
        "x_axis": plot.get("x_axis", {}),
        "y_axis": plot.get("y_axis", {}),
        "curves": plot.get("curves", []),
    }

    if plot.get("plot_id"):
        export["plot_id"] = plot["plot_id"]
    if plot.get("label"):
        export["label"] = plot["label"]
    if plot.get("position"):
        export["position"] = plot["position"]
    if plot.get("notes"):
        export["notes"] = plot["notes"]
    if plot.get("noise") is not None:
        export["noise"] = plot["noise"]
    if plot.get("background") is not None:
        export["background"] = plot["background"]

    return export


def process_single_plot(
    plot: dict[str, Any],
    args: argparse.Namespace,
    output_xy: str,
    user_set_min: bool,
    user_set_max: bool,
    seed_offset: int = 0,
    write_plot_json: bool = True,
    plot_json_path: str | None = None,
) -> None:
    curves = plot["curves"]

    if not curves:
        plot_name = plot.get("plot_id") or plot.get("label") or "plot"
        print(f"Error: No valid curves found in {plot_name}.")
        sys.exit(1)

    layout = detect_curve_layout(plot, curves, cli_layout=args.layout)
    y_axis = plot.get("y_axis") or {}
    min_x, max_x = get_x_range(args, plot, user_set_min, user_set_max)

    if args.points <= 1:
        print("Error: --points must be greater than 1.")
        sys.exit(1)

    x = np.linspace(min_x, max_x, args.points)
    y_by_curve = {}
    noise, background = resolve_plot_simulation_params(plot, args)

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
            background=background,
            noise=noise,
            seed=42 + seed_offset + i,
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
        print(
            f"Applied stacked layout with vertical offsets for {len(curves)} curves."
        )
    elif len(curves) > 1:
        print(f"Applied overlay layout for {len(curves)} curves.")

    os.makedirs(os.path.dirname(os.path.abspath(output_xy)), exist_ok=True)

    if len(curves) == 1:
        curve_id = curves[0]["curve_id"]
        write_single_curve_xy(output_xy, x, y_by_curve[curve_id])
    else:
        write_multi_curve_xy(
            output_path=output_xy,
            source_image=plot.get("source_image"),
            curves=curves,
            x=x,
            y_by_curve=y_by_curve,
            layout=layout,
            plot_id=plot.get("plot_id"),
            plot_label=plot.get("label"),
        )

    print(f"Successfully generated digitized XY data at: {output_xy}")

    plot_output = os.path.splitext(output_xy)[0] + ".png"
    title = plot.get("label") or plot.get("plot_id") or "Digitized XRD Pattern"

    save_plot(
        plot_output=plot_output,
        x=x,
        curves=curves,
        y_by_curve=y_by_curve,
        min_x=min_x,
        max_x=max_x,
        layout=layout,
        y_axis=y_axis,
        title=title,
    )

    print(f"Saved digitized plot image to: {plot_output}")

    if write_plot_json and plot_json_path:
        export = plot_to_export_json(plot)
        export["curve_layout"] = layout
        with open(plot_json_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
        print(f"Saved per-plot JSON to: {plot_json_path}")


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
        description=(
            "Digitize XRD peaks to .xy files and preview PNGs. Supports single-plot, "
            "multi-curve, and multi-panel JSON."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "JSON file containing peaks, a multi-curve object with 'curves', or a "
            "multi-panel object with 'plots'."
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

    parser.add_argument(
        "--write-plot-jsons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For multi-panel figures, also write one single-plot JSON file per panel "
            "(default: enabled)"
        ),
    )

    args = parser.parse_args()

    raw_data = load_json(args.input)
    normalized = normalize_input(raw_data)

    user_set_min = "--min-x" in sys.argv
    user_set_max = "--max-x" in sys.argv

    if normalized["mode"] == "multi_panel":
        plots = normalized["plots"]
        print(
            f"Detected multi-panel figure with {len(plots)} plots. "
            "Writing numbered outputs per panel."
        )

        for index, plot in enumerate(plots, start=1):
            if not plot.get("source_image"):
                plot["source_image"] = normalized.get("source_image")
            if not plot.get("figure_type"):
                plot["figure_type"] = normalized.get("figure_type", "xrd")

            plot_label = plot.get("label") or plot.get("plot_id") or f"plot_{index}"
            print(f"\nProcessing panel {index}: {plot_label}")

            output_xy = numbered_output_path(args.output, index)
            plot_json_path = (
                plot_json_output_path(args.input, index)
                if args.write_plot_jsons
                else None
            )

            process_single_plot(
                plot=plot,
                args=args,
                output_xy=output_xy,
                user_set_min=user_set_min,
                user_set_max=user_set_max,
                seed_offset=(index - 1) * 100,
                write_plot_json=args.write_plot_jsons,
                plot_json_path=plot_json_path,
            )

        save_skill_inputs_safely(args, args.output)
        return

    plot = {key: value for key, value in normalized.items() if key != "mode"}

    process_single_plot(
        plot=plot,
        args=args,
        output_xy=args.output,
        user_set_min=user_set_min,
        user_set_max=user_set_max,
        write_plot_json=False,
    )

    save_skill_inputs_safely(args, args.output)


if __name__ == "__main__":
    main()
