#!/usr/bin/env python3
"""Generate a simulated XRD .xy file from visually extracted peaks.

Single-curve only. Input is a JSON list of peaks:
  [{"2theta": 10.5, "intensity": 1.0, "fwhm": 0.3}, ...]

or an object with a "peaks" array (extra keys ignored).

Usage:
    python digitize_plot.py peaks.json --output digitized.xy --min-x 5 --max-x 80

Requirements:
    - Conda environment: base-agent
    - numpy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def pseudo_voigt(x, xc, A, w, eta=0.5):
    """Pseudo-Voigt profile (Gaussian + Lorentzian mix)."""
    w_g = w / np.sqrt(2 * np.log(2))
    w_l = w

    gaussian = (
        (2 / w_g)
        * np.sqrt(np.log(2) / np.pi)
        * np.exp(-4 * np.log(2) * ((x - xc) / w_g) ** 2)
    )
    lorentzian = (2 / np.pi) * (w_l / (4 * (x - xc) ** 2 + w_l**2))

    return A * (eta * lorentzian + (1 - eta) * gaussian) * 1.5


def load_peaks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        peaks = data.get("peaks")
        if isinstance(peaks, list):
            return peaks
        # Nested single-curve skill schema: {"curves":[{"peaks":[...]}]}
        curves = data.get("curves")
        if isinstance(curves, list) and curves:
            nested = curves[0].get("peaks") if isinstance(curves[0], dict) else None
            if isinstance(nested, list):
                return nested
    raise ValueError(
        "JSON must be a peak list or an object with a 'peaks' array "
        f"(got {type(data).__name__})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Digitize XRD peaks to an .xy file")
    parser.add_argument(
        "input",
        help="JSON peak list or object with 'peaks': [{'2theta', 'intensity', 'fwhm'}, ...]",
    )
    parser.add_argument("--output", default="digitized.xy", help="Output .xy file path")
    parser.add_argument("--min-x", type=float, default=5.0, help="Minimum 2-theta")
    parser.add_argument("--max-x", type=float, default=90.0, help="Maximum 2-theta")
    parser.add_argument("--points", type=int, default=4000, help="Number of points")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Amplitude of experimental noise",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=0.05,
        help="Amplitude of exponential background baseline",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found {args.input}")
        sys.exit(1)

    try:
        peaks = load_peaks(args.input)
    except Exception as exc:
        print(f"Error reading JSON file {args.input}: {exc}")
        sys.exit(1)

    if args.points <= 1:
        print("Error: --points must be > 1")
        sys.exit(1)

    x = np.linspace(args.min_x, args.max_x, args.points)
    y = np.zeros_like(x)

    for peak in peaks:
        if not isinstance(peak, dict):
            print(f"Warning: Skipping invalid peak entry: {peak}")
            continue
        xc = peak.get("2theta", peak.get("two_theta"))
        intensity = peak.get("intensity")
        width = peak.get("fwhm", 0.3)
        eta = peak.get("eta", 0.5)
        if xc is None or intensity is None:
            print(
                f"Warning: Skipping invalid peak entry {peak}. "
                "Missing '2theta' or 'intensity'."
            )
            continue
        y += pseudo_voigt(x, float(xc), float(intensity), float(width), float(eta))

    if args.background > 0.0:
        background_curve = args.background * np.exp(-(x - args.min_x) / 10) + (
            args.background * 0.4
        )
        y += background_curve

    if args.noise > 0.0:
        rng = np.random.default_rng(42)
        y += rng.normal(0, args.noise, len(x))

    y = np.clip(y, 0, None)
    if np.max(y) > 0:
        y = y * (1000.0 / np.max(y))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savetxt(args.output, np.column_stack((x, y)), fmt="%.3f %.3f")
    print(f"Successfully generated digitized XY data at: {args.output}")

    plot_output = os.path.splitext(args.output)[0] + ".png"
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color="red", linewidth=1.5)
    for peak in peaks:
        if not isinstance(peak, dict):
            continue
        xc = peak.get("2theta", peak.get("two_theta"))
        name = peak.get("name")
        if xc is None or name is None:
            continue
        idx = int(np.abs(x - float(xc)).argmin())
        plt.text(
            float(xc),
            y[idx] + 20,
            str(name),
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=9,
        )
    plt.xlabel("2 theta (deg)")
    plt.ylabel("Intensity (counts)")
    plt.title("Digitized XRD Pattern")
    plt.xlim(args.min_x, args.max_x)
    plt.ylim(0, float(np.max(y)) * 1.2 if np.max(y) > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(plot_output, dpi=300)
    plt.close()
    print(f"Saved digitized plot image to: {plot_output}")

    try:
        from src.utils.config_utils import save_skill_inputs

        save_skill_inputs(args, args.output)
    except ImportError:
        pass
    except Exception as exc:
        print(f"Warning: Could not save skill input config: {exc}")


if __name__ == "__main__":
    main()
