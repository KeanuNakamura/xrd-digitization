"""Compute spectral information divergence between a JSON and CSV spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def spectral_information_divergence(
    true_intensity,
    approximate_intensity,
    epsilon=1e-12,
):
    """
    Compute directional and symmetric spectral information divergence.

    Returns:
        D(true || approximate),
        D(approximate || true),
        symmetric SID
    """
    true_intensity = np.asarray(true_intensity, dtype=float)
    approximate_intensity = np.asarray(approximate_intensity, dtype=float)

    if true_intensity.shape != approximate_intensity.shape:
        raise ValueError("The intensity arrays must have the same shape.")

    if np.any(true_intensity < 0) or np.any(approximate_intensity < 0):
        raise ValueError("SID requires nonnegative intensities.")

    if true_intensity.sum() <= 0 or approximate_intensity.sum() <= 0:
        raise ValueError("Each spectrum must have a positive total intensity.")

    # Convert the spectra into probability distributions.
    p = true_intensity / true_intensity.sum()
    q = approximate_intensity / approximate_intensity.sum()

    # Avoid log(0) and division by zero.
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    # Clipping slightly changes the sums, so normalize again.
    p /= p.sum()
    q /= q.sum()

    true_to_approx = np.sum(p * np.log(p / q))
    approx_to_true = np.sum(q * np.log(q / p))
    symmetric_sid = true_to_approx + approx_to_true

    return true_to_approx, approx_to_true, symmetric_sid


def load_true_spectrum(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    try:
        two_theta = np.asarray(data["two_theta_values"], dtype=float)
        intensity = np.asarray(data["intensities"], dtype=float)
    except KeyError as error:
        raise ValueError(
            f"Missing required JSON field: {error.args[0]}"
        ) from error

    if len(two_theta) != len(intensity):
        raise ValueError(
            "JSON two_theta_values and intensities must have equal lengths."
        )

    return two_theta, intensity


def load_approximate_spectrum(csv_path):
    """
    Loads a headerless, whitespace-separated CSV containing:

        two_theta intensity
    """
    data = pd.read_csv(
        csv_path,
        sep=r"\s+|,",
        engine="python",
        comment="#",
        header=None,
        names=["two_theta", "intensity"],
    )

    # Remove rows that cannot be interpreted as numbers, such as a header row.
    data["two_theta"] = pd.to_numeric(data["two_theta"], errors="coerce")
    data["intensity"] = pd.to_numeric(data["intensity"], errors="coerce")
    data = data.dropna(subset=["two_theta", "intensity"])

    if data.empty:
        raise ValueError(
            "The CSV does not contain valid two_theta and intensity values."
        )

    data = data.sort_values("two_theta")

    return (
        data["two_theta"].to_numpy(dtype=float),
        data["intensity"].to_numpy(dtype=float),
    )


def compare_spectra(
    json_path: Path,
    csv_path: Path,
    *,
    shift: float = 0.0,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compare a true JSON spectrum to an approximated CSV spectrum.

    Returns a dict with SID values and the overlapping comparison arrays.
    """
    true_x, true_y = load_true_spectrum(json_path)
    approx_x, approx_y = load_approximate_spectrum(csv_path)

    approx_x = approx_x + shift

    overlap_mask = (true_x >= approx_x.min()) & (true_x <= approx_x.max())
    if not np.any(overlap_mask):
        raise ValueError(
            "The spectra have no overlapping two-theta range after shifting."
        )

    comparison_x = true_x[overlap_mask]
    comparison_true_y = true_y[overlap_mask]
    comparison_approx_y = np.interp(comparison_x, approx_x, approx_y)

    forward, reverse, sid = spectral_information_divergence(
        comparison_true_y,
        comparison_approx_y,
        epsilon=epsilon,
    )

    return {
        "forward": float(forward),
        "reverse": float(reverse),
        "symmetric_sid": float(sid),
        "shift": float(shift),
        "comparison_x": comparison_x,
        "comparison_true_y": comparison_true_y,
        "comparison_approx_y": comparison_approx_y,
        "true_x": true_x,
        "true_y": true_y,
        "approx_x": approx_x,
        "approx_y": approx_y,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute spectral information divergence between a true "
            "JSON spectrum and an approximated CSV spectrum."
        )
    )

    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the JSON file containing the true spectrum.",
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the CSV file containing the approximated spectrum.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help=(
            "Horizontal shift in degrees applied to the CSV two-theta "
            "values before interpolation. Default: 0."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Small value used to avoid log(0). Default: 1e-12.",
    )

    args = parser.parse_args()

    if not args.json_file.is_file():
        parser.error(f"JSON file does not exist: {args.json_file}")

    if not args.csv_file.is_file():
        parser.error(f"CSV file does not exist: {args.csv_file}")

    result = compare_spectra(
        args.json_file,
        args.csv_file,
        shift=args.shift,
        epsilon=args.epsilon,
    )

    comparison_x = result["comparison_x"]
    print(f"JSON file: {args.json_file}")
    print(f"CSV file: {args.csv_file}")
    print(f"Applied CSV shift: {result['shift']:.10g} degrees")
    print(
        f"Compared two-theta range: "
        f"{comparison_x.min():.6f} to {comparison_x.max():.6f}"
    )
    print(f"Number of comparison points: {len(comparison_x)}")
    print(f"D(true || approximate): {result['forward']:.10f}")
    print(f"D(approximate || true): {result['reverse']:.10f}")
    print(f"Symmetric SID: {result['symmetric_sid']:.10f}")


if __name__ == "__main__":
    main()
