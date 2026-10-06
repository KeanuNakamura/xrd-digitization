"""Compute spectral information divergence between a JSON and CSV spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_GAUSSIAN_SIGMA_DEG = 0.08
DEFAULT_MODIFIED_EPSILON = 1e-6
DEFAULT_PEAK_PROMINENCE = 0.05
DEFAULT_PEAK_TOLERANCE_DEG = 0.25

SCORE_SUMMARY_KEYS = (
    "raw_sid",
    "modified_sid",
    "peak_recall",
    "peak_precision",
    "peak_f1",
    "peak_penalty",
    "final_xrd_score",
    "number_true_peaks",
    "number_digitized_peaks",
    "number_matched_peaks",
    "true_peak_positions",
    "digitized_peak_positions",
    "matched_pairs",
    "D_true_approx_modified",
    "D_approx_true_modified",
    "comparison_x_start",
    "comparison_x_end",
    "peak_prominence",
    "peak_tolerance_deg",
    "gaussian_sigma_deg",
)


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


def _gaussian_smooth_1d(y: np.ndarray, x: np.ndarray, sigma_deg: float) -> np.ndarray:
    """Smooth intensities with a Gaussian whose sigma is in degrees 2θ."""
    y = np.asarray(y, dtype=float)
    if sigma_deg <= 0 or len(y) < 3:
        return y.copy()

    dx = float(np.median(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        return y.copy()

    sigma_samples = sigma_deg / dx
    if sigma_samples < 1e-6:
        return y.copy()

    radius = int(max(1, np.ceil(3.0 * sigma_samples)))
    t = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        return y.copy()
    kernel /= kernel_sum
    return np.convolve(y, kernel, mode="same")


def _to_probability(intensity: np.ndarray, epsilon: float) -> np.ndarray:
    """Clip nonnegative, add epsilon, normalize to a probability distribution."""
    values = np.clip(np.asarray(intensity, dtype=float), 0.0, None) + float(epsilon)
    total = values.sum()
    if total <= 0:
        raise ValueError("Spectrum has non-positive total intensity after clipping.")
    return values / total


def modified_spectral_information_divergence(
    true_x: np.ndarray,
    true_y: np.ndarray,
    approx_x: np.ndarray,
    approx_y: np.ndarray,
    *,
    gaussian_sigma_deg: float = DEFAULT_GAUSSIAN_SIGMA_DEG,
    epsilon: float = DEFAULT_MODIFIED_EPSILON,
) -> dict[str, Any]:
    """XRD-aware SID over the shared x-range with light Gaussian smoothing."""
    true_x = np.asarray(true_x, dtype=float)
    true_y = np.asarray(true_y, dtype=float)
    approx_x = np.asarray(approx_x, dtype=float)
    approx_y = np.asarray(approx_y, dtype=float)

    x_start = float(max(true_x.min(), approx_x.min()))
    x_end = float(min(true_x.max(), approx_x.max()))
    if not np.isfinite(x_start) or not np.isfinite(x_end) or x_end <= x_start:
        raise ValueError("The spectra have no overlapping two-theta range.")

    n_true = int(np.count_nonzero((true_x >= x_start) & (true_x <= x_end)))
    n_approx = int(np.count_nonzero((approx_x >= x_start) & (approx_x <= x_end)))
    n_points = max(n_true, n_approx, 2)
    grid = np.linspace(x_start, x_end, n_points)

    true_on_grid = np.interp(grid, true_x, true_y)
    approx_on_grid = np.interp(grid, approx_x, approx_y)

    true_smooth = _gaussian_smooth_1d(true_on_grid, grid, gaussian_sigma_deg)
    approx_smooth = _gaussian_smooth_1d(approx_on_grid, grid, gaussian_sigma_deg)

    p = _to_probability(true_smooth, epsilon)
    q = _to_probability(approx_smooth, epsilon)

    forward = float(np.sum(p * np.log(p / q)))
    reverse = float(np.sum(q * np.log(q / p)))
    modified_sid = forward + reverse

    return {
        "modified_sid": modified_sid,
        "D_true_approx_modified": forward,
        "D_approx_true_modified": reverse,
        "comparison_x_start": x_start,
        "comparison_x_end": x_end,
        "comparison_x": grid,
        "comparison_true_y": true_on_grid,
        "comparison_approx_y": approx_on_grid,
        "comparison_true_y_smooth": true_smooth,
        "comparison_approx_y_smooth": approx_smooth,
        "gaussian_sigma_deg": float(gaussian_sigma_deg),
    }


def _normalize_01(intensity: np.ndarray) -> np.ndarray:
    values = np.asarray(intensity, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def detect_significant_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE,
) -> np.ndarray:
    """Return 2θ positions of significant peaks (y normalized 0–1)."""
    from scipy.signal import find_peaks

    x = np.asarray(x, dtype=float)
    y01 = _normalize_01(y)
    if len(y01) < 3:
        return np.asarray([], dtype=float)
    indices, _ = find_peaks(y01, prominence=float(peak_prominence))
    return x[indices]


def match_peak_sets(
    true_peaks: np.ndarray,
    digitized_peaks: np.ndarray,
    *,
    peak_tolerance_deg: float = DEFAULT_PEAK_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Greedy one-to-one peak matching within a 2θ tolerance (±peak_tolerance_deg)."""
    true_peaks = np.asarray(true_peaks, dtype=float)
    digitized_peaks = np.asarray(digitized_peaks, dtype=float)

    used_digitized: set[int] = set()
    matched_pairs: list[dict[str, float]] = []
    for true_x in true_peaks:
        best_idx = None
        best_dist = None
        for dig_idx, dig_x in enumerate(digitized_peaks):
            if dig_idx in used_digitized:
                continue
            dist = abs(float(true_x) - float(dig_x))
            if dist <= peak_tolerance_deg and (best_dist is None or dist < best_dist):
                best_idx = dig_idx
                best_dist = dist
        if best_idx is not None:
            used_digitized.add(best_idx)
            matched_pairs.append(
                {
                    "true_2theta": float(true_x),
                    "digitized_2theta": float(digitized_peaks[best_idx]),
                    "delta_deg": float(best_dist),
                }
            )

    n_matched = len(matched_pairs)
    n_true = int(len(true_peaks))
    n_dig = int(len(digitized_peaks))
    if n_true == 0 and n_dig == 0:
        peak_recall = 1.0
        peak_precision = 1.0
    else:
        peak_recall = 1.0 if n_true == 0 else n_matched / n_true
        peak_precision = 0.0 if n_dig == 0 else n_matched / n_dig

    denom = peak_precision + peak_recall
    if denom > 0:
        peak_f1 = 2.0 * peak_precision * peak_recall / denom
    else:
        peak_f1 = 0.0

    # Bounded peak penalty: 0 (perfect F1) .. 0.5 (F1 = 0).
    peak_penalty = 0.5 * (1.0 - peak_f1)
    return {
        "peak_recall": float(peak_recall),
        "peak_precision": float(peak_precision),
        "peak_f1": float(peak_f1),
        "peak_penalty": float(peak_penalty),
        "number_true_peaks": n_true,
        "number_digitized_peaks": n_dig,
        "number_matched_peaks": int(n_matched),
        "true_peak_positions": true_peaks.tolist(),
        "digitized_peak_positions": digitized_peaks.tolist(),
        "matched_pairs": matched_pairs,
    }


def _nearest_peak_distances(
    sources: list[float],
    targets: list[float],
) -> list[tuple[float, float, float]]:
    """For each source peak, return (source, nearest_target, |Δ|)."""
    if not targets:
        return [(float(s), float("nan"), float("nan")) for s in sources]
    target_arr = np.asarray(targets, dtype=float)
    rows: list[tuple[float, float, float]] = []
    for source in sources:
        deltas = np.abs(target_arr - float(source))
        idx = int(np.argmin(deltas))
        rows.append((float(source), float(target_arr[idx]), float(deltas[idx])))
    return rows


def format_peak_match_debug(peaks: dict[str, Any]) -> str:
    """Human-readable peak detection / matching dump."""
    true_peaks = [float(t) for t in (peaks.get("true_peak_positions") or [])]
    dig_peaks = [float(d) for d in (peaks.get("digitized_peak_positions") or [])]
    pairs = peaks.get("matched_pairs") or []
    tolerance = float(peaks.get("peak_tolerance_deg") or DEFAULT_PEAK_TOLERANCE_DEG)

    lines = [
        (
            f"Peak detect: prominence={peaks.get('peak_prominence')} "
            f"tolerance=±{tolerance}° "
            f"(normalize each curve to [0,1] before find_peaks)"
        ),
        (
            f"True peaks ({peaks.get('number_true_peaks')}): "
            f"{np.round(np.asarray(true_peaks, dtype=float), 3).tolist()}"
        ),
        (
            f"Digitized peaks ({peaks.get('number_digitized_peaks')}): "
            f"{np.round(np.asarray(dig_peaks, dtype=float), 3).tolist()}"
        ),
        f"Matched pairs ({peaks.get('number_matched_peaks')}):",
    ]
    if not pairs:
        lines.append("  (none)")
    else:
        for pair in pairs:
            lines.append(
                "  "
                f"true={pair['true_2theta']:.3f} <-> dig={pair['digitized_2theta']:.3f} "
                f"(Δ={pair['delta_deg']:.3f}°)"
            )

    unmatched_true = [
        t
        for t in true_peaks
        if not any(abs(t - p["true_2theta"]) < 1e-9 for p in pairs)
    ]
    matched_dig = {p["digitized_2theta"] for p in pairs}
    unmatched_dig = [
        d for d in dig_peaks if not any(abs(d - md) < 1e-9 for md in matched_dig)
    ]
    lines.append(f"Unmatched true: {np.round(unmatched_true, 3).tolist()}")
    lines.append(f"Unmatched digitized: {np.round(unmatched_dig, 3).tolist()}")

    # Nearest-neighbor distances help explain zero matches (e.g. systematic shift
    # just outside ±tolerance).
    if unmatched_true or unmatched_dig:
        lines.append("Nearest distances (even if outside tolerance):")
        for true_x, dig_x, delta in _nearest_peak_distances(unmatched_true, dig_peaks):
            flag = "IN" if delta <= tolerance else "OUT"
            dig_txt = "n/a" if not np.isfinite(dig_x) else f"{dig_x:.3f}"
            delta_txt = "n/a" if not np.isfinite(delta) else f"{delta:.3f}°"
            lines.append(
                f"  true={true_x:.3f} -> nearest dig={dig_txt} "
                f"(Δ={delta_txt}, {flag})"
            )
        for dig_x, true_x, delta in _nearest_peak_distances(unmatched_dig, true_peaks):
            flag = "IN" if delta <= tolerance else "OUT"
            true_txt = "n/a" if not np.isfinite(true_x) else f"{true_x:.3f}"
            delta_txt = "n/a" if not np.isfinite(delta) else f"{delta:.3f}°"
            lines.append(
                f"  dig={dig_x:.3f} -> nearest true={true_txt} "
                f"(Δ={delta_txt}, {flag})"
            )

    lines.append(
        f"recall={peaks.get('peak_recall'):.4f} "
        f"precision={peaks.get('peak_precision'):.4f} "
        f"F1={peaks.get('peak_f1'):.4f} "
        f"penalty={peaks.get('peak_penalty'):.4f}"
    )
    return "\n".join(lines)


def peak_match_metrics(
    true_x: np.ndarray,
    true_y: np.ndarray,
    approx_x: np.ndarray,
    approx_y: np.ndarray,
    *,
    x_start: float,
    x_end: float,
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE,
    peak_tolerance_deg: float = DEFAULT_PEAK_TOLERANCE_DEG,
    debug: bool = False,
) -> dict[str, Any]:
    """Detect and match peaks on the shared-x interpolated curves."""
    true_x = np.asarray(true_x, dtype=float)
    true_y = np.asarray(true_y, dtype=float)
    approx_x = np.asarray(approx_x, dtype=float)
    approx_y = np.asarray(approx_y, dtype=float)

    n_true = int(np.count_nonzero((true_x >= x_start) & (true_x <= x_end)))
    n_approx = int(np.count_nonzero((approx_x >= x_start) & (approx_x <= x_end)))
    n_points = max(n_true, n_approx, 2)
    grid = np.linspace(x_start, x_end, n_points)
    true_on_grid = np.interp(grid, true_x, true_y)
    approx_on_grid = np.interp(grid, approx_x, approx_y)

    # Normalize each curve to [0, 1] independently before prominence filtering.
    true_peaks = detect_significant_peaks(
        grid, true_on_grid, peak_prominence=peak_prominence
    )
    dig_peaks = detect_significant_peaks(
        grid, approx_on_grid, peak_prominence=peak_prominence
    )
    matched = match_peak_sets(
        true_peaks,
        dig_peaks,
        peak_tolerance_deg=peak_tolerance_deg,
    )
    matched["peak_prominence"] = float(peak_prominence)
    matched["peak_tolerance_deg"] = float(peak_tolerance_deg)
    if debug:
        print(format_peak_match_debug(matched))
    return matched


def score_record(comparison: dict[str, Any], *, figure_id: str | None = None) -> dict[str, Any]:
    """Compact JSON-serializable score row for sid_summary / per-figure scores."""
    record = {key: comparison.get(key) for key in SCORE_SUMMARY_KEYS}
    if figure_id is not None:
        record["id"] = figure_id
    return record


def write_score_files(
    comparison: dict[str, Any],
    *,
    figure_dir: Path,
    figure_id: str,
    summary_path: Path | None = None,
) -> Path:
    """Write per-figure scores JSON and optionally merge into sid_summary.json."""
    import json

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    record = score_record(comparison, figure_id=figure_id)
    scores_path = figure_dir / f"{figure_id}_scores.json"
    scores_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if summary_path is not None:
        summary_path = Path(summary_path)
        by_id: dict[Any, dict[str, Any]] = {}
        if summary_path.is_file():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    for row in existing:
                        if isinstance(row, dict) and "id" in row:
                            by_id[row["id"]] = row
            except json.JSONDecodeError:
                pass
        by_id[figure_id] = record
        merged = sorted(
            by_id.values(),
            key=lambda row: (
                str(row.get("id", "")),
            ),
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    return scores_path


def compare_spectra(
    json_path: Path,
    csv_path: Path,
    *,
    shift: float = 0.0,
    epsilon: float = 1e-12,
    gaussian_sigma_deg: float = DEFAULT_GAUSSIAN_SIGMA_DEG,
    modified_epsilon: float = DEFAULT_MODIFIED_EPSILON,
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE,
    peak_tolerance_deg: float = DEFAULT_PEAK_TOLERANCE_DEG,
    debug_peaks: bool = False,
) -> dict[str, Any]:
    """Compare a true JSON spectrum to an approximated CSV spectrum.

    Returns raw SID, modified SID, peak-match metrics, and final XRD score.
    """
    true_x, true_y = load_true_spectrum(json_path)
    approx_x, approx_y = load_approximate_spectrum(csv_path)

    approx_x = approx_x + shift

    # --- Raw SID (legacy): interpolate approx onto true_x within approx range ---
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

    modified = modified_spectral_information_divergence(
        true_x,
        true_y,
        approx_x,
        approx_y,
        gaussian_sigma_deg=gaussian_sigma_deg,
        epsilon=modified_epsilon,
    )

    peaks = peak_match_metrics(
        true_x,
        true_y,
        approx_x,
        approx_y,
        x_start=modified["comparison_x_start"],
        x_end=modified["comparison_x_end"],
        peak_prominence=peak_prominence,
        peak_tolerance_deg=peak_tolerance_deg,
        debug=debug_peaks,
    )

    final_xrd_score = float(modified["modified_sid"]) + float(peaks["peak_penalty"])

    return {
        "forward": float(forward),
        "reverse": float(reverse),
        "symmetric_sid": float(sid),
        "raw_sid": float(sid),
        "modified_sid": float(modified["modified_sid"]),
        "D_true_approx_modified": float(modified["D_true_approx_modified"]),
        "D_approx_true_modified": float(modified["D_approx_true_modified"]),
        "comparison_x_start": float(modified["comparison_x_start"]),
        "comparison_x_end": float(modified["comparison_x_end"]),
        "gaussian_sigma_deg": float(modified["gaussian_sigma_deg"]),
        "peak_recall": float(peaks["peak_recall"]),
        "peak_precision": float(peaks["peak_precision"]),
        "peak_f1": float(peaks["peak_f1"]),
        "peak_penalty": float(peaks["peak_penalty"]),
        "final_xrd_score": final_xrd_score,
        "number_true_peaks": int(peaks["number_true_peaks"]),
        "number_digitized_peaks": int(peaks["number_digitized_peaks"]),
        "number_matched_peaks": int(peaks["number_matched_peaks"]),
        "peak_prominence": float(peak_prominence),
        "peak_tolerance_deg": float(peak_tolerance_deg),
        "true_peak_positions": peaks["true_peak_positions"],
        "digitized_peak_positions": peaks["digitized_peak_positions"],
        "matched_pairs": peaks["matched_pairs"],
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
        help="Small value used to avoid log(0) in raw SID. Default: 1e-12.",
    )
    parser.add_argument(
        "--gaussian-sigma-deg",
        type=float,
        default=DEFAULT_GAUSSIAN_SIGMA_DEG,
        help=(
            "Gaussian smoothing sigma in degrees 2θ for modified SID. "
            f"Default: {DEFAULT_GAUSSIAN_SIGMA_DEG}."
        ),
    )
    parser.add_argument(
        "--modified-epsilon",
        type=float,
        default=DEFAULT_MODIFIED_EPSILON,
        help=(
            "Epsilon added before normalizing modified SID distributions. "
            f"Default: {DEFAULT_MODIFIED_EPSILON}."
        ),
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=DEFAULT_PEAK_PROMINENCE,
        help=f"Peak prominence on 0–1 normalized intensity. Default: {DEFAULT_PEAK_PROMINENCE}.",
    )
    parser.add_argument(
        "--peak-tolerance-deg",
        type=float,
        default=DEFAULT_PEAK_TOLERANCE_DEG,
        help=f"Max 2θ distance for a peak match. Default: {DEFAULT_PEAK_TOLERANCE_DEG}.",
    )
    parser.add_argument(
        "--debug-peaks",
        action="store_true",
        help="Print detected peak positions and matched pairs.",
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
        gaussian_sigma_deg=args.gaussian_sigma_deg,
        modified_epsilon=args.modified_epsilon,
        peak_prominence=args.peak_prominence,
        peak_tolerance_deg=args.peak_tolerance_deg,
        debug_peaks=args.debug_peaks,
    )

    comparison_x = result["comparison_x"]
    print(f"JSON file: {args.json_file}")
    print(f"CSV file: {args.csv_file}")
    print(f"Applied CSV shift: {result['shift']:.10g} degrees")
    print(
        f"Raw compared two-theta range: "
        f"{comparison_x.min():.6f} to {comparison_x.max():.6f}"
    )
    print(f"Number of raw comparison points: {len(comparison_x)}")
    print(f"Raw SID: {result['raw_sid']:.10f}")
    print(
        f"Modified compared two-theta range: "
        f"{result['comparison_x_start']:.6f} to {result['comparison_x_end']:.6f}"
    )
    print(f"Modified SID: {result['modified_sid']:.10f}")
    print(format_peak_match_debug(result))
    print(f"Final XRD score: {result['final_xrd_score']:.10f}")


if __name__ == "__main__":
    main()
