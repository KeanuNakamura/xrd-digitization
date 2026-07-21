from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xrd_digitization.types import AxisCalibrationResult, CurveData

COLOR_MAP = {
    "black": "#222222",
    "blue": "#1f77b4",
    "red": "#d62728",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "cyan": "#17becf",
    "auto": "#9467bd",
}


def load_xy(xy_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load two-column .xy file (two_theta, intensity)."""
    data = np.loadtxt(xy_path)
    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError(f"Invalid .xy file (expected 2 columns): {xy_path}")
    return data[:, 0], data[:, 1]


def _resolve_color(color: str | None, index: int) -> str:
    if color is None:
        return list(COLOR_MAP.values())[index % len(COLOR_MAP)]
    return COLOR_MAP.get(color, color)


def _axis_labels(calibration: AxisCalibrationResult | None) -> tuple[str, str, float | None, float | None]:
    xlabel = "2θ (deg)"
    ylabel = "Relative intensity"
    y_min = 0.0
    y_max: float | None = None
    if calibration is not None and calibration.has_y_calibration:
        ylabel = "Intensity (cps)"
        y_min = calibration.y_min or 0.0
        y_max = calibration.y_max
    return xlabel, ylabel, y_min, y_max


def save_multi_column_xy(
    curves: list[CurveData],
    output_path: Path,
    *,
    num_points: int = 2000,
) -> Path:
    """Write a multi-column .xy file: 2θ, curve_1, curve_2, ..."""
    valid = [curve for curve in curves if curve.two_theta]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not valid:
        output_path.write_text("# no curve extracted\n", encoding="utf-8")
        return output_path

    if len(valid) == 1:
        data = np.column_stack((valid[0].two_theta, valid[0].intensity))
        np.savetxt(output_path, data, fmt="%.6f %.6f")
        return output_path

    x_min = min(min(curve.two_theta) for curve in valid)
    x_max = max(max(curve.two_theta) for curve in valid)
    grid = np.linspace(x_min, x_max, num_points)
    columns = [grid]
    for curve in valid:
        y = np.interp(grid, curve.two_theta, curve.intensity, left=np.nan, right=np.nan)
        columns.append(y)
    data = np.column_stack(columns)
    np.savetxt(output_path, data, fmt="%.6f " + " ".join(["%.6f"] * len(valid)))
    return output_path


def plot_from_curves(
    curves: list[CurveData],
    output_path: Path,
    *,
    calibration: AxisCalibrationResult | None = None,
    title: str | None = None,
) -> Path:
    """Render one digitized plot PNG from in-memory curve data."""
    valid = [curve for curve in curves if curve.two_theta]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not valid:
        return output_path

    xlabel, ylabel, y_min, y_max = _axis_labels(calibration)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    x_lo = float("inf")
    x_hi = float("-inf")

    for index, curve in enumerate(valid):
        x = np.asarray(curve.two_theta, dtype=float)
        y = np.asarray(curve.intensity, dtype=float)
        label = curve.label or curve.curve_id
        color = _resolve_color(curve.color, index)
        ax.plot(x, y, color=color, linewidth=1.0, label=label if len(valid) > 1 else None)
        x_lo = min(x_lo, float(x.min()))
        x_hi = max(x_hi, float(x.max()))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_lo, x_hi)
    y_top = y_max
    if valid:
        curve_max = max(float(max(curve.intensity)) for curve in valid)
        if y_top is None:
            y_top = curve_max * 1.05 if curve_max > 0 else None
        elif curve_max > y_top * 0.98:
            y_top = max(y_top, curve_max * 1.05)
    ax.set_ylim(bottom=y_min, top=y_top)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if len(valid) > 1:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_xy_curve(
    xy_path: Path,
    output_path: Path,
    *,
    title: str | None = None,
    xlabel: str = "2θ (deg)",
    ylabel: str = "Relative intensity",
    color: str | None = None,
) -> Path:
    """Plot a digitized XRD curve from a .xy file and save as PNG."""
    two_theta, intensity = load_xy(xy_path)
    line_color = _resolve_color(color, 0)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(two_theta, intensity, color=line_color, linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(two_theta.min(), two_theta.max())
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_multi_xy_curves(
    xy_paths: list[Path],
    output_path: Path,
    *,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    title: str | None = None,
    xlabel: str = "2θ (deg)",
    ylabel: str = "Relative intensity",
) -> Path:
    """Overlay multiple digitized curves in one plot."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for index, xy_path in enumerate(xy_paths):
        two_theta, intensity = load_xy(xy_path)
        label = labels[index] if labels and index < len(labels) else xy_path.stem
        color = colors[index] if colors and index < len(colors) else None
        color = _resolve_color(color, index)
        ax.plot(two_theta, intensity, color=color, linewidth=1.0, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if len(xy_paths) > 1:
        ax.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
