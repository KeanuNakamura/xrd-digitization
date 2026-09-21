"""Synthetic matplotlib figures with known ground-truth XY data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CurveTruth:
    label: str
    x: np.ndarray
    y: np.ndarray
    color: str = "C0"
    linestyle: str = "-"
    marker: Optional[str] = None


@dataclass
class PanelTruth:
    id: str
    curves: list[CurveTruth]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    xlabel: str = "x"
    ylabel: str = "y"
    xscale: str = "linear"
    yscale: str = "linear"
    legend: bool = False
    title: Optional[str] = None
    annotate: Optional[str] = None


@dataclass
class FigureTruth:
    panels: list[PanelTruth]
    rows: int = 1
    cols: int = 1
    sharex: bool = False
    sharey: bool = False
    path: Optional[Path] = None


def render_figure(truth: FigureTruth, path: Path, dpi: int = 120) -> FigureTruth:
    fig, axes = plt.subplots(
        truth.rows,
        truth.cols,
        figsize=(4.5 * truth.cols, 3.5 * truth.rows),
        sharex=truth.sharex,
        sharey=truth.sharey,
        dpi=dpi,
    )
    axes_arr = np.array(axes).reshape(-1)
    for ax, panel in zip(axes_arr, truth.panels):
        for curve in panel.curves:
            ax.plot(
                curve.x,
                curve.y,
                color=curve.color,
                linestyle=curve.linestyle,
                marker=curve.marker,
                markevery=max(1, len(curve.x) // 15) if curve.marker else None,
                linewidth=2.0,
                label=curve.label,
            )
        ax.set_xlim(*panel.xlim)
        ax.set_ylim(*panel.ylim)
        ax.set_xscale(panel.xscale)
        ax.set_yscale(panel.yscale)
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        if panel.title:
            ax.set_title(panel.title)
        if panel.legend:
            ax.legend(loc="best", fontsize=8)
        if panel.annotate:
            ax.text(0.05, 0.9, panel.annotate, transform=ax.transAxes, fontsize=10)
        ax.text(
            -0.12,
            1.05,
            f"({panel.id.lower()})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )

    for j in range(len(truth.panels), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    truth.path = path
    return truth


def make_single_curve(path: Path) -> FigureTruth:
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.1 * x
    panel = PanelTruth(
        id="A",
        curves=[CurveTruth("s1", x, y, color="tab:blue")],
        xlim=(0, 10),
        ylim=(-1.5, 2.5),
        xlabel="time",
        ylabel="signal",
    )
    return render_figure(FigureTruth([panel]), path)


def make_three_colored(path: Path) -> FigureTruth:
    x = np.linspace(0, 10, 250)
    curves = [
        CurveTruth("A", x, np.sin(x), color="tab:red"),
        CurveTruth("B", x, np.cos(x), color="tab:green"),
        CurveTruth("C", x, 0.5 * np.sin(2 * x), color="tab:blue"),
    ]
    panel = PanelTruth(
        id="A", curves=curves, xlim=(0, 10), ylim=(-1.5, 1.5), legend=True
    )
    return render_figure(FigureTruth([panel]), path)


def make_four_panels(path: Path) -> FigureTruth:
    panels = []
    for i, pid in enumerate(["A", "B", "C", "D"]):
        x = np.linspace(0, 5, 150)
        y = np.sin(x + i)
        panels.append(
            PanelTruth(
                id=pid,
                curves=[CurveTruth(f"c{i}", x, y, color="tab:purple")],
                xlim=(0, 5),
                ylim=(-1.5, 1.5),
            )
        )
    return render_figure(FigureTruth(panels, rows=2, cols=2), path)


def make_shared_x(path: Path) -> FigureTruth:
    x = np.linspace(10, 80, 200)
    panels = [
        PanelTruth(
            id="A",
            curves=[CurveTruth("top", x, np.exp(-((x - 30) ** 2) / 40), color="tab:red")],
            xlim=(10, 80),
            ylim=(0, 1.2),
            xlabel="",
            ylabel="I (a.u.)",
        ),
        PanelTruth(
            id="B",
            curves=[CurveTruth("bot", x, np.exp(-((x - 50) ** 2) / 50), color="tab:blue")],
            xlim=(10, 80),
            ylim=(0, 1.2),
            xlabel="2θ (deg)",
            ylabel="I (a.u.)",
        ),
    ]
    return render_figure(FigureTruth(panels, rows=2, cols=1, sharex=True), path)


def make_shared_y(path: Path) -> FigureTruth:
    x = np.linspace(0, 10, 200)
    panels = [
        PanelTruth(
            id="A",
            curves=[CurveTruth("L", x, np.sin(x), color="tab:orange")],
            xlim=(0, 10),
            ylim=(-1.2, 1.2),
            xlabel="x",
            ylabel="y",
        ),
        PanelTruth(
            id="B",
            curves=[CurveTruth("R", x, np.cos(x), color="tab:cyan")],
            xlim=(0, 10),
            ylim=(-1.2, 1.2),
            xlabel="x",
            ylabel="",
        ),
    ]
    return render_figure(FigureTruth(panels, rows=1, cols=2, sharey=True), path)


def make_markers(path: Path) -> FigureTruth:
    x = np.linspace(0, 8, 40)
    y = 0.3 * x + np.sin(x)
    panel = PanelTruth(
        id="A",
        curves=[CurveTruth("m", x, y, color="tab:blue", marker="o", linestyle="-")],
        xlim=(0, 8),
        ylim=(-1, 4),
    )
    return render_figure(FigureTruth([panel]), path)


def make_dashed(path: Path) -> FigureTruth:
    x = np.linspace(0, 10, 200)
    curves = [
        CurveTruth("solid", x, np.sin(x), color="black", linestyle="-"),
        CurveTruth("dash", x, np.cos(x), color="black", linestyle="--"),
    ]
    panel = PanelTruth(
        id="A", curves=curves, xlim=(0, 10), ylim=(-1.5, 1.5), legend=True
    )
    return render_figure(FigureTruth([panel]), path)


def make_stacked(path: Path) -> FigureTruth:
    x = np.linspace(20, 70, 300)
    curves = []
    colors = ["tab:red", "tab:green", "tab:blue"]
    for i in range(3):
        peak = np.exp(-((x - (35 + 8 * i)) ** 2) / 20)
        y = peak + i * 1.2
        curves.append(CurveTruth(f"s{i}", x, y, color=colors[i]))
    panel = PanelTruth(
        id="A",
        curves=curves,
        xlim=(20, 70),
        ylim=(-0.2, 4.0),
        xlabel="2θ",
        ylabel="Intensity (offset)",
        legend=True,
    )
    return render_figure(FigureTruth([panel]), path)


def make_annotations(path: Path) -> FigureTruth:
    x = np.linspace(0, 10, 200)
    panel = PanelTruth(
        id="A",
        curves=[CurveTruth("s", x, np.sin(x), color="tab:blue")],
        xlim=(0, 10),
        ylim=(-1.5, 1.5),
        annotate="peak",
    )
    return render_figure(FigureTruth([panel]), path)


def make_log_axis(path: Path) -> FigureTruth:
    x = np.logspace(0, 3, 200)
    y = 1.0 / x
    panel = PanelTruth(
        id="A",
        curves=[CurveTruth("inv", x, y, color="tab:red")],
        xlim=(1, 1000),
        ylim=(1e-3, 1.5),
        xscale="log",
        yscale="log",
        xlabel="f",
        ylabel="amp",
    )
    return render_figure(FigureTruth([panel]), path)
