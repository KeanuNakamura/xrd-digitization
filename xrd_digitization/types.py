from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FigureContext:
    """Metadata about a figure before digitization."""

    image_path: Path
    figure_id: str | None = None
    caption: str | None = None
    source_pdf: str | None = None
    page: int | None = None
    crop_bbox: list[float] | None = None


@dataclass
class ClassificationResult:
    is_xrd: bool
    confidence: float
    caption_score: float = 0.0
    image_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlotCropResult:
    cropped_bgr: Any
    bbox: tuple[int, int, int, int]
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlotPanel:
    index: int
    bbox: tuple[int, int, int, int]
    label: str | None = None


@dataclass
class AxisCalibrationResult:
    x_min: float
    x_max: float
    plot_left: int
    plot_right: int
    plot_top: int
    plot_bottom: int
    method: str
    confidence: float
    tick_pairs: list[tuple[int, float]] = field(default_factory=list)
    y_min: float | None = None
    y_max: float | None = None
    y_tick_pairs: list[tuple[int, float]] = field(default_factory=list)
    y_method: str = "relative"
    warnings: list[str] = field(default_factory=list)

    @property
    def has_y_calibration(self) -> bool:
        return (
            self.y_min is not None
            and self.y_max is not None
            and self.y_method.startswith("ocr")
        )


@dataclass
class CurveData:
    two_theta: list[float]
    intensity: list[float]
    curve_id: str = "curve_1"
    color: str | None = None
    label: str | None = None
    warnings: list[str] = field(default_factory=list)
    detected_peaks: list[PeakRecord] = field(default_factory=list)


@dataclass
class PeakRecord:
    two_theta: float
    relative_intensity: float
    prominence: float


@dataclass
class PanelDigitizationResult:
    panel: PlotPanel
    plot_crop: PlotCropResult
    calibration: AxisCalibrationResult
    curves: list[CurveData]
    peaks: list[list[PeakRecord]]
    warnings: list[str] = field(default_factory=list)
    output_stem: str = "figure_1_1"


@dataclass
class DigitizationResult:
    figure_context: FigureContext
    classification: ClassificationResult
    plot_crop: PlotCropResult
    panels: list[PanelDigitizationResult]
    confidence: float
    warnings: list[str] = field(default_factory=list)
    output_stem: str = "figure_1"

    @property
    def calibration(self) -> AxisCalibrationResult | None:
        return self.panels[0].calibration if self.panels else None

    @property
    def curve(self) -> CurveData:
        if self.panels and self.panels[0].curves:
            return self.panels[0].curves[0]
        return CurveData(two_theta=[], intensity=[])

    @property
    def curves(self) -> list[CurveData]:
        out: list[CurveData] = []
        for panel in self.panels:
            out.extend(panel.curves)
        return out

    @property
    def peaks(self) -> list[PeakRecord]:
        if self.panels and self.panels[0].peaks:
            return self.panels[0].peaks[0]
        return []
