"""Pydantic models for structured pipeline state and OpenAI outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DigitizationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_DIGITIZABLE = "not_digitizable"


class AxisScale(str, Enum):
    LINEAR = "linear"
    LOG = "log"
    UNKNOWN = "unknown"


class CurveSeparation(str, Enum):
    COLOR = "color"
    LINE_STYLE = "line_style"
    MARKERS = "markers"
    SPATIAL = "spatial"
    STACKED = "stacked"
    UNKNOWN = "unknown"
    SINGLE = "single"


class BBoxNorm(BaseModel):
    """Normalized bounding box [x0, y0, x1, y1] in 0–1 image coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_list(cls, values: list[float]) -> "BBoxNorm":
        if len(values) != 4:
            raise ValueError("bbox must have 4 values")
        return cls(x0=values[0], y0=values[1], x1=values[2], y1=values[3])

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def clamp(self) -> "BBoxNorm":
        return BBoxNorm(
            x0=max(0.0, min(1.0, self.x0)),
            y0=max(0.0, min(1.0, self.y0)),
            x1=max(0.0, min(1.0, self.x1)),
            y1=max(0.0, min(1.0, self.y1)),
        )

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = int(round(self.x0 * width))
        y0 = int(round(self.y0 * height))
        x1 = int(round(self.x1 * width))
        y1 = int(round(self.y1 * height))
        x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
        return x0, y0, x1, y1


class LegendEntry(BaseModel):
    label: str
    visual_description: str = ""
    color_hint: Optional[str] = None
    line_style_hint: Optional[str] = None


class CurveGuide(BaseModel):
    """OpenAI-provided extraction hints for one series (plot-normalized coords)."""

    label: Optional[str] = None
    color_rgb: Optional[list[int]] = None  # [R,G,B]
    # Vertical band within the plot rectangle, y=0 at top, y=1 at bottom
    y_band_normalized: list[float] = Field(default_factory=lambda: [0.0, 1.0])
    # Sparse key points [[x,y], ...] in plot-normalized coordinates (NOT data units)
    key_points_normalized: list[list[float]] = Field(default_factory=list)
    line_style_hint: Optional[str] = None


class PlotExtractionGuide(BaseModel):
    """Vision-guided plan for quantitative CV extraction."""

    # Text/annotation boxes in PLOT-normalized coords [x0,y0,x1,y1] (exclude glyphs only)
    text_regions: list[list[float]] = Field(default_factory=list)
    # Vertical dashed guide line x positions in plot-normalized [0,1]
    vertical_guides_x: list[float] = Field(default_factory=list)
    curves: list[CurveGuide] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AxisInfo(BaseModel):
    label: Optional[str] = None
    unit: Optional[str] = None
    scale: AxisScale = AxisScale.LINEAR
    visible_min: Optional[float] = None
    visible_max: Optional[float] = None
    shared: bool = False
    tick_values_hint: list[float] = Field(default_factory=list)
    confidence: float = 0.5


class SubplotSummary(BaseModel):
    id: str
    bbox_normalized: list[float]
    plot_type: str = "line"
    digitizable: bool = True
    label_text: Optional[str] = None
    confidence: float = 0.5
    notes: list[str] = Field(default_factory=list)

    @field_validator("bbox_normalized")
    @classmethod
    def _validate_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox_normalized must have length 4")
        return v


class AxisRelationships(BaseModel):
    shared_x_groups: list[list[str]] = Field(default_factory=list)
    shared_y_groups: list[list[str]] = Field(default_factory=list)


class FigureLayout(BaseModel):
    """Grid layout hint for multi-panel figures."""

    rows: Optional[int] = None
    columns: Optional[int] = None


class FigureAnalysis(BaseModel):
    digitizable: bool = True
    confidence: float = 0.5
    figure_type: str = "single_panel"
    subplot_count: int = 1
    layout: FigureLayout = Field(default_factory=FigureLayout)
    subplots: list[SubplotSummary] = Field(default_factory=list)
    axis_relationships: AxisRelationships = Field(default_factory=AxisRelationships)
    global_legend: bool = False
    notes: list[str] = Field(default_factory=list)


class SubplotAnalysis(BaseModel):
    subplot_id: str
    digitizable: bool = True
    confidence: float = 0.5
    plot_type: str = "line"
    x_axis: AxisInfo = Field(default_factory=AxisInfo)
    y_axis: AxisInfo = Field(default_factory=AxisInfo)
    estimated_curve_count: int = 1
    curve_separation: CurveSeparation = CurveSeparation.UNKNOWN
    stacked: bool = False
    vertically_offset: bool = False
    legend_present: bool = False
    legend_entries: list[LegendEntry] = Field(default_factory=list)
    annotations_present: bool = False
    annotation_regions: list[list[float]] = Field(default_factory=list)
    legend_bbox_normalized: Optional[list[float]] = None
    grid_present: bool = False
    markers_present: bool = False
    notes: list[str] = Field(default_factory=list)


class PlotArea(BaseModel):
    left: int
    right: int
    top: int
    bottom: int
    confidence: float = 0.5

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)


class TickMark(BaseModel):
    pixel: float
    value: Optional[float] = None
    axis: Literal["x", "y"] = "x"
    confidence: float = 0.5


class AxisCalibration(BaseModel):
    scale: AxisScale = AxisScale.LINEAR
    # For linear: data = a * pixel + b
    # For log: log10(data) = a * pixel + b
    a: float
    b: float
    ticks: list[TickMark] = Field(default_factory=list)
    residual_rmse: float = 0.0
    confidence: float = 0.5
    source: str = "local"
    calibration_type: str = "direct"
    inherited_from: Optional[str] = None
    data_min: Optional[float] = None
    data_max: Optional[float] = None

    def pixel_to_data(self, pixel: float) -> float:
        mapped = self.a * pixel + self.b
        if self.scale == AxisScale.LOG:
            return float(10**mapped)
        return float(mapped)

    def data_to_pixel(self, data: float) -> float:
        if self.scale == AxisScale.LOG:
            import math

            if data <= 0:
                raise ValueError("log axis requires positive data")
            mapped = math.log10(data)
        else:
            mapped = data
        if abs(self.a) < 1e-15:
            raise ValueError("degenerate calibration")
        return (mapped - self.b) / self.a


class CurveStyle(BaseModel):
    color_rgb: Optional[tuple[int, int, int]] = None
    color_name: Optional[str] = None
    line_style: str = "solid"
    marker: Optional[str] = None


class ExtractedCurve(BaseModel):
    id: str
    label: Optional[str] = None
    style: CurveStyle = Field(default_factory=CurveStyle)
    # Pixel-space ordered points within plot area (absolute image coords)
    pixels: list[tuple[float, float]] = Field(default_factory=list)
    # Calibrated data points
    data: list[tuple[float, float]] = Field(default_factory=list)
    displayed_values: bool = True
    vertical_offset_detected: bool = False
    offset_corrected: bool = False
    y_offset_corrected: Optional[list[tuple[float, float]]] = None
    confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)


class QAResult(BaseModel):
    """Vision QA payload. `metrics` is filled locally after the API call."""

    passed: bool = True
    confidence: float = 0.5
    issues: list[str] = Field(default_factory=list)
    suggested_retry_strategy: Optional[str] = None
    # Not sent to OpenAI structured schema (see VisionQAResponse).
    metrics: dict[str, float] = Field(default_factory=dict)


class VisionQAResponse(BaseModel):
    """Strict OpenAI-facing QA schema (no free-form dicts)."""

    passed: bool = True
    confidence: float = 0.5
    issues: list[str] = Field(default_factory=list)
    suggested_retry_strategy: Optional[str] = None


class SubplotResult(BaseModel):
    id: str
    status: DigitizationStatus = DigitizationStatus.FAILED
    analysis: Optional[SubplotAnalysis] = None
    plot_area: Optional[PlotArea] = None
    x_calibration: Optional[AxisCalibration] = None
    y_calibration: Optional[AxisCalibration] = None
    curves: list[ExtractedCurve] = Field(default_factory=list)
    strategy_name: Optional[str] = None
    strategy_reason: Optional[str] = None
    qa: Optional[QAResult] = None
    warnings: list[str] = Field(default_factory=list)
    attempts: int = 1
    crop_bbox_normalized: Optional[list[float]] = None


class PipelineReport(BaseModel):
    source_image: str
    status: DigitizationStatus
    figure_analysis: Optional[FigureAnalysis] = None
    subplots: list[SubplotResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stages_completed: list[str] = Field(default_factory=list)
    api_calls: int = 0
    cache_hits: int = 0
