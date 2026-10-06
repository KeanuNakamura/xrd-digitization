"""Compact XRD schemas for large-scale scientific datasets.

Model returns is_xrd first. Non-XRD figures yield analysis=null.
XRD figures use: sample → curves[] → trends[].
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class XrdCurve(BaseModel):
    condition: Optional[str] = None
    peak_positions: list[float] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    peak_width: Optional[str] = None


class FigureAnalysisPayload(BaseModel):
    """Model-facing payload (OpenAI structured output schema)."""

    is_xrd: bool = False
    sample: Optional[str] = None
    curves: list[XrdCurve] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    # Temporary axis bounds for validation; stripped from dataset dump.
    x_min: Optional[float] = None
    x_max: Optional[float] = None


class FigureAnalysisResult(FigureAnalysisPayload):
    """Result used to populate figure_analysis.json figure entries."""

    pass
