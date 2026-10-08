"""Compact XRD schemas for large-scale scientific datasets.

Model returns is_xrd first. Non-XRD figures yield analysis=null.
XRD figures use: sample → curves[] → trends[], plus optional literature
fields (fwhm, lattice_parameters, profile_function) when explicitly stated.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LatticeParameters(BaseModel):
    """Unit-cell parameters when explicitly reported in the paper."""

    phase: Optional[str] = None
    a_A: Optional[float] = None
    b_A: Optional[float] = None
    c_A: Optional[float] = None
    alpha_deg: Optional[float] = None
    beta_deg: Optional[float] = None
    gamma_deg: Optional[float] = None
    volume_A3: Optional[float] = None
    space_group: Optional[str] = None


class XrdCurve(BaseModel):
    condition: Optional[str] = None
    peak_positions: list[float] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    peak_width: Optional[str] = None
    # Literature-only numerical FWHM (° 2θ); null unless explicitly stated.
    fwhm: Optional[float] = None
    lattice_parameters: list[LatticeParameters] = Field(default_factory=list)
    # e.g. "Pseudo-Voigt", "Gaussian", "Lorentzian", "Pearson VII"
    profile_function: Optional[str] = None


class FigureAnalysisPayload(BaseModel):
    """Model-facing payload (OpenAI structured output schema)."""

    is_xrd: bool = False
    sample: Optional[str] = None
    curves: list[XrdCurve] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    # Figure-level literature values when not tied to one curve.
    fwhm: Optional[float] = None
    lattice_parameters: list[LatticeParameters] = Field(default_factory=list)
    profile_function: Optional[str] = None
    # Temporary axis bounds for validation; stripped from dataset dump.
    x_min: Optional[float] = None
    x_max: Optional[float] = None


class FigureAnalysisResult(FigureAnalysisPayload):
    """Result used to populate figure_analysis.json figure entries."""

    pass
