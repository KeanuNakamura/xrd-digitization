"""Compact XRD schemas for large-scale scientific datasets.

Model returns is_xrd first. Non-XRD figures yield analysis=null.
XRD figures use: sample → curves[] → trends[], plus optional literature
fields (fwhm, lattice_parameters, profile_function) when explicitly stated.
"""

from __future__ import annotations

from typing import Literal, Optional

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


class KeyFinding(BaseModel):
    """One cross-figure finding with supporting figure IDs."""

    finding: str = ""
    figure_ids: list[str] = Field(default_factory=list)


class EvidenceCitation(BaseModel):
    """Traceable pointer into the supplied paper JSON for one claim."""

    figure_id: str = ""
    # e.g. "caption", "text", "analysis.trends", "analysis.curves[0].fwhm"
    source: str = ""
    # Exact quote, measurement summary, or JSON field path for missing data.
    evidence: str = ""


class ScientificInterpretation(BaseModel):
    """One paper-level interpretation with claim, reasoning, and citations."""

    # Assertive conclusion only — no because/justification wording.
    claim: str = ""
    reasoning_type: Literal["author_reported", "ai_inferred", "hypothesis"] = (
        "ai_inferred"
    )
    # How the cited evidence supports the claim (kept separate from claim).
    reasoning: str = ""
    citations: list[EvidenceCitation] = Field(default_factory=list)


class PaperAnalysis(BaseModel):
    """Paper-level synthesis appended as ``paper_analysis`` in figure_analysis.json."""

    summary: str = ""
    experimental_variables: list[str] = Field(default_factory=list)
    key_findings: list[KeyFinding] = Field(default_factory=list)
    scientific_interpretation: list[ScientificInterpretation] = Field(
        default_factory=list
    )
