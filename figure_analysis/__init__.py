"""Compact XRD scientific figure metadata extraction (vision → structured JSON).

Extracts dataset-friendly XRD metadata (peaks, phases, hkl, FWHM/broadening,
conditions). Does NOT digitize curves or reconstruct XY spectra.
"""

from __future__ import annotations

from figure_analysis.analyzer import analyze_figure_image, analyze_paper
from figure_analysis.schemas import (
    EvidenceCitation,
    FigureAnalysisPayload,
    FigureAnalysisResult,
    PaperAnalysis,
    ScientificInterpretation,
)

__all__ = [
    "EvidenceCitation",
    "FigureAnalysisPayload",
    "FigureAnalysisResult",
    "PaperAnalysis",
    "ScientificInterpretation",
    "analyze_figure_image",
    "analyze_paper",
]
