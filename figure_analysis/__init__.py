"""Compact XRD scientific figure metadata extraction (vision → structured JSON).

Extracts dataset-friendly XRD metadata (peaks, phases, hkl, FWHM/broadening,
conditions). Does NOT digitize curves or reconstruct XY spectra.
"""

from __future__ import annotations

from figure_analysis.analyzer import analyze_figure_image
from figure_analysis.schemas import FigureAnalysisPayload, FigureAnalysisResult

__all__ = ["FigureAnalysisPayload", "FigureAnalysisResult", "analyze_figure_image"]
