"""Stage 2: global figure analysis via OpenAI vision."""

from __future__ import annotations

import logging

import numpy as np

from autodigitizer.models import FigureAnalysis, FigureLayout, SubplotSummary
from autodigitizer.vision.openai_client import OpenAIVisionClient

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are an expert scientific figure interpreter. "
    "Analyze the figure layout carefully. Return only structured JSON. "
    "Bounding boxes are normalized [x0, y0, x1, y1] in 0-1 relative to the full image. "
    "Each subplot bbox must include the plotting area PLUS nearby tick labels and axis titles "
    "needed for calibration, but should not heavily overlap neighboring subplots. "
    "Prefer slightly generous crops over tight crops that cut tick labels."
)

PROMPT = """Analyze this scientific figure completely.

Determine:
- whether any part is digitizable (line/scatter plots with readable axes)
- single vs multi-panel layout (rows/columns if applicable)
- each subplot id (prefer panel labels like A/B/a/b if visible, else A,B,C...)
- normalized bounding box for each subplot
- plot type hints (line, multi_line, scatter, stacked, bar, image, other)
- which subplots share x-axes and/or y-axes
- whether a global legend exists

If a panel is a photo/schematic/table and not a plot, mark digitizable=false.
Be conservative about digitizable=true only when axes and data traces appear recoverable.
"""


def analyze_figure(
    client: OpenAIVisionClient,
    image: np.ndarray,
    cache_key: str,
) -> FigureAnalysis:
    logger.info("Stage 2: global figure analysis")
    result = client.analyze_structured(
        prompt=PROMPT,
        schema_model=FigureAnalysis,
        images=[image],
        cache_key=cache_key,
        cache_name="figure_analysis.json",
        system=SYSTEM,
    )
    if not result.subplots:
        result.subplots = [
            SubplotSummary(
                id="A",
                bbox_normalized=[0.02, 0.02, 0.98, 0.98],
                plot_type="line",
                digitizable=result.digitizable,
                confidence=result.confidence,
            )
        ]
    result.subplot_count = len(result.subplots)
    logger.info(
        "Figure analysis: type=%s subplots=%d digitizable=%s",
        result.figure_type,
        result.subplot_count,
        result.digitizable,
    )
    return result


def offline_single_panel_analysis(image: np.ndarray) -> FigureAnalysis:
    """Deterministic fallback used in tests without OpenAI."""
    return FigureAnalysis(
        digitizable=True,
        confidence=0.4,
        figure_type="single_panel",
        subplot_count=1,
        layout=FigureLayout(rows=1, columns=1),
        subplots=[
            SubplotSummary(
                id="A",
                bbox_normalized=[0.0, 0.0, 1.0, 1.0],
                plot_type="line",
                digitizable=True,
                confidence=0.4,
            )
        ],
    )
