"""Stage 4: per-subplot semantic analysis via OpenAI vision."""

from __future__ import annotations

import logging

import numpy as np

from autodigitizer.models import SubplotAnalysis, SubplotSummary
from autodigitizer.vision.openai_client import OpenAIVisionClient

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are an expert at reading scientific plots. "
    "Given a subplot crop (and optionally the full figure), extract structured axis and series metadata as JSON. "
    "Do NOT invent precise curve coordinates. Focus on labels, scales, legend, and how curves are separated. "
    "annotation_regions and legend_bbox_normalized use normalized coords in the SUBPLOT crop (0-1). "
    "If the y-axis has no numeric tick labels (common for XRD intensity), still set "
    "visible_min=0 and visible_max=1 to indicate relative/arbitrary units spanning the plot height, "
    "and note that in notes. Never leave both visible_min and visible_max null when the plot is digitizable."
)


def analyze_subplot(
    client: OpenAIVisionClient,
    *,
    subplot: SubplotSummary,
    crop: np.ndarray,
    full_figure: np.ndarray | None,
    cache_key: str,
) -> SubplotAnalysis:
    logger.info("Stage 4: analyzing subplot %s", subplot.id)
    prompt = f"""Analyze subplot '{subplot.id}' of a scientific figure.

Global hints:
- plot_type hint: {subplot.plot_type}
- digitizable hint: {subplot.digitizable}

Return:
- whether this subplot is digitizable
- x/y axis labels, units, linear vs log, approximate visible min/max, any readable tick values
- estimated number of distinct data curves/series
- how curves are separated: color / line_style / markers / spatial / stacked / single
- whether curves are stacked or vertically offset (e.g. XRD waterfall)
- legend entries with visual descriptions
- approximate annotation/legend regions inside this crop (normalized 0-1)
- grid/markers presence

Image 1 is the subplot crop. Image 2 (if present) is the full figure for context.
"""
    images: list = [crop]
    if full_figure is not None:
        images.append(full_figure)

    result = client.analyze_structured(
        prompt=prompt,
        schema_model=SubplotAnalysis,
        images=images,
        cache_key=cache_key,
        cache_name=f"subplot_{subplot.id}_analysis.json",
        system=SYSTEM,
    )
    # Ensure id consistency
    result.subplot_id = subplot.id
    logger.info(
        "Subplot %s: type=%s curves~%d separation=%s stacked=%s",
        subplot.id,
        result.plot_type,
        result.estimated_curve_count,
        result.curve_separation,
        result.stacked,
    )
    return result
