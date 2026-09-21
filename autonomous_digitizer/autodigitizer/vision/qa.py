"""Stage 13: visual QA comparing original subplot vs reconstruction."""

from __future__ import annotations

import logging

import numpy as np

from autodigitizer.models import QAResult, VisionQAResponse
from autodigitizer.vision.openai_client import OpenAIVisionClient

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a strict QA reviewer for scientific plot digitization. "
    "Compare the original subplot (image 1) to a reconstruction from digitized data (image 2). "
    "Pass only if curves, axis ranges, and relative shapes substantially match. "
    "For stacked/offset spectra, matching relative peak positions and vertical ordering matters more "
    "than absolute y units. "
    "If failing, suggest ONE concrete retry strategy keyword from: "
    "guided_trace, expand_text_mask, continuity_trace, alternate_curve_count, "
    "recolor_cluster, retighten_plot_area, loosen_background, none. "
    "Prefer guided_trace when text annotations or stacked similar-color curves look wrong. "
    "Always respond with JSON."
)


def visual_qa(
    client: OpenAIVisionClient,
    *,
    subplot_id: str,
    original: np.ndarray,
    reconstruction: np.ndarray,
    cache_key: str,
    attempt: int,
    local_metrics: dict[str, float] | None = None,
) -> QAResult:
    logger.info("Stage 13: visual QA for subplot %s (attempt %d)", subplot_id, attempt)
    metrics_txt = ""
    if local_metrics:
        metrics_txt = "Local metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in local_metrics.items())
    prompt = f"""Compare original subplot '{subplot_id}' (image 1) with reconstruction (image 2).

{metrics_txt}

Return JSON with fields: passed, confidence, issues, suggested_retry_strategy.
"""
    vision = client.analyze_structured(
        prompt=prompt,
        schema_model=VisionQAResponse,
        images=[original, reconstruction],
        cache_key=cache_key,
        cache_name=f"qa_{subplot_id}_attempt{attempt}.json",
        system=SYSTEM,
    )
    result = QAResult(
        passed=vision.passed,
        confidence=vision.confidence,
        issues=vision.issues,
        suggested_retry_strategy=vision.suggested_retry_strategy,
        metrics=dict(local_metrics or {}),
    )
    return result
