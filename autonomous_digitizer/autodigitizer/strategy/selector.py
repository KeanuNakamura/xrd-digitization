"""Autonomous extraction strategy selection."""

from __future__ import annotations

from dataclasses import dataclass

from autodigitizer.models import CurveSeparation, PlotExtractionGuide, SubplotAnalysis


@dataclass
class ExtractionStrategy:
    name: str
    reason: str
    params: dict


def select_strategy(
    analysis: SubplotAnalysis,
    attempt: int = 0,
    retry_hint: str | None = None,
    guide: PlotExtractionGuide | None = None,
) -> ExtractionStrategy:
    """Choose curve extraction approach from subplot analysis + optional QA retry hint."""
    if retry_hint:
        mapping = {
            "recolor_cluster": ExtractionStrategy(
                "guided" if guide and guide.curves else "color",
                f"QA retry hint: {retry_hint}",
                {"color_clusters_override": max(2, analysis.estimated_curve_count + attempt)},
            ),
            "continuity_trace": ExtractionStrategy(
                "guided" if guide and guide.curves else "continuity",
                f"QA retry hint: {retry_hint}",
                {},
            ),
            "guided_trace": ExtractionStrategy(
                "guided",
                f"QA retry hint: {retry_hint}",
                {},
            ),
            "alternate_curve_count": ExtractionStrategy(
                "guided" if guide and guide.curves else "stacked",
                f"QA retry hint: {retry_hint}",
                {"color_clusters_override": max(1, analysis.estimated_curve_count + (1 if attempt % 2 else -1))},
            ),
            "expand_text_mask": ExtractionStrategy(
                "guided" if guide and guide.curves else "single",
                f"QA retry hint: {retry_hint}",
                {"expand_text_mask": True},
            ),
            "loosen_background": ExtractionStrategy(
                "guided" if guide and guide.curves else ("color" if analysis.estimated_curve_count > 1 else "single"),
                f"QA retry hint: {retry_hint}",
                {"background_threshold": 235},
            ),
            "retighten_plot_area": ExtractionStrategy(
                "guided" if guide and guide.curves else "auto",
                f"QA retry hint: {retry_hint}",
                {"retighten_plot_area": True},
            ),
        }
        if retry_hint in mapping:
            return mapping[retry_hint]

    if not analysis.digitizable:
        return ExtractionStrategy("none", "subplot marked not digitizable", {})

    if guide and guide.curves:
        return ExtractionStrategy(
            "guided",
            "using OpenAI extraction guide (bands/keypoints/text masks)",
            {},
        )

    if analysis.stacked or analysis.vertically_offset or analysis.curve_separation in (
        CurveSeparation.STACKED,
        CurveSeparation.SPATIAL,
    ):
        return ExtractionStrategy(
            "stacked",
            "stacked / vertically offset / spatially separated curves",
            {},
        )

    if analysis.estimated_curve_count <= 1 or analysis.curve_separation == CurveSeparation.SINGLE:
        return ExtractionStrategy("single", "single curve expected", {})

    if analysis.curve_separation == CurveSeparation.COLOR:
        return ExtractionStrategy(
            "color",
            "curves differentiated by color",
            {"color_clusters_override": analysis.estimated_curve_count},
        )

    if analysis.curve_separation in (CurveSeparation.LINE_STYLE, CurveSeparation.MARKERS):
        return ExtractionStrategy(
            "continuity",
            f"curves differentiated by {analysis.curve_separation.value}",
            {},
        )

    if analysis.estimated_curve_count > 1:
        return ExtractionStrategy(
            "stacked",
            "multiple curves; defaulting to spatial/stacked tracing",
            {},
        )

    return ExtractionStrategy("single", "default single-curve strategy", {})
