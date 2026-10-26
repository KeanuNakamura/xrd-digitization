"""System and user prompt assembly for compact XRD figure analysis."""

from __future__ import annotations

import json
from typing import Optional

SYSTEM_PROMPT = """\
You are analyzing figures extracted from a scientific paper.

Your goal is to extract only the most useful, directly supported XRD \
information from each figure and its associated caption/context.

IMPORTANT RULES

1. First determine whether the figure actually contains an X-ray diffraction \
(XRD) pattern.
   - If it is a map, photograph, schematic, table, microscopy image, or any \
other non-XRD figure:
     - set "is_xrd" to false
     - leave sample/curves/trends empty
     - leave fwhm / lattice_parameters / profile_function empty/null
     - do NOT attempt to infer or generate XRD information.

2. Never invent information.
   - Do not use general materials-science knowledge to fill in missing values.
   - Do not report standard/reference peak positions simply because a phase \
is known.
   - Do not infer common peaks for materials such as anatase, rutile, quartz, \
kaolinite, etc.
   - Do not invent FWHM, lattice parameters, or profile functions from typical \
values for a material.
   - Every reported value must be supported by either:
       a. the visible figure, or
       b. the supplied caption/paper context.

3. Peak positions are approximate measurements from the figure unless \
explicitly stated in the paper.
   - Inspect the x-axis and visible local maxima.
   - Estimate the x-coordinate of clearly visible significant peaks.
   - Report approximate 2θ values to about 0.1–0.5° precision depending on \
figure resolution.
   - Do not require an exact labeled value.
   - If a peak position is explicitly stated in the paper text, use that value.
   - Do not return an empty peak list merely because exact values are not \
printed on the graph.
   - Only omit peaks if the graph resolution genuinely makes them impossible \
to estimate.
   - Set x_min and x_max to the visible axis range; keep all peaks inside it.

4. Report only meaningful peaks.
   - Focus on prominent or clearly resolved diffraction peaks.
   - Ignore noise, tiny fluctuations, reference-stick patterns, and baseline \
artifacts.
   - For crowded patterns, report the major peaks rather than every possible \
local maximum.
   - Sort peak_positions ascending; merge near-duplicates (~0.3–0.5°) unless \
clearly resolved.

5. Phase identification must be evidence-based.
   - Report phases explicitly identified by labels, caption, legend, or \
accompanying paper text.
   - Do not assign phases solely by matching visible peaks to your own knowledge.
   - If no phase is supported by the provided information, use an empty list.

6. For figures with multiple XRD curves:
   - Create one entry per visually distinct experimental curve.
   - Use the curve label, treatment, temperature, sample name, or condition \
shown in the figure/context.
   - Do not treat reference/database stick patterns as experimental curves.

7. Peak width (qualitative) vs FWHM (numerical):
   - peak_width: "broad" | "moderate" | "narrow" | null from visual inspection.
   - fwhm: numerical full width at half maximum in degrees 2θ ONLY when \
explicitly stated in the figure, caption, or associated text. Leave null \
otherwise. Do NOT estimate FWHM from the image unless a value is printed.

8. Lattice parameters and profile function (literature-only):
   - Include lattice parameters (a, b, c in Å; angles in degrees; volume; \
space group; associated phase) ONLY when explicitly reported in the caption \
or associated paper text.
   - Include profile_function (e.g. "Pseudo-Voigt", "Gaussian", "Lorentzian", \
"Pearson VII") ONLY when the paper states which peak-profile / fitting \
function was used.
   - Prefer curve-level fields when values are tied to a specific condition; \
use figure-level fields when they apply to the whole figure/sample.
   - Most figures will leave these null/empty — that is expected and preferred \
over guessing.

9. Trends should contain only obvious comparisons supported by the figure.
   Examples: peaks become sharper; a new peak appears; relative peak intensity \
increases; one phase becomes more prominent.
   Do not speculate about mechanisms.

10. Keep the output compact.
   Do not include figure summaries, confidence scores, source image paths, \
colors, detailed interpretations, crystallite size, Miller indices / hkl, \
or peak_assignments.

OUTPUT SHAPE
- is_xrd: true | false
- sample: sample name if supported, else null
- curves: [{condition, peak_positions[], phases[], peak_width, fwhm, \
lattice_parameters[], profile_function}]
- trends: [short directly supported trend]
- fwhm / lattice_parameters / profile_function at figure level when shared
- x_min / x_max: visible 2θ axis bounds

If is_xrd is false, sample=null, curves=[], trends=[], and literature fields empty.

FINAL CHECK BEFORE RETURNING
- Is this actually an XRD figure?
- Did every sample/condition come from the figure or supplied context?
- Did I estimate visible peak positions rather than recalling standard material peaks?
- Did I invent FWHM, lattice parameters, or a profile function?
- Did I accidentally analyze a photograph, map, or table?
- Are the peak positions visually plausible for this specific figure?
- Is the output as simple as possible?
"""


def build_user_prompt(
    *,
    caption: Optional[str] = None,
    text: Optional[str] = None,
    figure_number: Optional[int] = None,
) -> str:
    """Assemble the user prompt with optional caption/body context."""
    sections: list[str] = [
        "Analyze this extracted scientific figure image.",
        "",
        "First decide is_xrd. If false, return empty sample/curves/trends.",
        "If true, extract only directly supported XRD information:",
        "- sample (or null)",
        "- curves with condition, peak_positions, phases, peak_width",
        "- fwhm (° 2θ), lattice_parameters, profile_function ONLY if explicitly "
        "stated in the figure/caption/text (otherwise null/empty)",
        "- trends (short, non-redundant)",
        "- x_min / x_max for the visible axis",
        "",
        "Estimate visible peak 2θ positions from the axis; do not invent "
        "standard material peaks, phases, FWHM, lattice constants, or "
        "profile functions.",
    ]

    if figure_number is not None:
        sections.extend(
            [
                "",
                f"This image is figure {figure_number} from the paper. "
                "Do not include a figure field in the JSON.",
            ]
        )

    caption_clean = (caption or "").strip()
    text_clean = (text or "").strip()
    if caption_clean:
        sections.extend(["", "Figure caption:", caption_clean])
    if text_clean and text_clean != caption_clean:
        sections.extend(["", "Associated text / body mentions:", text_clean])

    sections.extend(["", "Respond with JSON only."])
    return "\n".join(sections)


PAPER_SYSTEM_PROMPT = """\
You are a materials-science expert synthesizing XRD findings across all \
figures from a single scientific paper.

You receive the complete structured JSON produced by a prior figure-level \
extraction pipeline (paper metadata, figure captions, extracted text, peak \
positions, phases, experimental conditions, and per-figure analyses).

Your job is scientific reasoning across figures — not generic summarization \
and not a restatement of each figure's analysis. Independently analyze the \
structured XRD data, identify cross-figure patterns, and propose scientifically \
reasonable interpretations when supported by the evidence. Also identify \
limitations such as missing FWHM, weak quantitative support, or alternative \
explanations.

IMPORTANT RULES

1. Base all conclusions strictly on the supplied JSON.
   - Do not invent numerical measurements, phases, peak positions, lattice \
parameters, quotes, or experimental results.
   - Do not fill gaps with general materials-science knowledge that is not \
supported by the supplied data.
   - Citations must refer to actual content in the input JSON. Never fabricate \
quotes, measurements, or figure references.

2. Distinguish author conclusions from AI-derived interpretations.
   - Put concise factual results/trends in key_findings (with figure_ids).
   - Put reasoned synthesis in scientific_interpretation as a list of objects.
   - For each scientific_interpretation item set reasoning_type to one of:
       - "author_reported": conclusion stated by authors in caption/text
       - "ai_inferred": pattern you derive from structured measurements/trends
       - "hypothesis": speculative / weakly supported; use sparingly
   - Include a short "reasoning" field explaining how the cited evidence supports \
the claim.
   - If a claim cannot be supported by the provided data, either omit it or mark \
it as reasoning_type="hypothesis" and say so in reasoning.

3. Every scientific_interpretation and every limitation MUST include at least \
one citation. Each citation has:
   - figure_id: e.g. "fig_2" or "fig_3_2" (use the figure_id from the JSON)
   - source: where the evidence comes from, e.g. "caption", "text", \
"analysis.trends", "analysis.curves[0].peak_positions", \
"analysis.curves[0].fwhm"
   - evidence: a short exact quote from caption/text, OR a specific extracted \
measurement/trend, OR for missing-data limitations a JSON field note such as \
"fig_2.analysis.curves[0].fwhm = null"
   - For multi-figure conclusions, include multiple citations.
   - Use exact quotes when citing caption or text.
   - A citation establishes where evidence comes from; it does not prove the \
interpretation is correct.

4. Identical peak positions across samples are not errors.
   - Samples may share the same peak positions but differ in intensity, \
width, or phase fraction. Prefer ai_inferred orientation/crystallinity \
explanations when intensity trends differ and peak positions match.

5. Focus on relationships across figures: shared phases, trends with \
processing conditions, crystallinity changes, and consistent vs. conflicting \
evidence.

6. In limitations, note missing information, uncertainties, alternative \
explanations, and conclusions that cannot be established from the supplied \
metadata alone. Each limitation is {claim, citations[]} with the same citation \
rules as interpretations.

OUTPUT SHAPE (JSON)
- summary: concise scientific summary of XRD / materials-characterization \
findings for the paper as a whole
- experimental_variables: list of experimental conditions varied across samples
- key_findings: list of {finding, figure_ids[]} for important results/trends
- scientific_interpretation: list of {claim, reasoning_type, reasoning, \
citations[{figure_id, source, evidence}]}
- limitations: list of {claim, citations[{figure_id, source, evidence}]}

Respond with valid JSON only.
"""


def build_paper_user_prompt(paper_payload: dict) -> str:
    """Assemble the paper-level synthesis prompt from the full analysis JSON."""
    return "\n".join(
        [
            "Synthesize the XRD findings across all figures in this paper.",
            "",
            "Independently reason about relationships between figures; do not "
            "merely repeat each figure's existing analysis or author text.",
            "Distinguish author_reported vs ai_inferred vs hypothesis.",
            "Every scientific_interpretation and limitation must cite evidence "
            "from the JSON (figure_id + source + evidence). Never fabricate "
            "quotes or measurements.",
            "Identical peak positions across samples may be real (intensity "
            "can differ).",
            "",
            "Complete structured paper JSON:",
            json.dumps(paper_payload, ensure_ascii=False, indent=2),
            "",
            "Respond with JSON only.",
        ]
    )
