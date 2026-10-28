"""Tests for compact XRD figure_analysis schemas and OpenAI strict compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_analysis.analyzer import analysis_to_dataset_dict, post_validate
from figure_analysis.openai_client import to_openai_strict_schema
from figure_analysis.prompts import (
    PAPER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_paper_user_prompt,
    build_user_prompt,
)
from figure_analysis.schemas import (
    EvidenceCitation,
    FigureAnalysisPayload,
    FigureAnalysisResult,
    KeyFinding,
    LatticeParameters,
    PaperAnalysis,
    ScientificInterpretation,
    XrdCurve,
)


def _assert_strict(node: dict, path: str = "root") -> None:
    if not isinstance(node, dict):
        return
    if "properties" in node:
        assert node.get("additionalProperties") is False, path
        assert "required" in node, path
        assert set(node["required"]) == set(node["properties"].keys()), (
            f"{path}: required={node.get('required')} props={list(node['properties'])}"
        )
        for k, v in node["properties"].items():
            _assert_strict(v, f"{path}.{k}")
    for key in ("$defs", "definitions"):
        if key in node:
            for name, defn in node[key].items():
                _assert_strict(defn, f"{path}/{key}/{name}")
    if "items" in node and isinstance(node["items"], dict):
        _assert_strict(node["items"], f"{path}.items")
    for key in ("anyOf", "oneOf", "allOf"):
        if key in node:
            for i, alt in enumerate(node[key]):
                _assert_strict(alt, f"{path}.{key}[{i}]")


def test_payload_schema_is_strict():
    schema = to_openai_strict_schema(FigureAnalysisPayload.model_json_schema())
    _assert_strict(schema)
    props = schema.get("properties", {})
    assert "is_xrd" in props
    assert "sample" in props
    assert "curves" in props
    assert "trends" in props
    assert "composition" in props
    assert "space_group" in props
    assert "fwhm" in props
    assert "lattice_parameters" in props
    assert "profile_function" in props
    defs = schema.get("$defs", {})
    assert "XrdCurve" in defs
    assert "LatticeParameters" in defs
    curve_props = defs["XrdCurve"]["properties"]
    assert "composition" in curve_props
    assert "space_group" in curve_props
    assert "fwhm" in curve_props
    assert "lattice_parameters" in curve_props
    assert "profile_function" in curve_props


def test_non_xrd_yields_null_analysis():
    result = FigureAnalysisResult(is_xrd=False, sample="should clear", curves=[XrdCurve()])
    out = post_validate(result)
    assert out.is_xrd is False
    assert out.sample is None
    assert out.curves == []
    assert analysis_to_dataset_dict(out) is None


def test_dataset_dict_includes_literature_fields():
    result = FigureAnalysisResult(
        is_xrd=True,
        sample="TiO2",
        composition=["TiO2"],
        space_group=["I41/amd"],
        profile_function="Pseudo-Voigt",
        lattice_parameters=[
            LatticeParameters(phase="anatase", a_A=3.78, c_A=9.51, space_group="I41/amd")
        ],
        curves=[
            XrdCurve(
                condition="700 °C",
                peak_positions=[25.3, 37.8],
                phases=["anatase"],
                composition=["TiO2"],
                space_group=["I41/amd"],
                peak_width="narrow",
                fwhm=0.22,
                profile_function="Pseudo-Voigt",
                lattice_parameters=[
                    LatticeParameters(phase="anatase", a_A=3.78, c_A=9.51)
                ],
            )
        ],
        trends=[],
    )
    data = analysis_to_dataset_dict(result)
    assert data is not None
    assert data["composition"] == ["TiO2"]
    assert data["space_group"] == ["I41/amd"]
    assert data["profile_function"] == "Pseudo-Voigt"
    assert data["lattice_parameters"][0]["a_A"] == 3.78
    assert data["lattice_parameters"][0]["b_A"] is None
    assert data["curves"][0]["fwhm"] == 0.22
    assert data["curves"][0]["composition"] == ["TiO2"]
    assert data["curves"][0]["space_group"] == ["I41/amd"]
    assert data["curves"][0]["lattice_parameters"][0]["c_A"] == 9.51
    assert data["curves"][0]["phases"] == ["anatase"]
    assert data["trends"] == []
    assert data["fwhm"] is None


def test_dataset_dict_emits_empty_placeholders():
    result = FigureAnalysisResult(is_xrd=True, sample=None, curves=[], trends=[])
    data = analysis_to_dataset_dict(result)
    assert data == {
        "sample": None,
        "curves": [],
        "trends": [],
        "composition": [],
        "space_group": [],
        "fwhm": None,
        "lattice_parameters": [],
        "profile_function": None,
    }

    result2 = FigureAnalysisResult(
        is_xrd=True,
        sample="quartz",
        curves=[XrdCurve(peak_positions=[26.6])],
    )
    data2 = analysis_to_dataset_dict(result2)
    assert data2 is not None
    curve = data2["curves"][0]
    assert curve == {
        "condition": None,
        "peak_positions": [26.6],
        "phases": [],
        "peak_width": None,
        "composition": [],
        "space_group": [],
        "fwhm": None,
        "lattice_parameters": [],
        "profile_function": None,
    }
    assert data2["composition"] == []
    assert data2["space_group"] == []
    assert data2["fwhm"] is None
    assert data2["lattice_parameters"] == []
    assert data2["profile_function"] is None
    assert data2["trends"] == []


def test_post_validate_dedupes_and_clips_axis():
    result = FigureAnalysisResult(
        is_xrd=True,
        sample="  ",
        x_min=20.0,
        x_max=80.0,
        composition=["TiO2", "  "],
        space_group=["", "I41/amd"],
        curves=[
            XrdCurve(
                condition="  ",
                peak_positions=[48.0, 25.3, 25.5, 82.0, 37.8],
                phases=["anatase", ""],
                composition=["TiO2", ""],
                space_group=["  ", "P42/mnm"],
                peak_width="  ",
                fwhm=-1.0,
                profile_function="  ",
                lattice_parameters=[LatticeParameters()],
            )
        ],
        trends=["", "peaks sharpen with temperature"],
    )
    out = post_validate(result)
    assert out.sample is None
    assert out.composition == ["TiO2"]
    assert out.space_group == ["I41/amd"]
    assert out.trends == ["peaks sharpen with temperature"]
    curve = out.curves[0]
    assert curve.condition is None
    assert curve.peak_width is None
    assert curve.fwhm is None
    assert curve.profile_function is None
    assert curve.lattice_parameters == []
    assert curve.phases == ["anatase"]
    assert curve.composition == ["TiO2"]
    assert curve.space_group == ["P42/mnm"]
    assert curve.peak_positions == [25.3, 37.8, 48.0]


def test_prompt_mentions_literature_fields():
    prompt = build_user_prompt(figure_number=1, caption="XRD of anatase.")
    assert "fwhm" in prompt.lower() or "FWHM" in SYSTEM_PROMPT
    assert "composition" in SYSTEM_PROMPT.lower()
    assert "space_group" in SYSTEM_PROMPT or "space group" in SYSTEM_PROMPT.lower()
    assert "lattice" in SYSTEM_PROMPT.lower()
    assert "profile_function" in SYSTEM_PROMPT or "profile function" in SYSTEM_PROMPT.lower()
    assert "Never invent" in SYSTEM_PROMPT


def test_paper_analysis_schema_is_strict():
    schema = to_openai_strict_schema(PaperAnalysis.model_json_schema())
    _assert_strict(schema)
    props = schema.get("properties", {})
    assert "summary" in props
    assert "experimental_variables" in props
    assert "key_findings" in props
    assert "scientific_interpretation" in props
    assert "limitations" not in props
    defs = schema.get("$defs", {})
    assert "KeyFinding" in defs
    assert "finding" in defs["KeyFinding"]["properties"]
    assert "figure_ids" in defs["KeyFinding"]["properties"]
    assert "EvidenceCitation" in defs
    assert "ScientificInterpretation" in defs
    assert "Limitation" not in defs
    interp_props = defs["ScientificInterpretation"]["properties"]
    assert "claim" in interp_props
    assert "reasoning_type" in interp_props
    assert "reasoning" in interp_props
    assert "citations" in interp_props
    cite_props = defs["EvidenceCitation"]["properties"]
    assert set(cite_props) >= {"figure_id", "source", "evidence"}


def test_paper_analysis_roundtrip():
    result = PaperAnalysis(
        summary="XRD shows phase-pure anatase.",
        experimental_variables=["annealing temperature"],
        key_findings=[
            KeyFinding(finding="Peaks sharpen with temperature", figure_ids=["fig_1", "fig_2"])
        ],
        scientific_interpretation=[
            ScientificInterpretation(
                claim="Higher annealing increases crystallinity.",
                reasoning_type="ai_inferred",
                reasoning="Peak widths narrow while 2θ positions remain similar.",
                citations=[
                    EvidenceCitation(
                        figure_id="fig_2",
                        source="analysis.trends",
                        evidence="Peaks become sharper with increasing temperature",
                    ),
                    EvidenceCitation(
                        figure_id="fig_2",
                        source="analysis.curves[0].peak_positions",
                        evidence="[25.3, 37.8]",
                    ),
                ],
            )
        ],
    )
    dumped = result.model_dump(mode="json")
    assert "limitations" not in dumped
    assert dumped["key_findings"][0]["figure_ids"] == ["fig_1", "fig_2"]
    assert dumped["scientific_interpretation"][0]["claim"] == (
        "Higher annealing increases crystallinity."
    )
    assert dumped["scientific_interpretation"][0]["reasoning_type"] == "ai_inferred"
    assert dumped["scientific_interpretation"][0]["reasoning"]
    assert dumped["scientific_interpretation"][0]["citations"][0]["figure_id"] == "fig_2"
    assert PaperAnalysis.model_validate(dumped).summary == result.summary


def test_paper_prompt_embeds_payload_and_rules():
    payload = {"title": "Test", "figures": [{"figure_id": "fig_1"}]}
    prompt = build_paper_user_prompt(payload)
    assert "fig_1" in prompt
    assert "Test" in prompt
    assert "identical peak" in PAPER_SYSTEM_PROMPT.lower()
    assert "figure_id" in PAPER_SYSTEM_PROMPT
    assert "author_reported" in PAPER_SYSTEM_PROMPT
    assert "ai_inferred" in PAPER_SYSTEM_PROMPT
    assert "citations" in PAPER_SYSTEM_PROMPT.lower()
    assert "limitations" not in PAPER_SYSTEM_PROMPT.lower()
    assert "claim and reasoning" in prompt.lower() or "claim is assertive" in prompt.lower()
    assert "Never fabricate" in prompt or "never fabricate" in PAPER_SYSTEM_PROMPT.lower()
