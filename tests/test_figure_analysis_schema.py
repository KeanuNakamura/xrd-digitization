"""Tests for compact XRD figure_analysis schemas and OpenAI strict compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "autonomous_digitizer"
for path in (ROOT, AUTO):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from figure_analysis.analyzer import analysis_to_dataset_dict, post_validate
from figure_analysis.prompts import SYSTEM_PROMPT, build_user_prompt
from figure_analysis.schemas import (
    FigureAnalysisPayload,
    FigureAnalysisResult,
    LatticeParameters,
    XrdCurve,
)
from autodigitizer.vision.openai_client import to_openai_strict_schema


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
    assert "fwhm" in props
    assert "lattice_parameters" in props
    assert "profile_function" in props
    defs = schema.get("$defs", {})
    assert "XrdCurve" in defs
    assert "LatticeParameters" in defs
    assert "fwhm" in defs["XrdCurve"]["properties"]
    assert "lattice_parameters" in defs["XrdCurve"]["properties"]
    assert "profile_function" in defs["XrdCurve"]["properties"]


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
        profile_function="Pseudo-Voigt",
        lattice_parameters=[
            LatticeParameters(phase="anatase", a_A=3.78, c_A=9.51, space_group="I41/amd")
        ],
        curves=[
            XrdCurve(
                condition="700 °C",
                peak_positions=[25.3, 37.8],
                phases=["anatase"],
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
    assert data["profile_function"] == "Pseudo-Voigt"
    assert data["lattice_parameters"][0]["a_A"] == 3.78
    assert data["lattice_parameters"][0]["b_A"] is None
    assert data["curves"][0]["fwhm"] == 0.22
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
        "fwhm": None,
        "lattice_parameters": [],
        "profile_function": None,
    }
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
        curves=[
            XrdCurve(
                condition="  ",
                peak_positions=[48.0, 25.3, 25.5, 82.0, 37.8],
                phases=["anatase", ""],
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
    assert out.trends == ["peaks sharpen with temperature"]
    curve = out.curves[0]
    assert curve.condition is None
    assert curve.peak_width is None
    assert curve.fwhm is None
    assert curve.profile_function is None
    assert curve.lattice_parameters == []
    assert curve.phases == ["anatase"]
    assert curve.peak_positions == [25.3, 37.8, 48.0]


def test_prompt_mentions_literature_fields():
    prompt = build_user_prompt(figure_number=1, caption="XRD of anatase.")
    assert "fwhm" in prompt.lower() or "FWHM" in SYSTEM_PROMPT
    assert "lattice" in SYSTEM_PROMPT.lower()
    assert "profile_function" in SYSTEM_PROMPT or "profile function" in SYSTEM_PROMPT.lower()
    assert "Never invent" in SYSTEM_PROMPT
