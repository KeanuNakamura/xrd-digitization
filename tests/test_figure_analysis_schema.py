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
    assert "peak_assignments" not in props
    assert "plots" not in props


def test_non_xrd_yields_null_analysis():
    result = FigureAnalysisResult(is_xrd=False, sample="should clear", curves=[XrdCurve()])
    out = post_validate(result)
    assert out.is_xrd is False
    assert out.sample is None
    assert out.curves == []
    assert analysis_to_dataset_dict(out) is None


def test_dataset_dict_xrd():
    result = FigureAnalysisResult(
        is_xrd=True,
        sample="clay sample",
        x_min=5.0,
        x_max=70.0,
        curves=[
            XrdCurve(
                condition="clay oriented",
                peak_positions=[12.4, 20.9, 26.6],
                phases=["kaolinite", "quartz"],
                peak_width="narrow",
            )
        ],
        trends=[],
    )
    data = analysis_to_dataset_dict(result)
    assert data is not None
    assert "is_xrd" not in data
    assert "x_min" not in data
    assert data["sample"] == "clay sample"
    assert data["curves"][0]["peak_positions"] == [12.4, 20.9, 26.6]


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
    assert curve.phases == ["anatase"]
    assert curve.peak_positions == [25.3, 37.8, 48.0]


def test_prompt_requires_xrd_gate():
    prompt = build_user_prompt(figure_number=1, caption="Map of sampling sites.")
    assert "is_xrd" in prompt or "is_xrd" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT
    assert "photograph" in SYSTEM_PROMPT.lower() or "map" in SYSTEM_PROMPT.lower()
