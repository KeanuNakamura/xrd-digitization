"""Tests for OpenAI strict JSON schema conversion."""

from __future__ import annotations

from autodigitizer.models import FigureAnalysis, SubplotAnalysis, VisionQAResponse, PlotExtractionGuide
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


def test_figure_analysis_schema_is_strict():
    schema = to_openai_strict_schema(FigureAnalysis.model_json_schema())
    _assert_strict(schema)
    assert "shared_x_groups" in schema["$defs"]["AxisRelationships"]["required"]
    assert "layout" in schema["required"]


def test_subplot_analysis_schema_is_strict():
    schema = to_openai_strict_schema(SubplotAnalysis.model_json_schema())
    _assert_strict(schema)


def test_vision_qa_schema_is_strict():
    schema = to_openai_strict_schema(VisionQAResponse.model_json_schema())
    _assert_strict(schema)
    assert "metrics" not in schema.get("properties", {})


def test_extraction_guide_schema_is_strict():
    schema = to_openai_strict_schema(PlotExtractionGuide.model_json_schema())
    _assert_strict(schema)