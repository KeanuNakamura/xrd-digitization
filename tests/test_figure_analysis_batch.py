"""Tests for figure_analysis batch processing (mocked, no live API)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "autonomous_digitizer"
for path in (ROOT, AUTO):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from figure_analysis.batch import discover_images, run_batch
from figure_analysis.io_util import analysis_json_path, load_analysis_if_valid
from figure_analysis.schemas import FigureAnalysisResult, XrdCurve


def _make_png(path: Path, color: tuple[int, int, int] = (255, 255, 255)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return path


def _fake_result() -> FigureAnalysisResult:
    return FigureAnalysisResult(
        is_xrd=True,
        sample="0.5 wt.% Mg-TiO2",
        x_min=20.0,
        x_max=80.0,
        curves=[
            XrdCurve(
                condition="700 °C",
                peak_positions=[25.3, 27.4, 37.8],
                phases=["anatase", "rutile"],
                peak_width="narrow",
            )
        ],
        trends=["peaks sharpen with increasing calcination temperature"],
    )


def test_discover_images_recursive(tmp_path: Path):
    _make_png(tmp_path / "a.png")
    (tmp_path / "sub").mkdir(parents=True, exist_ok=True)
    _make_png(tmp_path / "sub" / "b.jpg")
    (tmp_path / "sub" / "note.txt").write_text("x")
    found = discover_images(tmp_path, recursive=True)
    assert [p.name for p in found] == ["a.png", "b.jpg"]


def test_batch_resume_and_failure_isolation(tmp_path: Path):
    _make_png(tmp_path / "inputs" / "figure_ok.png")
    _make_png(tmp_path / "inputs" / "figure_bad.png")
    _make_png(tmp_path / "inputs" / "figure_skip.png")
    out = tmp_path / "out"

    skip_result = _fake_result()
    skip_json = analysis_json_path(out, "figure_skip")
    skip_json.parent.mkdir(parents=True, exist_ok=True)
    # Write in sidecar shape used by write_json.
    skip_json.write_text(
        json.dumps({"is_xrd": True, "analysis": {
            "sample": skip_result.sample,
            "curves": [c.model_dump(mode="json") for c in skip_result.curves],
            "trends": skip_result.trends,
        }}, indent=2),
        encoding="utf-8",
    )

    calls: list[str] = []

    def mock_analyze(image_path, **kwargs):
        calls.append(Path(image_path).name)
        if Path(image_path).name == "figure_bad.png":
            raise RuntimeError("boom")
        return _fake_result()

    stats = run_batch(
        tmp_path / "inputs",
        out,
        workers=2,
        overwrite=False,
        write_summary=True,
        analyze_fn=mock_analyze,
        client=object(),
    )

    assert stats.total == 3
    assert stats.succeeded == 1
    assert stats.skipped == 1
    assert stats.failed == 1

    ok_json = analysis_json_path(out, "figure_ok")
    loaded = load_analysis_if_valid(ok_json)
    assert loaded is not None
    assert loaded.is_xrd is True
    assert loaded.curves[0].peak_positions[0] == 25.3

    dumped = json.loads(ok_json.read_text(encoding="utf-8"))
    assert dumped["is_xrd"] is True
    assert dumped["analysis"]["curves"][0]["peak_positions"] == [25.3, 27.4, 37.8]

    calls.clear()
    stats2 = run_batch(
        tmp_path / "inputs",
        out,
        workers=1,
        overwrite=True,
        write_summary=False,
        analyze_fn=mock_analyze,
        client=object(),
    )
    assert stats2.skipped == 0
    assert "figure_skip.png" in calls


def test_enrich_removes_non_xrd_crops(tmp_path: Path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "legacy"))
    import scrape_and_digitize as sad

    paper = tmp_path / "clay"
    figures = paper / "figures"
    figures.mkdir(parents=True)
    fig1 = figures / "fig_1.png"
    fig3 = figures / "fig_3.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(fig1)
    Image.new("RGB", (8, 8), (0, 255, 0)).save(fig3)
    fa = paper / "clay.figure_analysis.json"
    fa.write_text(
        json.dumps(
            {
                "title": "Clay paper",
                "doi": "10.1/x",
                "authors": ["A Author"],
                "figures": [
                    {
                        "figure": 1,
                        "figure_path": str(fig1),
                        "caption": "Map",
                        "text": "sampling map",
                    },
                    {
                        "figure": 3,
                        "figure_path": str(fig3),
                        "caption": "XRD",
                        "text": "diffraction pattern",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_analyze(image_path, **kwargs):
        name = Path(image_path).name
        if name == "fig_1.png":
            return FigureAnalysisResult(is_xrd=False)
        return FigureAnalysisResult(
            is_xrd=True,
            sample="clay",
            curves=[
                XrdCurve(
                    condition=None,
                    peak_positions=[12.4, 26.6],
                    phases=["kaolinite", "quartz"],
                    peak_width=None,
                )
            ],
            trends=[],
        )

    sad.analyze_figure_image = fake_analyze
    sad.enrich_figure_analysis_json(paper, client=object())

    data = json.loads(fa.read_text(encoding="utf-8"))
    fig_nums = [e["figure"] for e in data["figures"]]
    assert fig_nums == [3]
    assert data["figures"][0]["is_xrd"] is True
    assert data["figures"][0]["analysis"]["curves"][0]["peak_positions"] == [12.4, 26.6]
    assert not fig1.exists()
    assert fig3.exists()


@pytest.mark.skipif(
    not __import__("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_live_analyze_data_analysis_figure():
    from figure_analysis.analyzer import analyze_figure_image

    sample = ROOT / "data" / "analysis" / "figure_1.png"
    if not sample.is_file():
        pytest.skip("data/analysis/figure_1.png missing")

    result = analyze_figure_image(sample, figure_number=1)
    assert isinstance(result.is_xrd, bool)
