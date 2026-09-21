"""Tests for subplot / plot-area detection."""

from __future__ import annotations

from autodigitizer.models import FigureAnalysis, FigureLayout, SubplotSummary
from autodigitizer.processing.preprocessing import load_image
from autodigitizer.segmentation.plot_area_detector import detect_plot_area
from autodigitizer.segmentation.subplot_detector import detect_subplots, split_equal_grid
from tests.synthetic import make_four_panels, make_single_curve


def test_plot_area_inside_single(tmp_fig_dir):
    truth = make_single_curve(tmp_fig_dir / "pa.png")
    img = load_image(truth.path)
    area = detect_plot_area(img)
    h, w = img.shape[:2]
    assert 0 <= area.left < area.right <= w
    assert 0 <= area.top < area.bottom <= h
    assert area.width > 0.4 * w
    assert area.height > 0.4 * h


def test_equal_grid_four(tmp_fig_dir):
    truth = make_four_panels(tmp_fig_dir / "g4.png")
    img = load_image(truth.path)
    crops = split_equal_grid(img, 2, 2)
    assert [c.id for c in crops] == ["A", "B", "C", "D"]
    # Non-overlapping roughly
    boxes = [c.bbox_normalized for c in crops]
    assert boxes[0][2] <= boxes[1][0] + 0.05


def test_detect_subplots_uses_llm_boxes(tmp_fig_dir):
    truth = make_four_panels(tmp_fig_dir / "d4.png")
    img = load_image(truth.path)
    analysis = FigureAnalysis(
        digitizable=True,
        figure_type="multi_panel",
        subplot_count=4,
        layout=FigureLayout(rows=2, columns=2),
        subplots=[
            SubplotSummary(id="A", bbox_normalized=[0.0, 0.0, 0.5, 0.5]),
            SubplotSummary(id="B", bbox_normalized=[0.5, 0.0, 1.0, 0.5]),
            SubplotSummary(id="C", bbox_normalized=[0.0, 0.5, 0.5, 1.0]),
            SubplotSummary(id="D", bbox_normalized=[0.5, 0.5, 1.0, 1.0]),
        ],
    )
    crops = detect_subplots(img, analysis)
    assert len(crops) == 4
    assert all(c.image.size > 0 for c in crops)
