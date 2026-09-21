"""Tests for OpenAI-guided band tracing (no API)."""

from __future__ import annotations

import numpy as np

from autodigitizer.curves.guided_trace import (
    extract_guided_curves,
    guided_column_trace,
    largest_horizontal_component,
)
from autodigitizer.models import CurveGuide, PlotArea, PlotExtractionGuide
from autodigitizer.processing.masks import build_exclusion_mask, ink_mask


def test_guided_bands_separate_horizontal_lines():
    h, w = 200, 300
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    # three colored horizontal curves
    colors = [(180, 40, 40), (40, 160, 40), (40, 40, 180)]
    ys = [40, 100, 160]
    for y, c in zip(ys, colors):
        img[y - 1 : y + 2, 20:280] = c

    pa = PlotArea(left=20, right=280, top=10, bottom=190)
    excl = build_exclusion_mask(img, pa, include_textlike_cc=False)
    ink = ink_mask(img, excl)

    guide = PlotExtractionGuide(
        curves=[
            CurveGuide(
                label="a",
                color_rgb=list(colors[0]),
                y_band_normalized=[0.05, 0.3],
                key_points_normalized=[[0.1, 0.17], [0.5, 0.17], [0.9, 0.17]],
            ),
            CurveGuide(
                label="b",
                color_rgb=list(colors[1]),
                y_band_normalized=[0.35, 0.6],
                key_points_normalized=[[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]],
            ),
            CurveGuide(
                label="c",
                color_rgb=list(colors[2]),
                y_band_normalized=[0.65, 0.95],
                key_points_normalized=[[0.1, 0.83], [0.5, 0.83], [0.9, 0.83]],
            ),
        ]
    )
    traces = extract_guided_curves(img, ink, pa, guide)
    assert len(traces) == 3
    means = [np.mean([p[1] for p in pts]) for pts, _ in traces if pts]
    assert means[0] < means[1] < means[2]


def test_largest_component_drops_text_blob():
    h, w = 120, 200
    ink = np.zeros((h, w), dtype=bool)
    pa = PlotArea(left=10, right=190, top=10, bottom=110)
    # long curve
    ink[60, 15:185] = True
    # text-like blob
    ink[20:35, 40:48] = True
    cleaned = largest_horizontal_component(ink, pa)
    assert cleaned[60, 100]
    assert cleaned[25, 44] == False or cleaned[60, :].sum() > cleaned[20:35, 40:48].sum()
