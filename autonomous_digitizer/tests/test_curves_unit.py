"""Tests for color segmentation and curve tracing."""

from __future__ import annotations

import numpy as np

from autodigitizer.curves.color_segmentation import segment_by_color
from autodigitizer.curves.curve_tracer import continuity_multi_trace, mask_to_column_trace, resample_points
from autodigitizer.models import PlotArea


def _blank(h=120, w=200):
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    return img


def test_color_segmentation_separates_rgb_lines():
    img = _blank()
    # three horizontal colored lines
    img[30, 20:180] = (220, 20, 20)
    img[60, 20:180] = (20, 180, 20)
    img[90, 20:180] = (20, 20, 220)
    ink = np.zeros(img.shape[:2], dtype=bool)
    ink[30, 20:180] = True
    ink[60, 20:180] = True
    ink[90, 20:180] = True
    # thicken
    for y in (30, 60, 90):
        ink[y - 1 : y + 2, 20:180] = True
        img[y - 1 : y + 2, 20:180] = img[y, 20]

    clusters = segment_by_color(img, ink, n_clusters=3, min_pixels=20)
    assert len(clusters) >= 3


def test_column_trace_and_resample():
    mask = np.zeros((100, 150), dtype=bool)
    for x in range(10, 140):
        y = int(50 + 20 * np.sin(x / 15))
        mask[y - 1 : y + 2, x] = True
    area = PlotArea(left=0, right=150, top=0, bottom=100)
    pts = mask_to_column_trace(mask, area)
    assert len(pts) > 50
    rs = resample_points(pts, 100)
    assert len(rs) == 100


def test_continuity_multi_trace():
    mask = np.zeros((120, 160), dtype=bool)
    for x in range(5, 155):
        mask[30, x] = True
        mask[70, x] = True
        mask[100, x] = True
    area = PlotArea(left=0, right=160, top=0, bottom=120)
    curves = continuity_multi_trace(mask, area, n_curves=3, max_jump=10)
    assert len(curves) == 3
