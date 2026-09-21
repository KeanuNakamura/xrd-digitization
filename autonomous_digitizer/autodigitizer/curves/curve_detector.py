"""High-level curve detection orchestration."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from autodigitizer.curves.color_segmentation import segment_by_color
from autodigitizer.curves.curve_tracer import (
    calibrate_points,
    continuity_multi_trace,
    mask_to_column_trace,
    resample_points,
    skeleton_trace,
)
from autodigitizer.curves.guided_trace import (
    extract_guided_curves,
    largest_horizontal_component,
    merge_guide_polyline,
)
from autodigitizer.curves.stacked_curves import (
    annotate_offset_metadata,
    detect_vertical_offsets,
    extract_stacked_from_mask,
)
from autodigitizer.models import (
    AxisCalibration,
    CurveSeparation,
    CurveStyle,
    ExtractedCurve,
    PlotArea,
    PlotExtractionGuide,
    SubplotAnalysis,
)
from autodigitizer.processing.masks import build_exclusion_mask, ink_mask
from autodigitizer.processing.preprocessing import save_image

logger = logging.getLogger(__name__)


def extract_curves(
    image: np.ndarray,
    plot_area: PlotArea,
    analysis: SubplotAnalysis,
    x_cal: AxisCalibration,
    y_cal: AxisCalibration,
    *,
    samples: int = 1000,
    strategy: str = "auto",
    color_clusters_override: int | None = None,
    background_threshold: int = 245,
    debug_dir=None,
    subplot_id: str = "A",
    guide: PlotExtractionGuide | None = None,
) -> list[ExtractedCurve]:
    text_regions = guide.text_regions if guide else None
    vguides = guide.vertical_guides_x if guide else None

    excl = build_exclusion_mask(
        image,
        plot_area,
        annotation_regions=None if text_regions else analysis.annotation_regions,
        legend_bbox=analysis.legend_bbox_normalized,
        text_regions_plot=text_regions,
        vertical_guides_x=vguides,
        include_grid=False,  # grid heuristic too aggressive for XRD; use vguides instead
        # Peak spikes look like tall thin CCs — only use heuristic text CC if no LLM text boxes
        include_textlike_cc=not bool(text_regions),
        annotation_is_plot_normalized=False,
    )
    ink = ink_mask(image, excl, luma_threshold=background_threshold)

    if debug_dir is not None:
        save_image(debug_dir / f"{subplot_id}_exclusion_mask.png", (excl.astype(np.uint8) * 255))
        save_image(debug_dir / f"{subplot_id}_curve_mask.png", (ink.astype(np.uint8) * 255))

    sep = analysis.curve_separation
    n = max(1, analysis.estimated_curve_count)

    if strategy == "auto":
        if guide and guide.curves:
            strategy = "guided"
        elif analysis.stacked or analysis.vertically_offset or sep == CurveSeparation.STACKED:
            strategy = "stacked"
        elif n == 1 or sep == CurveSeparation.SINGLE:
            strategy = "single"
        elif sep == CurveSeparation.COLOR:
            strategy = "color"
        elif sep == CurveSeparation.SPATIAL:
            strategy = "stacked"
        else:
            strategy = "continuity"

    # Prefer guided whenever we have a usable guide for multi-curve / annotated plots
    if guide and guide.curves and strategy in ("stacked", "continuity", "color", "single", "auto"):
        if len(guide.curves) >= 1:
            strategy = "guided"

    logger.info("Curve extraction strategy=%s n~%d guide_curves=%d", strategy, n, len(guide.curves) if guide else 0)

    pixel_curves: list[tuple[list[tuple[float, float]], CurveStyle, Optional[str]]] = []

    if strategy == "guided" and guide and guide.curves:
        guided = extract_guided_curves(image, ink, plot_area, guide)
        for i, (pts, cg) in enumerate(guided):
            pts = merge_guide_polyline(pts, cg, plot_area, samples)
            color = tuple(cg.color_rgb[:3]) if cg.color_rgb and len(cg.color_rgb) >= 3 else None
            style = CurveStyle(
                color_rgb=color,  # type: ignore[arg-type]
                line_style=cg.line_style_hint or "solid",
            )
            label = cg.label or _label_at(analysis, i)
            pixel_curves.append((pts, style, label))
            if debug_dir is not None:
                _save_trace_overlay(image, pts, debug_dir / f"{subplot_id}_guided_{i:03d}.png")

    elif strategy == "single":
        cleaned = largest_horizontal_component(ink, plot_area)
        from autodigitizer.curves.guided_trace import upper_envelope_trace

        pts = upper_envelope_trace(cleaned, plot_area)
        if len(pts) < 5:
            pts = upper_envelope_trace(ink, plot_area)
        if len(pts) < 5:
            pts = mask_to_column_trace(ink, plot_area)
        if len(pts) < 5:
            pts = skeleton_trace(ink, plot_area)
        pixel_curves.append((pts, CurveStyle(line_style="solid"), _label_at(analysis, 0)))

    elif strategy == "color":
        clusters = segment_by_color(
            image,
            ink,
            n_clusters=color_clusters_override or (n if n > 1 else None),
            min_pixels=50,
        )
        if not clusters:
            pts = mask_to_column_trace(ink, plot_area)
            pixel_curves.append((pts, CurveStyle(), _label_at(analysis, 0)))
        else:
            if n > 1 and len(clusters) > n + 1:
                clusters = clusters[:n]
            for i, cl in enumerate(clusters):
                pts = mask_to_column_trace(cl.mask, plot_area)
                if len(pts) < 5:
                    pts = skeleton_trace(cl.mask, plot_area)
                style = CurveStyle(color_rgb=cl.center_rgb, line_style="solid")
                pixel_curves.append((pts, style, _label_at(analysis, i)))
                if debug_dir is not None:
                    save_image(
                        debug_dir / f"{subplot_id}_cluster_{i}.png",
                        (cl.mask.astype(np.uint8) * 255),
                    )

    elif strategy == "stacked":
        traces = extract_stacked_from_mask(ink, plot_area, n_curves=n)
        if not traces:
            traces = continuity_multi_trace(ink, plot_area, n_curves=n)
        for i, pts in enumerate(traces):
            pixel_curves.append((pts, CurveStyle(line_style="solid"), _label_at(analysis, i)))

    else:  # continuity
        traces = continuity_multi_trace(ink, plot_area, n_curves=n)
        if len(traces) < n:
            clusters = segment_by_color(image, ink, n_clusters=n, min_pixels=40)
            if clusters:
                traces = [mask_to_column_trace(cl.mask, plot_area) for cl in clusters[:n]]
        for i, pts in enumerate(traces):
            pixel_curves.append((pts, CurveStyle(), _label_at(analysis, i)))

    offset_detected = (
        analysis.vertically_offset
        or analysis.stacked
        or detect_vertical_offsets([p for p, _, _ in pixel_curves])
    )

    curves: list[ExtractedCurve] = []
    for i, (pts, style, label) in enumerate(pixel_curves):
        if len(pts) < 3:
            continue
        # guided path already resampled
        resampled = pts if (strategy == "guided" and len(pts) >= samples // 2) else resample_points(pts, samples)
        if len(resampled) != samples and len(resampled) >= 2:
            resampled = resample_points(resampled, samples)
        data = calibrate_points(resampled, x_cal, y_cal)
        cid = f"{subplot_id}_curve_{i + 1:03d}"
        curve = ExtractedCurve(
            id=cid,
            label=label,
            style=style,
            pixels=resampled,
            data=data,
            confidence=0.75 if strategy == "guided" else 0.6,
        )
        if debug_dir is not None:
            _save_trace_overlay(image, resampled, debug_dir / f"{subplot_id}_curve_{i + 1:03d}_trace.png")
        curves.append(curve)

    annotate_offset_metadata(curves, offset_detected)
    return curves


def _label_at(analysis: SubplotAnalysis, i: int) -> Optional[str]:
    if i < len(analysis.legend_entries):
        return analysis.legend_entries[i].label
    return None


def _save_trace_overlay(image: np.ndarray, points: list[tuple[float, float]], path) -> None:
    vis = image.copy()
    step = max(1, len(points) // 500) if points else 1
    for x, y in points[::step]:
        cv2.circle(vis, (int(x), int(y)), 1, (255, 0, 0), -1)
    save_image(path, vis)
