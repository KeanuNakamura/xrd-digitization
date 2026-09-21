"""OpenAI extraction guide: text regions, vertical guides, per-curve bands/keypoints."""

from __future__ import annotations

import logging

import numpy as np

from autodigitizer.models import PlotArea, PlotExtractionGuide, SubplotAnalysis
from autodigitizer.vision.openai_client import OpenAIVisionClient

logger = logging.getLogger(__name__)

SYSTEM = (
    "You guide quantitative digitization of a scientific subplot. "
    "Return JSON only. All coordinates are normalized to the FULL IMAGE you see "
    "(0,0)=top-left of the image, (1,1)=bottom-right. "
    "Do NOT invent dense curve samples. Provide sparse, accurate guidance only."
)


def request_extraction_guide(
    client: OpenAIVisionClient,
    *,
    crop: np.ndarray,
    analysis: SubplotAnalysis,
    cache_key: str,
) -> PlotExtractionGuide:
    logger.info("Requesting OpenAI extraction guide for subplot %s", analysis.subplot_id)
    n = max(1, analysis.estimated_curve_count)
    labels = [e.label for e in analysis.legend_entries]
    prompt = f"""Plan CV extraction for subplot '{analysis.subplot_id}'.

Known analysis:
- plot_type={analysis.plot_type}
- estimated_curve_count={n}
- curve_separation={analysis.curve_separation.value}
- stacked={analysis.stacked} vertically_offset={analysis.vertically_offset}
- legend labels (top→bottom if offset): {labels}
- annotations_present={analysis.annotations_present}

Coordinate system: normalized to THIS IMAGE (0..1). Not data units.

Provide:
1) text_regions: tight boxes around in-plot TEXT ONLY (Miller indices, nm labels, titles).
   Keep boxes on glyphs; do NOT cover peak tips or curve strokes.
2) vertical_guides_x: x positions of dashed/solid vertical guide lines to ignore (empty if none).
3) curves: one entry per visible data series (~{n}). For each:
   - label (match legend / nm labels when possible; order top→bottom for offset plots)
   - color_rgb approximate [R,G,B] of the LINE (not text)
   - y_band_normalized [y0,y1]: horizontal strip in IMAGE coords containing ONLY that series
     (for waterfall/offset plots, bands must be non-overlapping and ordered top→bottom)
   - key_points_normalized: 12–40 points [[x,y],...] ON the curve stroke in IMAGE coords,
     including every major peak tip and several baseline samples. Prefer peak tips especially.
   - line_style_hint if useful

For a single black XRD curve with Miller labels above peaks: text_regions around labels only;
key_points must sit on the black curve (peak tips), never on the text.
For similar-color stacked spectra: rely on distinct y_band_normalized bands; key_points per band.
"""
    guide = client.analyze_structured(
        prompt=prompt,
        schema_model=PlotExtractionGuide,
        images=[crop],
        cache_key=cache_key,
        cache_name=f"subplot_{analysis.subplot_id}_extraction_guide_v2.json",
        system=SYSTEM,
    )
    cleaned = []
    for c in guide.curves:
        if c.color_rgb is not None and len(c.color_rgb) >= 3:
            c.color_rgb = [int(np.clip(v, 0, 255)) for v in c.color_rgb[:3]]
        if len(c.y_band_normalized) != 2:
            c.y_band_normalized = [0.0, 1.0]
        ya, yb = float(c.y_band_normalized[0]), float(c.y_band_normalized[1])
        y0, y1 = (ya, yb) if ya <= yb else (yb, ya)
        c.y_band_normalized = [max(0.0, y0), min(1.0, y1)]
        pts = []
        for p in c.key_points_normalized:
            if len(p) >= 2:
                pts.append([float(np.clip(p[0], 0, 1)), float(np.clip(p[1], 0, 1))])
        c.key_points_normalized = pts
        cleaned.append(c)
    guide.curves = cleaned
    logger.info(
        "Guide: %d curves, %d text regions, %d vertical guides",
        len(guide.curves),
        len(guide.text_regions),
        len(guide.vertical_guides_x),
    )
    return guide


def normalize_box_xyxy(box: list[float]) -> list[float]:
    """Accept [x0,y0,x1,y1] or accidental [x,y,w,h] from the model."""
    if len(box) != 4:
        return box
    a, b, c, d = [float(v) for v in box]
    # Already ordered xyxy
    if c >= a and d >= b:
        # Heuristic: tiny second pair may still be w,h if both dimensions look like sizes
        if c - a < 1e-6 and d - b < 1e-6:
            return [a, b, min(1.0, a + max(c, 0.02)), min(1.0, b + max(d, 0.02))]
        return [a, b, c, d]
    # Unordered or xywh: if c,d look like small positive sizes, treat as w,h
    if 0 < c < 0.45 and 0 < d < 0.45 and a + c <= 1.15 and b + d <= 1.15:
        return [a, b, min(1.0, a + c), min(1.0, b + d)]
    return [min(a, c), min(b, d), max(a, c), max(b, d)]


def crop_norm_to_plot_norm(
    x_n: float, y_n: float, image_shape: tuple[int, ...], plot_area: PlotArea
) -> tuple[float, float]:
    """Convert image/crop-normalized coords to plot-interior normalized coords."""
    h, w = image_shape[:2]
    x_px = x_n * w
    y_px = y_n * h
    xp = (x_px - plot_area.left) / max(1, plot_area.width)
    yp = (y_px - plot_area.top) / max(1, plot_area.height)
    return float(xp), float(yp)


def guide_to_plot_space(
    guide: PlotExtractionGuide, image_shape: tuple[int, ...], plot_area: PlotArea
) -> PlotExtractionGuide:
    """Rewrite a crop-normalized guide into plot-normalized coordinates for CV."""

    def conv_box(box: list[float]) -> list[float]:
        box = normalize_box_xyxy(box)
        if len(box) != 4:
            return box
        x0, y0 = crop_norm_to_plot_norm(box[0], box[1], image_shape, plot_area)
        x1, y1 = crop_norm_to_plot_norm(box[2], box[3], image_shape, plot_area)
        return [
            float(np.clip(min(x0, x1), -0.05, 1.05)),
            float(np.clip(min(y0, y1), -0.05, 1.05)),
            float(np.clip(max(x0, x1), -0.05, 1.05)),
            float(np.clip(max(y0, y1), -0.05, 1.05)),
        ]

    text = [conv_box(b) for b in guide.text_regions]
    vguides = []
    for xn in guide.vertical_guides_x:
        xp, _ = crop_norm_to_plot_norm(float(xn), 0.5, image_shape, plot_area)
        if 0.0 <= xp <= 1.0:
            vguides.append(xp)

    curves = []
    for c in guide.curves:
        y0, y1 = c.y_band_normalized if len(c.y_band_normalized) == 2 else [0.0, 1.0]
        _, py0 = crop_norm_to_plot_norm(0.5, y0, image_shape, plot_area)
        _, py1 = crop_norm_to_plot_norm(0.5, y1, image_shape, plot_area)
        py0, py1 = sorted([float(np.clip(py0, 0, 1)), float(np.clip(py1, 0, 1))])
        if py1 - py0 < 0.04:
            mid = 0.5 * (py0 + py1)
            py0, py1 = max(0.0, mid - 0.03), min(1.0, mid + 0.03)
        kps = []
        for p in c.key_points_normalized:
            if len(p) < 2:
                continue
            xp, yp = crop_norm_to_plot_norm(p[0], p[1], image_shape, plot_area)
            if -0.05 <= xp <= 1.05 and -0.05 <= yp <= 1.05:
                kps.append([float(np.clip(xp, 0, 1)), float(np.clip(yp, 0, 1))])
        curves.append(
            c.model_copy(
                update={
                    "y_band_normalized": [py0, py1],
                    "key_points_normalized": kps,
                }
            )
        )

    return PlotExtractionGuide(
        text_regions=text,
        vertical_guides_x=vguides,
        curves=curves,
        notes=list(guide.notes) + ["coords_converted_crop_to_plot"],
    )
