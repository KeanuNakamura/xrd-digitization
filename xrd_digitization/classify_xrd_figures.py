from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from xrd_digitization.types import ClassificationResult, FigureContext

XRD_CAPTION_KEYWORDS = {
    "xrd",
    "x-ray diffraction",
    "x ray diffraction",
    "powder diffraction",
    "diffraction pattern",
    "diffractogram",
    "diffractograms",
    "2θ",
    "2 theta",
    "two theta",
    "2-theta",
    "rietveld",
    "bragg",
    "pxrd",
    "synchrotron diffraction",
}

RIETVELD_KEYWORDS = {
    "rietveld",
    "observed",
    "calculated",
    "difference",
    "residual",
}

MULTI_PANEL_PATTERNS = (
    re.compile(r"\(\s*[a-d]\s*\)", re.IGNORECASE),
    re.compile(r"\b[a-d]\s*\)", re.IGNORECASE),
    re.compile(r"panel\s+[a-d]\b", re.IGNORECASE),
)


def _caption_keyword_score(caption: str | None) -> tuple[float, list[str]]:
    if not caption:
        return 0.0, []

    text = caption.lower()
    hits: list[str] = []
    score = 0.0

    for keyword in XRD_CAPTION_KEYWORDS:
        if keyword in text:
            hits.append(keyword)
            score += 1.5 if keyword in {"xrd", "x-ray diffraction", "2θ", "2 theta"} else 1.0

    return score, hits


def _image_heuristic_score(image_bgr: np.ndarray) -> tuple[float, list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0.0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if height < 120 or width < 120:
        warnings.append("low_resolution_image")
        score -= 0.5

    aspect = width / max(height, 1)
    if 1.1 <= aspect <= 2.5:
        score += 0.75
        reasons.append("plot_like_aspect_ratio")

    # Bottom band often contains 2theta labels in XRD plots.
    bottom_band = gray[int(height * 0.82) :, :]
    _, bottom_bin = cv2.threshold(bottom_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bottom_ink = float(np.count_nonzero(bottom_bin)) / bottom_bin.size
    if 0.01 <= bottom_ink <= 0.12:
        score += 0.75
        reasons.append("axis_label_band_detected")

    # Peak-like vertical structure in the plot interior.
    interior = gray[int(height * 0.15) : int(height * 0.82), int(width * 0.12) : int(width * 0.95)]
    if interior.size:
        edges = cv2.Canny(interior, 50, 150)
        col_density = edges.mean(axis=0)
        peaks = int(np.sum(col_density > np.percentile(col_density, 85)))
        if peaks >= 3:
            score += 1.0
            reasons.append("multiple_vertical_peak_structures")

    # Detect likely multi-panel layout via vertical whitespace valleys.
    col_var = gray.var(axis=0)
    low_var_cols = col_var < np.percentile(col_var, 20)
    gaps = []
    in_gap = False
    start = 0
    for idx, is_low in enumerate(low_var_cols):
        if is_low and not in_gap:
            in_gap = True
            start = idx
        elif not is_low and in_gap:
            if idx - start > width * 0.04:
                gaps.append((start, idx))
            in_gap = False
    if len(gaps) >= 1:
        warnings.append("possible_multi_panel_figure")
        score -= 0.25

    return score, reasons, warnings


def classify_xrd_figure(
    image_path: str | Path,
    *,
    caption: str | None = None,
    min_confidence: float = 1.5,
) -> ClassificationResult:
    """
    Classify whether an extracted figure image is likely an XRD plot.

    Uses caption keywords when available and simple image heuristics otherwise.
    """
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return ClassificationResult(
            is_xrd=False,
            confidence=0.0,
            reasons=["failed_to_load_image"],
            warnings=["failed_to_load_image"],
        )

    caption_score, caption_hits = _caption_keyword_score(caption)
    image_score, image_reasons, image_warnings = _image_heuristic_score(image_bgr)

    reasons = []
    if caption_hits:
        reasons.append(f"caption_keywords:{','.join(caption_hits[:5])}")
    reasons.extend(image_reasons)

    warnings = list(image_warnings)
    if caption and any(k in caption.lower() for k in RIETVELD_KEYWORDS):
        warnings.append("possible_rietveld_or_multi_curve_plot")
    if caption and any(p.search(caption) for p in MULTI_PANEL_PATTERNS):
        warnings.append("multi_panel_caption_pattern")

    combined = caption_score + image_score
    confidence = min(1.0, combined / 4.0)
    is_xrd = combined >= min_confidence

    return ClassificationResult(
        is_xrd=is_xrd,
        confidence=confidence,
        caption_score=caption_score,
        image_score=image_score,
        reasons=reasons,
        warnings=warnings,
    )


def classify_from_context(context: FigureContext, **kwargs: object) -> ClassificationResult:
    return classify_xrd_figure(context.image_path, caption=context.caption, **kwargs)
