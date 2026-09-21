"""Decide whether a panel needs ClipDrop cleaning before curve tracing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import cv2
import numpy as np

CleaningStrategy = Literal["none", "clipdrop_full", "targeted"]


@dataclass
class CleaningDecision:
    cleaning_strategy: CleaningStrategy
    cleaning_reason: str
    needs_cleaning: bool
    curve_color_mode: str  # "colored" | "gray_black" | "mixed" | "unknown"
    text_interference_score: float
    colored_curve_fraction: float
    axis_source: str = "original_panel"
    digitization_source: str = "original"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _saturation_stats(image_bgr: np.ndarray) -> tuple[float, float, float]:
    """Return (colored_ink_frac, dark_ink_frac, mean_sat_of_ink)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    colored = (sat > 40) & (gray < 245)
    dark = (gray < 120) & (sat <= 40)
    ink = colored | dark
    ink_count = float(max(1, int(ink.sum())))
    colored_frac = float(colored.sum()) / ink_count
    dark_frac = float(dark.sum()) / ink_count
    mean_sat = float(sat[ink].mean()) if ink.any() else 0.0
    return colored_frac, dark_frac, mean_sat


def _text_like_dark_on_colored(
    image_bgr: np.ndarray,
    *,
    plot_bbox: tuple[int, int, int, int] | None = None,
) -> float:
    """
    Heuristic interference score in [0, 1].

    High when dark/low-sat ink (text-like) sits near high-sat curve ink inside
    the plot interior.
    """
    if plot_bbox is not None:
        x0, y0, x1, y1 = plot_bbox
        region = image_bgr[y0:y1, x0:x1]
    else:
        region = image_bgr
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    colored = (sat > 40) & (gray < 245)
    text_like = (gray < 90) & (sat < 35)
    if not colored.any():
        # Gray/black curves: text and curve share similar ink → high interference.
        return 0.85 if text_like.mean() > 0.004 else 0.55

    # Dilate colored strokes; measure text-like pixels adjacent to curves.
    kernel = np.ones((5, 5), np.uint8)
    near_curve = cv2.dilate(colored.astype(np.uint8), kernel, iterations=2).astype(bool)
    adjacent_text = text_like & near_curve & ~colored
    # Score by density of adjacent text relative to curve mass.
    score = float(adjacent_text.sum()) / float(max(1, colored.sum()))
    return float(np.clip(score * 8.0, 0.0, 1.0))


def decide_cleaning_strategy(
    image_bgr: np.ndarray,
    *,
    plot_bbox: tuple[int, int, int, int] | None = None,
    triage_needs_clipdrop: bool | None = None,
    force_clean: bool = False,
    force_none: bool = False,
) -> CleaningDecision:
    """
    Choose cleaning strategy from color/text interference, not mere text presence.

    - Strongly colored curves with low adjacent black text → ``none``
    - Gray/black curves or high text interference → ``clipdrop_full``
    """
    if force_none:
        return CleaningDecision(
            cleaning_strategy="none",
            cleaning_reason="forced_none",
            needs_cleaning=False,
            curve_color_mode="unknown",
            text_interference_score=0.0,
            colored_curve_fraction=0.0,
            digitization_source="original",
        )
    if force_clean:
        return CleaningDecision(
            cleaning_strategy="clipdrop_full",
            cleaning_reason="forced_clean",
            needs_cleaning=True,
            curve_color_mode="unknown",
            text_interference_score=1.0,
            colored_curve_fraction=0.0,
            digitization_source="cleaned",
        )

    region = image_bgr
    if plot_bbox is not None:
        x0, y0, x1, y1 = plot_bbox
        region = image_bgr[y0:y1, x0:x1]
    colored_frac, dark_frac, _mean_sat = _saturation_stats(region)
    interference = _text_like_dark_on_colored(image_bgr, plot_bbox=plot_bbox)

    if colored_frac >= 0.55 and dark_frac < 0.45:
        mode = "colored"
    elif dark_frac >= 0.55 and colored_frac < 0.35:
        mode = "gray_black"
    elif colored_frac >= 0.25 and dark_frac >= 0.25:
        mode = "mixed"
    else:
        mode = "unknown"

    # Triage needs_clipdrop is only a soft hint — never sufficient alone.
    if mode == "colored" and interference < 0.35:
        return CleaningDecision(
            cleaning_strategy="none",
            cleaning_reason=(
                f"colored_curves_separable interference={interference:.2f}"
                + ("; triage_hint_ignored" if triage_needs_clipdrop else "")
            ),
            needs_cleaning=False,
            curve_color_mode=mode,
            text_interference_score=interference,
            colored_curve_fraction=colored_frac,
            digitization_source="original",
        )

    if mode == "gray_black" or interference >= 0.45 or (
        mode == "mixed" and interference >= 0.30
    ):
        reason_bits = [f"mode={mode}", f"interference={interference:.2f}"]
        if triage_needs_clipdrop:
            reason_bits.append("triage_hint")
        # Anti-aliased black XRD traces are damaged by aggressive text removal.
        # Prefer digitizing the original; ClipDrop remains optional for display.
        if mode == "gray_black":
            return CleaningDecision(
                cleaning_strategy="none",
                cleaning_reason="; ".join(reason_bits + ["digitize_original_gray_black"]),
                needs_cleaning=False,
                curve_color_mode=mode,
                text_interference_score=interference,
                colored_curve_fraction=colored_frac,
                digitization_source="original",
            )
        return CleaningDecision(
            cleaning_strategy="clipdrop_full",
            cleaning_reason="; ".join(reason_bits),
            needs_cleaning=True,
            curve_color_mode=mode,
            text_interference_score=interference,
            colored_curve_fraction=colored_frac,
            digitization_source="cleaned",
        )

    # Borderline colored with mild interference: still prefer original.
    return CleaningDecision(
        cleaning_strategy="none",
        cleaning_reason=f"borderline_prefer_original interference={interference:.2f}",
        needs_cleaning=False,
        curve_color_mode=mode,
        text_interference_score=interference,
        colored_curve_fraction=colored_frac,
        digitization_source="original",
    )
