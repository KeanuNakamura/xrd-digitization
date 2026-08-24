"""Persist and reload axis calibration for text-stripped (e.g. ClipDrop) figures.

Calibrate on the original image, save a sidecar, then digitize the cleaned PNG
by reusing the saved crop bbox and AxisCalibrationResult — never re-OCR or
re-crop the cleaned image.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xrd_digitization.calibrate_axes import (
    _select_arithmetic_number_sequence,
    calibrate_axes,
)
from xrd_digitization.crop_plot_area import crop_plot_area
from xrd_digitization.types import AxisCalibrationResult, PlotCropResult

LOGGER = logging.getLogger(__name__)

MIN_OCR_TICK_PAIRS = 2
MIN_OCR_CONFIDENCE = 0.55
SIDECAR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AxisSidecar:
    """Saved axis calibration plus the geometry needed to reuse it on a clean image."""

    calibration: AxisCalibrationResult
    image_width: int
    image_height: int
    plot_crop_bbox: tuple[int, int, int, int]
    source_image: str
    warnings: list[str]


class AxisSidecarError(ValueError):
    """Invalid or unusable axis sidecar."""


def axis_calibration_to_dict(calibration: AxisCalibrationResult) -> dict[str, Any]:
    return {
        "x_min": calibration.x_min,
        "x_max": calibration.x_max,
        "plot_left": calibration.plot_left,
        "plot_right": calibration.plot_right,
        "plot_top": calibration.plot_top,
        "plot_bottom": calibration.plot_bottom,
        "method": calibration.method,
        "confidence": calibration.confidence,
        "tick_pairs": [
            {"pixel_x": int(px), "two_theta": float(val)}
            for px, val in calibration.tick_pairs
        ],
        "y_min": calibration.y_min,
        "y_max": calibration.y_max,
        "y_tick_pairs": [
            {"pixel_y": int(py), "intensity": float(val)}
            for py, val in calibration.y_tick_pairs
        ],
        "y_method": calibration.y_method,
        "warnings": list(calibration.warnings),
    }


def axis_calibration_from_dict(payload: dict[str, Any]) -> AxisCalibrationResult:
    tick_pairs = [
        (int(item["pixel_x"]), float(item["two_theta"]))
        for item in payload.get("tick_pairs") or []
    ]
    y_tick_pairs = [
        (int(item["pixel_y"]), float(item["intensity"]))
        for item in payload.get("y_tick_pairs") or []
    ]
    return AxisCalibrationResult(
        x_min=float(payload["x_min"]),
        x_max=float(payload["x_max"]),
        plot_left=int(payload["plot_left"]),
        plot_right=int(payload["plot_right"]),
        plot_top=int(payload["plot_top"]),
        plot_bottom=int(payload["plot_bottom"]),
        method=str(payload["method"]),
        confidence=float(payload["confidence"]),
        tick_pairs=tick_pairs,
        y_min=None if payload.get("y_min") is None else float(payload["y_min"]),
        y_max=None if payload.get("y_max") is None else float(payload["y_max"]),
        y_tick_pairs=y_tick_pairs,
        y_method=str(payload.get("y_method") or "relative"),
        warnings=list(payload.get("warnings") or []),
    )


def _arithmetic_tick_warnings(tick_pairs: list[tuple[int, float]]) -> list[str]:
    """Flag weak X tick sequences that should not be trusted blindly."""
    warnings: list[str] = []
    if len(tick_pairs) < MIN_OCR_TICK_PAIRS:
        return warnings
    values = [val for _, val in tick_pairs]
    sequence = _select_arithmetic_number_sequence(values)
    if len(sequence) < MIN_OCR_TICK_PAIRS:
        warnings.append("x_ticks_not_arithmetic")
    elif len(sequence) < len(values):
        warnings.append("x_ticks_partial_arithmetic")
    return warnings


def x_calibration_is_usable(
    calibration: AxisCalibrationResult,
    *,
    min_ticks: int = MIN_OCR_TICK_PAIRS,
    min_confidence: float = MIN_OCR_CONFIDENCE,
) -> tuple[bool, list[str]]:
    """Return whether saved X calibration is safe to drive PlotDigitizer remapping."""
    reasons: list[str] = []
    if not str(calibration.method).startswith("ocr"):
        reasons.append(f"method_not_ocr:{calibration.method}")
    if len(calibration.tick_pairs) < min_ticks:
        reasons.append(f"too_few_tick_pairs:{len(calibration.tick_pairs)}")
    if calibration.confidence < min_confidence:
        reasons.append(f"low_confidence:{calibration.confidence:.3f}")
    if calibration.x_max <= calibration.x_min:
        reasons.append("invalid_x_range")
    return (not reasons), reasons


def sidecar_to_dict(sidecar: AxisSidecar) -> dict[str, Any]:
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "source_image": sidecar.source_image,
        "image_width": sidecar.image_width,
        "image_height": sidecar.image_height,
        "plot_crop_bbox": list(sidecar.plot_crop_bbox),
        "calibration": axis_calibration_to_dict(sidecar.calibration),
        "warnings": list(sidecar.warnings),
    }


def sidecar_from_dict(payload: dict[str, Any]) -> AxisSidecar:
    bbox_raw = payload.get("plot_crop_bbox")
    if not bbox_raw or len(bbox_raw) != 4:
        raise AxisSidecarError("sidecar missing plot_crop_bbox")
    plot_crop_bbox = (
        int(bbox_raw[0]),
        int(bbox_raw[1]),
        int(bbox_raw[2]),
        int(bbox_raw[3]),
    )
    calibration = axis_calibration_from_dict(payload["calibration"])
    return AxisSidecar(
        calibration=calibration,
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        plot_crop_bbox=plot_crop_bbox,
        source_image=str(payload.get("source_image") or ""),
        warnings=list(payload.get("warnings") or []),
    )


def save_axis_sidecar(path: Path, sidecar: AxisSidecar) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar_to_dict(sidecar), indent=2), encoding="utf-8")
    return path


def load_axis_sidecar(path: Path) -> AxisSidecar:
    path = Path(path)
    if not path.is_file():
        raise AxisSidecarError(f"axis sidecar not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AxisSidecarError(f"invalid axis sidecar JSON: {path}") from exc
    return sidecar_from_dict(payload)


def resolve_axis_sidecar_path(image_path: Path) -> Path | None:
    """Find a sibling .axes.json for an original or *_clean.png path."""
    image_path = Path(image_path)
    stem = image_path.stem
    candidates: list[Path] = []
    if stem.endswith("_clean"):
        base = stem[: -len("_clean")]
        candidates.append(image_path.parent / f"{base}.axes.json")
    candidates.extend(
        [
            image_path.with_name(f"{stem}.axes.json"),
            image_path.with_suffix(".axes.json"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            return candidate
    return None


def crop_from_saved_bbox(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> PlotCropResult:
    """Slice a plot crop using a previously saved bbox (no re-detection)."""
    height, width = image_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
        raise AxisSidecarError(
            f"saved plot_crop_bbox {bbox} incompatible with image shape {(height, width)}"
        )
    cropped = image_bgr[y0:y1, x0:x1].copy()
    return PlotCropResult(
        cropped_bgr=cropped,
        bbox=(x0, y0, x1, y1),
        confidence=1.0,
        warnings=["crop_from_saved_bbox"],
    )


def assert_sidecar_matches_image(
    sidecar: AxisSidecar,
    image_bgr: np.ndarray,
) -> None:
    height, width = image_bgr.shape[:2]
    if width != sidecar.image_width or height != sidecar.image_height:
        raise AxisSidecarError(
            "cleaned image shape does not match sidecar geometry: "
            f"image={(width, height)} sidecar=({sidecar.image_width}, {sidecar.image_height})"
        )


def build_sidecar_from_image(
    image_bgr: np.ndarray,
    *,
    source_image: str,
) -> AxisSidecar:
    """Calibrate axes on an original (text-bearing) figure and package a sidecar."""
    plot_crop = crop_plot_area(image_bgr)
    calibration = calibrate_axes(plot_crop, full_image_bgr=image_bgr)
    warnings = list(calibration.warnings)
    warnings.extend(_arithmetic_tick_warnings(calibration.tick_pairs))
    usable, reasons = x_calibration_is_usable(calibration)
    if not usable:
        warnings.extend(reasons)
        warnings.append("x_calibration_unusable")
    height, width = image_bgr.shape[:2]
    return AxisSidecar(
        calibration=calibration,
        image_width=int(width),
        image_height=int(height),
        plot_crop_bbox=(
            int(plot_crop.bbox[0]),
            int(plot_crop.bbox[1]),
            int(plot_crop.bbox[2]),
            int(plot_crop.bbox[3]),
        ),
        source_image=source_image,
        warnings=warnings,
    )


def extract_axis_sidecar_for_path(
    image_path: Path,
    *,
    output_path: Path | None = None,
) -> AxisSidecar:
    """Load an original PNG, calibrate, and write ``*.axes.json`` beside it."""
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise AxisSidecarError(f"Could not load image: {image_path}")
    sidecar = build_sidecar_from_image(image_bgr, source_image=image_path.name)
    out = output_path or image_path.with_name(f"{image_path.stem}.axes.json")
    save_axis_sidecar(out, sidecar)
    LOGGER.info(
        "Wrote axis sidecar %s (method=%s x=[%.3f,%.3f] ticks=%d)",
        out.name,
        sidecar.calibration.method,
        sidecar.calibration.x_min,
        sidecar.calibration.x_max,
        len(sidecar.calibration.tick_pairs),
    )
    return sidecar


def load_sidecar_for_digitize(
    image_path: Path,
    image_bgr: np.ndarray,
    *,
    axes_sidecar_path: Path | None = None,
    require_usable_x: bool = True,
) -> tuple[AxisSidecar, PlotCropResult] | None:
    """
    Load a usable axis sidecar for digitizing ``image_path``.

    Returns ``(sidecar, plot_crop)`` with crop sliced from the saved bbox, or
    ``None`` if no sidecar is available.
    """
    path = (
        Path(axes_sidecar_path)
        if axes_sidecar_path is not None
        else resolve_axis_sidecar_path(image_path)
    )
    if path is None:
        return None
    sidecar = load_axis_sidecar(path)
    assert_sidecar_matches_image(sidecar, image_bgr)
    if require_usable_x:
        usable, reasons = x_calibration_is_usable(sidecar.calibration)
        if not usable:
            raise AxisSidecarError(
                f"axis sidecar {path} has unusable X calibration: {', '.join(reasons)}"
            )
    plot_crop = crop_from_saved_bbox(image_bgr, sidecar.plot_crop_bbox)
    return sidecar, plot_crop
