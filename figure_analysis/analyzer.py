"""Single-image compact XRD figure analysis via OpenAI vision."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from figure_analysis.prompts import SYSTEM_PROMPT, build_user_prompt
from figure_analysis.schemas import (
    FigureAnalysisPayload,
    FigureAnalysisResult,
    LatticeParameters,
    XrdCurve,
)

logger = logging.getLogger(__name__)

_PEAK_DEDUPE_TOL = 0.4


def _ensure_autodigitizer_path() -> None:
    root = Path(__file__).resolve().parents[1]
    auto = root / "autonomous_digitizer"
    for path in (root, auto):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _get_client(config=None):
    _ensure_autodigitizer_path()
    from autodigitizer.config import Config
    from autodigitizer.vision.openai_client import OpenAIVisionClient

    cfg = config or Config.from_env()
    return OpenAIVisionClient(cfg), cfg


def _round_2theta(value: float) -> float:
    return round(float(value), 1)


def _dedupe_sorted_positions(
    values: list[float],
    *,
    x_min: Optional[float],
    x_max: Optional[float],
    tol: float = _PEAK_DEDUPE_TOL,
) -> list[float]:
    cleaned: list[float] = []
    for raw in values:
        try:
            pos = _round_2theta(raw)
        except (TypeError, ValueError):
            continue
        if x_min is not None and pos < float(x_min) - 1e-6:
            continue
        if x_max is not None and pos > float(x_max) + 1e-6:
            continue
        cleaned.append(pos)
    cleaned.sort()
    out: list[float] = []
    for pos in cleaned:
        if out and abs(pos - out[-1]) < tol:
            continue
        out.append(pos)
    return out


def _clean_lattice(item: LatticeParameters) -> LatticeParameters | None:
    if item.phase is not None and not str(item.phase).strip():
        item.phase = None
    if item.space_group is not None and not str(item.space_group).strip():
        item.space_group = None
    useful = any(
        v is not None
        for v in (
            item.phase,
            item.a_A,
            item.b_A,
            item.c_A,
            item.alpha_deg,
            item.beta_deg,
            item.gamma_deg,
            item.volume_A3,
            item.space_group,
        )
    )
    return item if useful else None


def _clean_curve(
    curve: XrdCurve,
    *,
    x_min: Optional[float],
    x_max: Optional[float],
) -> XrdCurve:
    if curve.condition is not None and not str(curve.condition).strip():
        curve.condition = None
    if curve.peak_width is not None and not str(curve.peak_width).strip():
        curve.peak_width = None
    if curve.peak_width is not None and curve.peak_width not in {
        "broad",
        "moderate",
        "narrow",
    }:
        if len(str(curve.peak_width)) > 40:
            curve.peak_width = None
    if curve.profile_function is not None and not str(curve.profile_function).strip():
        curve.profile_function = None
    if curve.fwhm is not None:
        try:
            curve.fwhm = float(curve.fwhm)
            if curve.fwhm <= 0:
                curve.fwhm = None
        except (TypeError, ValueError):
            curve.fwhm = None
    curve.phases = [p for p in curve.phases if str(p).strip()]
    curve.peak_positions = _dedupe_sorted_positions(
        list(curve.peak_positions),
        x_min=x_min,
        x_max=x_max,
    )
    cleaned_lp: list[LatticeParameters] = []
    for item in curve.lattice_parameters:
        cleaned = _clean_lattice(item)
        if cleaned is not None:
            cleaned_lp.append(cleaned)
    curve.lattice_parameters = cleaned_lp
    return curve


def post_validate(result: FigureAnalysisResult) -> FigureAnalysisResult:
    """Clear non-XRD payloads; sort/dedupe peaks for XRD figures."""
    if not result.is_xrd:
        result.sample = None
        result.curves = []
        result.trends = []
        result.fwhm = None
        result.lattice_parameters = []
        result.profile_function = None
        result.x_min = None
        result.x_max = None
        return result

    if result.sample is not None and not str(result.sample).strip():
        result.sample = None
    if result.profile_function is not None and not str(result.profile_function).strip():
        result.profile_function = None
    if result.fwhm is not None:
        try:
            result.fwhm = float(result.fwhm)
            if result.fwhm <= 0:
                result.fwhm = None
        except (TypeError, ValueError):
            result.fwhm = None
    cleaned_lp: list[LatticeParameters] = []
    for item in result.lattice_parameters:
        cleaned = _clean_lattice(item)
        if cleaned is not None:
            cleaned_lp.append(cleaned)
    result.lattice_parameters = cleaned_lp
    result.trends = [t for t in result.trends if str(t).strip()]
    result.curves = [
        _clean_curve(c, x_min=result.x_min, x_max=result.x_max)
        for c in result.curves
    ]
    return result


_LATTICE_KEYS = (
    "phase",
    "a_A",
    "b_A",
    "c_A",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
    "volume_A3",
    "space_group",
)


def _normalize_lattice_dict(item: dict[str, Any] | None = None) -> dict[str, Any]:
    """Always emit every lattice-parameter field (null when unknown)."""
    src = item or {}
    return {key: src.get(key) for key in _LATTICE_KEYS}


def _normalize_curve_dict(curve: dict[str, Any] | None = None) -> dict[str, Any]:
    """Always emit every curve field; lists default to []."""
    src = curve or {}
    return {
        "condition": src.get("condition"),
        "peak_positions": list(src.get("peak_positions") or []),
        "phases": list(src.get("phases") or []),
        "peak_width": src.get("peak_width"),
        "fwhm": src.get("fwhm"),
        "lattice_parameters": [
            _normalize_lattice_dict(item)
            for item in (src.get("lattice_parameters") or [])
            if isinstance(item, dict)
        ],
        "profile_function": src.get("profile_function"),
    }


def analysis_to_dataset_dict(result: FigureAnalysisResult) -> dict[str, Any] | None:
    """
    Dump analysis JSON with every schema field present.

    Empty scalars are ``null``; empty collections are ``[]``. Temporary axis
    bounds (``x_min`` / ``x_max``) are omitted. Returns None for non-XRD.
    """
    if not result.is_xrd:
        return None

    raw = result.model_dump(mode="json", exclude_none=False)
    raw.pop("is_xrd", None)
    raw.pop("x_min", None)
    raw.pop("x_max", None)

    return {
        "sample": raw.get("sample"),
        "curves": [
            _normalize_curve_dict(curve)
            for curve in (raw.get("curves") or [])
            if isinstance(curve, dict)
        ],
        "trends": list(raw.get("trends") or []),
        "fwhm": raw.get("fwhm"),
        "lattice_parameters": [
            _normalize_lattice_dict(item)
            for item in (raw.get("lattice_parameters") or [])
            if isinstance(item, dict)
        ],
        "profile_function": raw.get("profile_function"),
    }


def analyze_figure_image(
    image_path: Path | str,
    *,
    client=None,
    config=None,
    cache_key: Optional[str] = None,
    source_image_name: Optional[str] = None,
    caption: Optional[str] = None,
    text: Optional[str] = None,
    figure_number: Optional[int] = None,
) -> FigureAnalysisResult:
    """Run one multimodal XRD gate + metadata extraction on a cropped figure."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if client is None:
        client, cfg = _get_client(config)
    else:
        cfg = config or getattr(client, "config", None)

    key = cache_key or f"figure_analysis/{path.stem}"
    prompt = build_user_prompt(
        caption=caption,
        text=text,
        figure_number=figure_number,
    )

    t0 = time.time()
    logger.info("Analyzing figure (XRD gate): %s", path)
    payload = client.analyze_structured(
        prompt=prompt,
        schema_model=FigureAnalysisPayload,
        images=[path],
        cache_key=key,
        cache_name="xrd_analysis_v7.json",
        system=SYSTEM_PROMPT,
    )
    elapsed = time.time() - t0
    logger.debug(
        "Figure analysis finished in %.2fs (model=%s is_xrd=%s)",
        elapsed,
        getattr(cfg, "model", None),
        getattr(payload, "is_xrd", None),
    )

    if not isinstance(payload, FigureAnalysisPayload):
        raise TypeError(f"Expected FigureAnalysisPayload, got {type(payload)}")

    result = FigureAnalysisResult.model_validate(payload.model_dump())
    return post_validate(result)


def empty_result(*, is_xrd: bool = False) -> FigureAnalysisResult:
    """Minimal result used by offline / mock paths."""
    if not is_xrd:
        return FigureAnalysisResult(is_xrd=False)
    return FigureAnalysisResult(is_xrd=True, curves=[XrdCurve()])
