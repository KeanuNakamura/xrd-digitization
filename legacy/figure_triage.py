"""OpenAI vision triage: figure digitizability, layout, and ClipDrop need.

Standalone helper for the scrape-and-digitize pipeline. Does not use the
native XRD digitizer or agent-guidance stack.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"

HttpPost = Callable[..., Any]

DigitizationRoute = Literal["single", "stacked", "multipanel", "unsupported", "reject"]

PREPROCESSING_REASONS = frozenset(
    {
        "multiple_panels",
        "multiple_curves",
        "vertically_offset_curves",
        "overlapping_annotations",
    }
)

# Canonical layouts used for branching.
CANONICAL_LAYOUTS = frozenset(
    {
        "single",
        "multiple_stacked",
        "multiple_overlapping",
        "multiple_subplots",
        "not_applicable",
    }
)

# Accept legacy / short aliases from older prompts and model responses.
_LAYOUT_ALIASES: dict[str, str] = {
    "single": "single",
    "stacked": "multiple_stacked",
    "multiple_stacked": "multiple_stacked",
    "overlay": "multiple_overlapping",
    "overlapping": "multiple_overlapping",
    "multiple_overlapping": "multiple_overlapping",
    "subplots": "multiple_subplots",
    "multiple_subplots": "multiple_subplots",
    "other": "not_applicable",
    "not_applicable": "not_applicable",
    "n/a": "not_applicable",
    "na": "not_applicable",
}

TRIAGE_SYSTEM_PROMPT = """\
You analyze scientific figure images (often XRD / diffraction plots).
Return structured JSON only — no markdown fences.
Classify curve layout, estimate curve count, decide whether the figure is
digitizable (single curve or vertically stacked shared-x curves), extract
curve labels when readable, and say whether in-plot text must be removed
before digitizing.
"""

TRIAGE_USER_PROMPT = """\
Inspect this figure image and return JSON with this schema:
{
  "digitizable": <bool>,
  "curve_count": <int>,
  "curve_layout": "single" | "multiple_stacked" | "multiple_overlapping" | "multiple_subplots" | "not_applicable",
  "shared_x_axis": <bool>,
  "vertically_offset": <bool>,
  "curves_cross": <bool>,
  "needs_clipdrop": <bool>,
  "curve_labels": [{"text": "<label>", "vertical_order": <int>}, ...],
  "reason": "<short explanation>"
}

Rules:
- digitizable is true if:
  (a) exactly one data curve in one axes frame (curve_layout="single"), OR
  (b) multiple vertically stacked traces that share one x-axis
      (curve_layout="multiple_stacked", shared_x_axis=true, curves_cross=false).
  Set digitizable false for overlapping/crossing overlays, separate subplot
  frames that need independent axes, photos, tables, or non-line plots.
- curve_count: number of distinct data curves / traces visible (estimated).
- curve_layout (prefer these canonical values; aliases "stacked"/"overlay"/"other"
  are also accepted by the parser):
  - "single" — one curve in one axes frame
  - "multiple_stacked" — multiple curves in vertically offset bands, same axes
  - "multiple_overlapping" — multiple curves sharing axes and overlapping in y
  - "multiple_subplots" — separate subplot panels with their own frames/axes
  - "not_applicable" — not a digitizable line/spectrum plot
- shared_x_axis: true when all curves share one horizontal axis scale.
- vertically_offset: true when stacked traces are artificially shifted in y.
- curves_cross: true if traces cross or heavily interleave in the same band.
- needs_clipdrop: true if in-plot text annotations (Miller indices, peak labels,
  inset labels, etc.) overlap or sit on the curve / plot interior in a way that
  would interfere with automatic curve tracing. Axis tick labels and axis titles
  outside the plot interior do NOT require ClipDrop. false if the plot interior
  is clean enough to digitize without text removal.
- curve_labels: readable per-curve labels ordered top→bottom (vertical_order 0
  at top). Examples: "715 nm", "(a)", "as-prepared". Empty list if none.
- reason: one short sentence.
"""


@dataclass
class CurveLabelHint:
    text: str
    vertical_order: int


@dataclass
class FigureTriageResult:
    digitizable: bool
    curve_count: int
    curve_layout: str
    needs_clipdrop: bool
    reason: str
    shared_x_axis: bool = True
    vertically_offset: bool = False
    curves_cross: bool = False
    curve_labels: list[CurveLabelHint] = field(default_factory=list)
    preprocessing_reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.raw is None:
            payload.pop("raw", None)
        return payload


def normalize_curve_layout(raw_layout: str | None) -> str:
    """Map model / legacy layout strings to a canonical layout value."""
    key = str(raw_layout or "not_applicable").strip().lower() or "not_applicable"
    return _LAYOUT_ALIASES.get(key, "not_applicable")


def is_digitizable(result: FigureTriageResult) -> bool:
    """True only for a single-curve plot suitable for the existing single path."""
    layout = normalize_curve_layout(result.curve_layout)
    return (
        bool(result.digitizable)
        and layout == "single"
        and int(result.curve_count) == 1
    )


def is_stacked_digitizable(result: FigureTriageResult) -> bool:
    """True for vertically stacked shared-x figures suitable for stacked path."""
    layout = normalize_curve_layout(result.curve_layout)
    if layout != "multiple_stacked":
        return False
    if result.curves_cross:
        return False
    if not result.shared_x_axis:
        return False
    if int(result.curve_count) < 2:
        return False
    return True


def is_multipanel_digitizable(result: FigureTriageResult) -> bool:
    """True when multi-subplot figures should be split then digitized per panel."""
    layout = normalize_curve_layout(result.curve_layout)
    if layout != "multiple_subplots":
        return False
    # Split first; per-panel triage rejects stick/non-spectrum panels.
    return int(result.curve_count) >= 1


def digitization_route(result: FigureTriageResult) -> DigitizationRoute:
    """
    High-level routing decision for the scrape/digitize pipeline.

    Returns:
      - ``single`` — existing single-curve PlotDigitizer path
      - ``stacked`` — stacked multi-curve path
      - ``multipanel`` — split subplot panels, then digitize each
      - ``unsupported`` — multi-curve but not handled yet (overlay)
      - ``reject`` — not digitizable / not applicable
    """
    layout = normalize_curve_layout(result.curve_layout)

    if layout == "single" and is_digitizable(result):
        return "single"

    if layout == "multiple_stacked" and is_stacked_digitizable(result):
        return "stacked"

    if layout == "multiple_subplots" and is_multipanel_digitizable(result):
        return "multipanel"

    if layout == "multiple_subplots":
        return "reject"

    if layout == "multiple_overlapping":
        return "unsupported"

    if layout == "multiple_stacked":
        # Structural mismatch (crossing / no shared x / too few curves).
        if result.curves_cross or not result.shared_x_axis or int(result.curve_count) < 2:
            return "unsupported"
        # Model rejected a geometrically stacked figure (e.g. stick patterns).
        return "reject"

    return "reject"


def _encode_image_png_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image for triage request")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Triage response must be a JSON object")
    return data


def _parse_curve_labels(raw_labels: Any) -> list[CurveLabelHint]:
    if not isinstance(raw_labels, list):
        return []
    labels: list[CurveLabelHint] = []
    for index, item in enumerate(raw_labels):
        if isinstance(item, str):
            text = item.strip()
            if text:
                labels.append(CurveLabelHint(text=text, vertical_order=index))
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            order = int(item.get("vertical_order") if item.get("vertical_order") is not None else index)
        except (TypeError, ValueError):
            order = index
        labels.append(CurveLabelHint(text=text, vertical_order=order))
    labels.sort(key=lambda item: item.vertical_order)
    return labels


def parse_triage_payload(raw: dict[str, Any]) -> FigureTriageResult:
    """Normalize a triage JSON object into ``FigureTriageResult``."""
    layout = normalize_curve_layout(str(raw.get("curve_layout") or ""))

    try:
        curve_count = int(raw.get("curve_count") if raw.get("curve_count") is not None else 0)
    except (TypeError, ValueError):
        curve_count = 0
    curve_count = max(0, curve_count)

    shared_x_axis = bool(raw["shared_x_axis"]) if "shared_x_axis" in raw else True
    vertically_offset = bool(raw.get("vertically_offset", layout == "multiple_stacked"))
    curves_cross = bool(raw.get("curves_cross", False))
    curve_labels = _parse_curve_labels(raw.get("curve_labels"))

    digitizable = bool(raw.get("digitizable"))
    # Enforce consistency with layout / count even if the model is inconsistent.
    if layout == "single":
        if curve_count != 1:
            digitizable = False
    elif layout == "multiple_stacked":
        if curve_count < 2 or not shared_x_axis or curves_cross:
            digitizable = False
        elif digitizable is False and curve_count >= 2 and shared_x_axis and not curves_cross:
            # Prefer routing stacked figures when flags look good.
            digitizable = True
    elif layout == "multiple_subplots":
        # Keep the model's digitizable flag; routing still splits panels.
        pass
    else:
        digitizable = False

    preprocessing_reasons: list[str] = []
    raw_reasons = raw.get("preprocessing_reasons")
    if isinstance(raw_reasons, list):
        for item in raw_reasons:
            key = str(item or "").strip().lower()
            if key in PREPROCESSING_REASONS and key not in preprocessing_reasons:
                preprocessing_reasons.append(key)
    if layout == "multiple_subplots" and "multiple_panels" not in preprocessing_reasons:
        preprocessing_reasons.append("multiple_panels")
    if layout == "multiple_stacked" and "vertically_offset_curves" not in preprocessing_reasons:
        preprocessing_reasons.append("vertically_offset_curves")
    if bool(raw.get("needs_clipdrop")) and "overlapping_annotations" not in preprocessing_reasons:
        preprocessing_reasons.append("overlapping_annotations")

    # Store canonical layout; keep short aliases only in raw.
    return FigureTriageResult(
        digitizable=digitizable,
        curve_count=curve_count,
        curve_layout=layout,
        needs_clipdrop=bool(raw.get("needs_clipdrop")),
        reason=str(raw.get("reason") or "").strip(),
        shared_x_axis=shared_x_axis,
        vertically_offset=vertically_offset,
        curves_cross=curves_cross,
        curve_labels=curve_labels,
        preprocessing_reasons=preprocessing_reasons,
        raw=dict(raw),
    )


def _normalize_api_key(raw: str | None) -> str | None:
    """Strip whitespace and wrapping quotes (incl. curly) from an API key.

    Copy-pasted keys often arrive as ``’sk-…’``; HTTP headers must be latin-1,
    so those characters blow up before the request leaves the client.
    """
    if raw is None:
        return None
    key = raw.strip()
    if not key:
        return None
    quotes = "'\"‘’“”`"
    while len(key) >= 2 and key[0] in quotes and key[-1] in quotes:
        key = key[1:-1].strip()
    # Also drop a lone leading/trailing smart quote from partial paste.
    key = key.strip(quotes + " \t\r\n")
    if not key:
        return None
    try:
        key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "OPENAI_API_KEY contains non-ASCII characters after quote stripping; "
            "re-export it without smart quotes (e.g. export OPENAI_API_KEY=sk-...)"
        ) from exc
    return key


def call_openai_triage(
    image_bgr: np.ndarray,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 120.0,
    http_post: HttpPost | None = None,
) -> FigureTriageResult:
    """Call an OpenAI-compatible vision endpoint for figure triage."""
    key = _normalize_api_key(api_key or os.environ.get("OPENAI_API_KEY"))
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY for figure triage")

    url_base = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
        "/"
    )
    model_name = model or os.environ.get("OPENAI_TRIAGE_MODEL") or DEFAULT_MODEL
    b64 = _encode_image_png_b64(image_bgr)

    body = {
        "model": model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRIAGE_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    post = http_post or requests.post
    response = post(
        f"{url_base}/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected triage response shape: {data!r}") from exc
    if not isinstance(content, str):
        raise ValueError("Triage message content must be a string")
    return parse_triage_payload(_parse_json_content(content))


def triage_figure_image(
    image: np.ndarray | str | Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    http_post: HttpPost | None = None,
) -> FigureTriageResult:
    """Triage a figure image path or BGR array."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
    else:
        bgr = np.asarray(image)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            raise ValueError("Expected a BGR image array")

    return call_openai_triage(
        bgr,
        api_key=api_key,
        base_url=base_url,
        model=model,
        http_post=http_post,
    )


def save_triage_result(result: FigureTriageResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
