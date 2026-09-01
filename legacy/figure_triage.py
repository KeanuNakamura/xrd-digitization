"""OpenAI vision triage: single-curve digitizability and ClipDrop need.

Standalone helper for the scrape-and-digitize pipeline. Does not use the
native XRD digitizer or agent-guidance stack.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"

HttpPost = Callable[..., Any]

TRIAGE_SYSTEM_PROMPT = """\
You analyze scientific figure images (often XRD / diffraction plots).
Return structured JSON only — no markdown fences.
Decide whether the figure is digitizable as a single curve and whether
in-plot text must be removed before digitizing.
"""

TRIAGE_USER_PROMPT = """\
Inspect this figure image and return JSON with this schema:
{
  "digitizable": <bool>,
  "curve_count": <int>,
  "curve_layout": "single" | "overlay" | "stacked" | "other",
  "needs_clipdrop": <bool>,
  "reason": "<short explanation>"
}

Rules:
- digitizable is true ONLY if the plot contains exactly one data curve
  (one continuous spectrum / line). If there are multiple overlaid curves,
  stacked panels with multiple curves, legends with several series, or the
  image is not a 2D line/spectrum plot, set digitizable to false.
- curve_count: number of distinct data curves visible.
- curve_layout:
  - "single" — one curve in one axes frame
  - "overlay" — multiple curves sharing the same axes
  - "stacked" — multiple curves in vertically stacked bands/panels
  - "other" — not a single-curve plot (photos, tables, multi-panel non-XRD, etc.)
- needs_clipdrop: true if in-plot text annotations (Miller indices, peak labels,
  inset labels, etc.) overlap or sit on the curve / plot interior in a way that
  would interfere with automatic curve tracing. Axis tick labels and axis titles
  outside the plot interior do NOT require ClipDrop. false if the plot interior
  is clean enough to digitize without text removal.
- reason: one short sentence.
"""


@dataclass
class FigureTriageResult:
    digitizable: bool
    curve_count: int
    curve_layout: str
    needs_clipdrop: bool
    reason: str
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.raw is None:
            payload.pop("raw", None)
        return payload


def is_digitizable(result: FigureTriageResult) -> bool:
    """True only for a single-curve plot suitable for PlotDigitizer."""
    return (
        bool(result.digitizable)
        and int(result.curve_count) == 1
        and str(result.curve_layout).strip().lower() == "single"
    )


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


def parse_triage_payload(raw: dict[str, Any]) -> FigureTriageResult:
    """Normalize a triage JSON object into ``FigureTriageResult``."""
    layout = str(raw.get("curve_layout") or "other").strip().lower() or "other"
    if layout not in {"single", "overlay", "stacked", "other"}:
        layout = "other"

    try:
        curve_count = int(raw.get("curve_count") if raw.get("curve_count") is not None else 0)
    except (TypeError, ValueError):
        curve_count = 0
    curve_count = max(0, curve_count)

    digitizable = bool(raw.get("digitizable"))
    # Enforce single-curve rule even if the model is inconsistent.
    if curve_count != 1 or layout != "single":
        digitizable = False

    return FigureTriageResult(
        digitizable=digitizable,
        curve_count=curve_count,
        curve_layout=layout,
        needs_clipdrop=bool(raw.get("needs_clipdrop")),
        reason=str(raw.get("reason") or "").strip(),
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
