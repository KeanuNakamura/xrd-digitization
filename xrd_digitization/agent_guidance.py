"""Vision-agent guidance metadata for hybrid XRD digitization.

The agent supplies semantic priors (plot box, text regions, approximate peaks).
Pixel-level extraction remains the responsibility of the image digitizer.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests

from xrd_digitization.coords import transform_bbox

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_COORDINATE_SPACE = "original_image_pixels"

AGENT_SYSTEM_PROMPT = """\
You analyze XRD (X-ray diffraction) figure images and return structured JSON only.
All pixel coordinates use the original image pixel space (origin top-left).
Return image_width and image_height of the image you inspected.
Bounding boxes must be TIGHT around the complete visible text glyphs — not
approximate boxes floating above the annotations, and not loose padded regions.
For vertical Miller-index labels, the box must cover the full rotated string.
approximate_peaks must be actual peak-center 2θ values (apex of the curve),
not annotation positions and not rough integer guesses.
Do not invent precise curve intensities; optional approximate_curve is a sparse prior only.
Return JSON matching the schema exactly — no markdown fences.
"""

AGENT_USER_PROMPT = """\
Extract XRD figure guidance metadata as JSON with this schema:
{
  "image_width": <int>,
  "image_height": <int>,
  "coordinate_space": "original_image_pixels",
  "plot_bbox": [x1, y1, x2, y2],
  "x_axis": {"min": <float>, "max": <float>},
  "y_axis": {"min": <float>, "max": <float>},
  "text_regions": [
    {
      "bbox": [x1, y1, x2, y2],
      "text": "(104)",
      "orientation": "vertical"|"horizontal"|"other",
      "confidence": 0.0-1.0,
      "type": "peak_annotation"|"label"|"legend"|"other"
    }
  ],
  "curve_count": <int>,
  "curve_layout": "overlay"|"stacked"|"single",
  "approximate_peaks": [<float>, ...],
  "approximate_curve": null | {"two_theta": [...], "intensity": [...]}
}

Rules:
- image_width / image_height: exact pixel size of this image.
- coordinate_space: always "original_image_pixels".
- plot_bbox: tightly around the data axes (include tick labels if visible).
- text_regions.bbox: TIGHT boxes around the complete visible annotation text.
  Boxes must overlap the ink of the characters. Do not place boxes above the text.
- text_regions.text: transcribed label when readable (e.g. "(104)").
- text_regions.orientation: "vertical" for rotated Miller indices above peaks.
- approximate_peaks: 2θ of major peak centers (curve apex), typically within ~0.3°.
- Pixel coordinates relative to the full provided image.
"""


@dataclass
class AgentTextRegion:
    bbox: list[float]
    type: str = "other"
    text: str = ""
    orientation: str = "other"
    confidence: float = 0.0


@dataclass
class AgentAxisRange:
    min: float
    max: float


@dataclass
class AgentApproximateCurve:
    two_theta: list[float] = field(default_factory=list)
    intensity: list[float] = field(default_factory=list)


@dataclass
class AgentFigureMetadata:
    plot_bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    x_axis: AgentAxisRange = field(default_factory=lambda: AgentAxisRange(0.0, 90.0))
    y_axis: AgentAxisRange = field(default_factory=lambda: AgentAxisRange(0.0, 1.0))
    text_regions: list[AgentTextRegion] = field(default_factory=list)
    curve_count: int = 1
    curve_layout: str = "single"
    approximate_peaks: list[float] = field(default_factory=list)
    approximate_curve: AgentApproximateCurve | None = None
    image_width: int | None = None
    image_height: int | None = None
    coordinate_space: str = DEFAULT_COORDINATE_SPACE
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_space": self.coordinate_space,
            "plot_bbox": list(self.plot_bbox),
            "x_axis": {"min": self.x_axis.min, "max": self.x_axis.max},
            "y_axis": {"min": self.y_axis.min, "max": self.y_axis.max},
            "text_regions": [
                {
                    "bbox": list(r.bbox),
                    "type": r.type,
                    "text": r.text,
                    "orientation": r.orientation,
                    "confidence": float(r.confidence),
                }
                for r in self.text_regions
            ],
            "curve_count": int(self.curve_count),
            "curve_layout": self.curve_layout,
            "approximate_peaks": [float(p) for p in self.approximate_peaks],
            "approximate_curve": None,
        }
        if self.approximate_curve is not None:
            payload["approximate_curve"] = {
                "two_theta": list(self.approximate_curve.two_theta),
                "intensity": list(self.approximate_curve.intensity),
            }
        return payload


def _clamp_bbox(
    box: list[float] | tuple[float, ...] | None,
    width: int,
    height: int,
) -> list[float]:
    if not box or len(box) < 4:
        return [0.0, 0.0, float(max(width - 1, 0)), float(max(height - 1, 0))]
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    x2 = float(np.clip(x2, 0, width))
    y2 = float(np.clip(y2, 0, height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _as_axis(value: Any, default_min: float, default_max: float) -> AgentAxisRange:
    if isinstance(value, dict):
        try:
            lo = float(value.get("min", default_min))
            hi = float(value.get("max", default_max))
        except (TypeError, ValueError):
            return AgentAxisRange(default_min, default_max)
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            hi = lo + 1.0
        return AgentAxisRange(lo, hi)
    return AgentAxisRange(default_min, default_max)


def _as_layout(value: Any) -> str:
    text = str(value or "single").strip().lower()
    if text in {"overlay", "stacked", "single"}:
        return text
    return "single"


def _as_orientation(value: Any) -> str:
    text = str(value or "other").strip().lower()
    if text in {"vertical", "horizontal", "other"}:
        return text
    return "other"


def _parse_approximate_curve(value: Any) -> AgentApproximateCurve | None:
    if not isinstance(value, dict):
        return None
    xs = value.get("two_theta") or value.get("x") or []
    ys = value.get("intensity") or value.get("y") or []
    try:
        two_theta = [float(v) for v in xs]
        intensity = [float(v) for v in ys]
    except (TypeError, ValueError):
        return None
    if not two_theta or len(two_theta) != len(intensity):
        return None
    return AgentApproximateCurve(two_theta=two_theta, intensity=intensity)


def _infer_source_image_size(
    raw: dict[str, Any],
    *,
    image_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    """Return (width, height) for the agent coordinate space."""
    width = raw.get("image_width")
    height = raw.get("image_height")
    try:
        if width is not None and height is not None:
            return max(1, int(width)), max(1, int(height))
    except (TypeError, ValueError):
        pass

    # Infer from max extents in boxes when dims omitted (legacy agent JSON).
    max_x = 0.0
    max_y = 0.0
    for key in ("plot_bbox",):
        box = raw.get(key)
        if box and len(box) >= 4:
            max_x = max(max_x, float(box[2]))
            max_y = max(max_y, float(box[3]))
    for item in raw.get("text_regions") or []:
        if isinstance(item, dict) and item.get("bbox") and len(item["bbox"]) >= 4:
            max_x = max(max_x, float(item["bbox"][2]))
            max_y = max(max_y, float(item["bbox"][3]))

    if image_shape is not None:
        ih, iw = image_shape
        # If extents clearly exceed / undershoot the real image, keep inferred
        # source size so transform_bbox can rescale.
        if max_x > 1 and max_y > 1:
            # When extents fit inside the real image, assume same space.
            if max_x <= iw * 1.05 and max_y <= ih * 1.05:
                return iw, ih
            return max(1, int(round(max_x))), max(1, int(round(max_y)))
        return iw, ih

    if max_x > 1 and max_y > 1:
        return max(1, int(round(max_x))), max(1, int(round(max_y)))
    return 10_000, 10_000


def extract_agent_metadata(
    payload: dict[str, Any] | AgentFigureMetadata | str | Path,
    *,
    image_shape: tuple[int, int] | None = None,
) -> AgentFigureMetadata:
    """Normalize agent JSON / dataclass into ``AgentFigureMetadata``.

    ``image_shape`` is ``(height, width)`` when available for bbox clamping.
    """
    if isinstance(payload, AgentFigureMetadata):
        meta = payload
        raw = meta.raw or meta.to_dict()
    elif isinstance(payload, (str, Path)):
        path = Path(payload)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Agent metadata file must contain a JSON object: {path}")
    elif isinstance(payload, dict):
        raw = payload
    else:
        raise TypeError(f"Unsupported agent metadata type: {type(payload)!r}")

    src_w, src_h = _infer_source_image_size(raw, image_shape=image_shape)
    # Clamp in the agent/source space (not necessarily the actual image).
    clamp_w, clamp_h = src_w, src_h

    text_regions: list[AgentTextRegion] = []
    for item in raw.get("text_regions") or []:
        if not isinstance(item, dict):
            continue
        bbox = _clamp_bbox(item.get("bbox"), clamp_w, clamp_h)
        region_type = str(item.get("type") or "other").strip().lower() or "other"
        try:
            conf = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        text_regions.append(
            AgentTextRegion(
                bbox=bbox,
                type=region_type,
                text=str(item.get("text") or ""),
                orientation=_as_orientation(item.get("orientation")),
                confidence=float(np.clip(conf, 0.0, 1.0)),
            )
        )

    peaks: list[float] = []
    for p in raw.get("approximate_peaks") or []:
        try:
            peaks.append(float(p))
        except (TypeError, ValueError):
            continue

    try:
        curve_count = max(1, int(raw.get("curve_count") or 1))
    except (TypeError, ValueError):
        curve_count = 1

    coord_space = str(raw.get("coordinate_space") or DEFAULT_COORDINATE_SPACE).strip()
    if not coord_space:
        coord_space = DEFAULT_COORDINATE_SPACE

    return AgentFigureMetadata(
        plot_bbox=_clamp_bbox(raw.get("plot_bbox"), clamp_w, clamp_h),
        x_axis=_as_axis(raw.get("x_axis"), 0.0, 90.0),
        y_axis=_as_axis(raw.get("y_axis"), 0.0, 1.0),
        text_regions=text_regions,
        curve_count=curve_count,
        curve_layout=_as_layout(raw.get("curve_layout")),
        approximate_peaks=peaks,
        approximate_curve=_parse_approximate_curve(raw.get("approximate_curve")),
        image_width=src_w,
        image_height=src_h,
        coordinate_space=coord_space,
        raw=dict(raw),
    )


def save_agent_metadata(meta: AgentFigureMetadata, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return path


def load_agent_metadata(
    path: Path,
    *,
    image_shape: tuple[int, int] | None = None,
) -> AgentFigureMetadata:
    return extract_agent_metadata(path, image_shape=image_shape)


def _encode_image_png_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image for agent request")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _parse_agent_response_content(content: str) -> dict[str, Any]:
    text = _strip_json_fence(content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Agent response JSON must be an object")
    return payload


HttpPost = Callable[..., Any]


def call_vision_agent(
    image_bgr: np.ndarray,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 120.0,
    http_post: HttpPost | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible vision chat completion endpoint."""
    key = api_key or os.environ.get("XRD_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing XRD_AGENT_API_KEY or OPENAI_API_KEY for hybrid agent guidance"
        )
    url_base = (base_url or os.environ.get("XRD_AGENT_BASE_URL") or DEFAULT_BASE_URL).rstrip(
        "/"
    )
    model_name = model or os.environ.get("XRD_AGENT_MODEL") or DEFAULT_MODEL
    b64 = _encode_image_png_b64(image_bgr)
    # Embed true pixel size so the model can echo it back.
    h, w = image_bgr.shape[:2]
    user_prompt = (
        AGENT_USER_PROMPT
        + f"\n\nThis image is {w} pixels wide and {h} pixels tall. "
        f'Set image_width={w}, image_height={h}, coordinate_space="original_image_pixels".'
    )
    body = {
        "model": model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
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
        raise ValueError(f"Unexpected agent response shape: {data!r}") from exc
    if not isinstance(content, str):
        raise ValueError("Agent message content must be a string")
    return _parse_agent_response_content(content)


def run_agent_guidance(
    image: np.ndarray | str | Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    metadata_path: Path | None = None,
    http_post: HttpPost | None = None,
) -> AgentFigureMetadata:
    """Run the vision agent (or load ``metadata_path`` offline) and normalize."""
    if metadata_path is not None and Path(metadata_path).exists():
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            shape = None if bgr is None else bgr.shape[:2]
        else:
            shape = image.shape[:2]
        return load_agent_metadata(metadata_path, image_shape=shape)

    if isinstance(image, (str, Path)):
        image_bgr = cv2.imread(str(image))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image for agent guidance: {image}")
    else:
        image_bgr = image

    raw = call_vision_agent(
        image_bgr,
        api_key=api_key,
        base_url=base_url,
        model=model,
        http_post=http_post,
    )
    # Ensure dims are present even if the model omitted them.
    raw.setdefault("image_width", int(image_bgr.shape[1]))
    raw.setdefault("image_height", int(image_bgr.shape[0]))
    raw.setdefault("coordinate_space", DEFAULT_COORDINATE_SPACE)
    meta = extract_agent_metadata(raw, image_shape=image_bgr.shape[:2])
    LOGGER.info(
        "Agent guidance: %d text region(s), %d peak(s), layout=%s, size=%sx%s",
        len(meta.text_regions),
        len(meta.approximate_peaks),
        meta.curve_layout,
        meta.image_width,
        meta.image_height,
    )
    return meta


def offset_text_regions_to_crop(
    meta: AgentFigureMetadata,
    crop_bbox: tuple[int, int, int, int],
    *,
    target_image_size: tuple[int, int] | None = None,
    actual_image_size: tuple[int, int] | None = None,
) -> list[AgentTextRegion]:
    """
    Convert agent text boxes into plot-crop coordinates via ``transform_bbox``.

    ``target_image_size`` is ``(width, height)`` of the cropped array.
    ``actual_image_size`` is ``(width, height)`` of the real original image.
    """
    src_w = int(meta.image_width or 0)
    src_h = int(meta.image_height or 0)
    if src_w <= 0 or src_h <= 0:
        # Legacy fallback: plain origin subtract.
        ox, oy, _, _ = crop_bbox
        out: list[AgentTextRegion] = []
        for region in meta.text_regions:
            x1, y1, x2, y2 = region.bbox
            out.append(
                AgentTextRegion(
                    bbox=[x1 - ox, y1 - oy, x2 - ox, y2 - oy],
                    type=region.type,
                    text=region.text,
                    orientation=region.orientation,
                    confidence=region.confidence,
                )
            )
        return out

    crop_x0, crop_y0, crop_x1, crop_y1 = crop_bbox
    if target_image_size is None:
        target_image_size = (max(1, crop_x1 - crop_x0), max(1, crop_y1 - crop_y0))

    out = []
    for region in meta.text_regions:
        local = transform_bbox(
            region.bbox,
            (src_w, src_h),
            crop_bbox,
            target_image_size,
            actual_image_size=actual_image_size,
        )
        out.append(
            AgentTextRegion(
                bbox=local,
                type=region.type,
                text=region.text,
                orientation=region.orientation,
                confidence=region.confidence,
            )
        )
    return out
