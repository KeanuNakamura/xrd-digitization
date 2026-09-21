"""Split multi-panel figures into independent panel crops for digitization."""

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

from xrd_digitization.detect_panels import detect_plot_panels
from xrd_digitization.types import PlotPanel

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
HttpPost = Callable[..., Any]

PANEL_LOCATOR_PROMPT = """\
Locate distinct subplot / axes-frame panels in this scientific figure.
Return JSON only:
{
  "panels": [
    {
      "label": "A",
      "bbox_norm": [x0, y0, x1, y1],
      "plot_type": "stacked_xrd" | "single_xrd" | "other",
      "digitizable_after_split": true
    }
  ]
}
bbox_norm values are fractions of image width/height in [0, 1].
Prefer visible panel letters (A, B, C...) when present.
Include axes ticks/labels inside each bbox. Omit non-plot decorations.
"""


@dataclass
class PanelCropRecord:
    panel: PlotPanel
    image_path: Path
    figure_caption: str | None = None
    panel_body_context: str | None = None
    axes_bbox: tuple[int, int, int, int] | None = None
    panel_source_bbox: tuple[int, int, int, int] | None = None


@dataclass
class FigurePanelSplitResult:
    figure_id: str
    source_png: Path
    panels: list[PanelCropRecord] = field(default_factory=list)
    panels_json_path: Path | None = None
    method: str = "geometric"

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "source_png": str(self.source_png),
            "method": self.method,
            "panel_count": len(self.panels),
            "panels": [
                {
                    "label": rec.panel.label,
                    "index": rec.panel.index,
                    "bbox": list(rec.panel.bbox),
                    "axes_bbox": list(rec.axes_bbox) if rec.axes_bbox else None,
                    "panel_source_bbox": list(rec.panel_source_bbox)
                    if rec.panel_source_bbox
                    else list(rec.panel.bbox),
                    "plot_type": rec.panel.plot_type,
                    "digitizable_after_split": rec.panel.digitizable_after_split,
                    "label_confidence": rec.panel.label_confidence,
                    "detection_method": rec.panel.detection_method,
                    "image_path": str(rec.image_path),
                    "figure_caption": rec.figure_caption,
                    "panel_body_context": rec.panel_body_context,
                }
                for rec in self.panels
            ],
        }


def _encode_image_png_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image for panel locator")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Panel locator response must be a JSON object")
    return data


def _normalize_api_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.strip()
    if not key:
        return None
    quotes = "'\"‘’“”`"
    while len(key) >= 2 and key[0] in quotes and key[-1] in quotes:
        key = key[1:-1].strip()
    key = key.strip(quotes + " \t\r\n")
    return key or None


def call_openai_panel_locator(
    image_bgr: np.ndarray,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 120.0,
    http_post: HttpPost | None = None,
) -> list[PlotPanel]:
    """Vision fallback: return panel bboxes from an OpenAI-compatible API."""
    key = _normalize_api_key(api_key or os.environ.get("OPENAI_API_KEY"))
    if not key and http_post is None:
        raise RuntimeError("Missing OPENAI_API_KEY for panel locator")
    key = key or "test-key"

    url_base = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_name = model or os.environ.get("OPENAI_TRIAGE_MODEL") or DEFAULT_MODEL
    b64 = _encode_image_png_b64(image_bgr)
    height, width = image_bgr.shape[:2]

    body = {
        "model": model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PANEL_LOCATOR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
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
        raise ValueError(f"Unexpected panel locator response: {data!r}") from exc
    payload = _parse_json_content(content if isinstance(content, str) else "")
    raw_panels = payload.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise ValueError("Panel locator returned no panels")

    panels: list[PlotPanel] = []
    for index, item in enumerate(raw_panels, start=1):
        if not isinstance(item, dict):
            continue
        bbox_norm = item.get("bbox_norm") or item.get("bbox")
        if not isinstance(bbox_norm, (list, tuple)) or len(bbox_norm) != 4:
            continue
        nx0, ny0, nx1, ny1 = [float(v) for v in bbox_norm]
        # Accept either normalized [0,1] or absolute pixel coords.
        if max(nx0, ny0, nx1, ny1) <= 1.5:
            x0 = int(round(nx0 * width))
            y0 = int(round(ny0 * height))
            x1 = int(round(nx1 * width))
            y1 = int(round(ny1 * height))
        else:
            x0, y0, x1, y1 = int(nx0), int(ny0), int(nx1), int(ny1)
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        label = str(item.get("label") or "").strip().upper() or None
        if label and len(label) > 2:
            label = label[0]
        panels.append(
            PlotPanel(
                index=index,
                bbox=(x0, y0, x1, y1),
                label=label,
                plot_type=str(item.get("plot_type") or "") or None,
                digitizable_after_split=bool(item.get("digitizable_after_split", True)),
                label_confidence=0.9 if label else 0.0,
                detection_method="openai_vision",
            )
        )
    if len(panels) < 2:
        raise ValueError("Panel locator returned fewer than 2 usable panels")

    # Fill missing labels in reading order without clobbering detected ones.
    used = {p.label for p in panels if p.label}
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    next_i = 0
    for panel in panels:
        if panel.label:
            continue
        while next_i < len(alphabet) and alphabet[next_i] in used:
            next_i += 1
        panel.label = alphabet[next_i] if next_i < len(alphabet) else str(panel.index)
        used.add(panel.label)
        next_i += 1
    return panels


def extract_panel_body_context(
    *,
    panel_label: str | None,
    figure_caption: str | None,
    figure_body_text: str | None,
) -> str | None:
    """
    Return panel-specific body-text snippets (not a split caption).

    Caption is inherited wholesale by callers; this only mines nearby body text
    mentioning the panel letter (e.g. ``Figure 1A``, ``Fig. 1B``).
    """
    if not panel_label:
        return None
    text = " ".join(
        part for part in (figure_body_text or "",) if part
    ).strip()
    if not text:
        return None

    label = panel_label.strip().upper()
    patterns = [
        rf"(?i)(?:figure|fig\.?)\s*\d+\s*{re.escape(label)}\b[^.?!]*[.?!]?",
        rf"(?i)(?:figure|fig\.?)\s*\d+\s*[\(\[]?{re.escape(label)}[\)\]]?[^.?!]*[.?!]?",
        rf"(?i)\b{re.escape(label)}\)\s+[^.?!]*[.?!]?",
    ]
    snippets: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            snippet = match.group(0).strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)
    if not snippets:
        return None
    return " ".join(snippets[:4])


def load_figure_text_context(
    paper_dir: Path,
    *,
    figure_id: str,
) -> tuple[str | None, str | None]:
    """Load (caption, body_text) for ``figure_id`` from figure_analysis JSON if present."""
    paper_dir = Path(paper_dir)
    candidates = list(paper_dir.glob("*.figure_analysis.json"))
    if not candidates:
        return None, None
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, list):
        return None, None

    # fig_1 → 1
    m = re.search(r"(\d+)$", figure_id)
    fig_num = int(m.group(1)) if m else None
    for item in payload:
        if not isinstance(item, dict):
            continue
        if fig_num is not None and int(item.get("figure") or -1) == fig_num:
            caption = str(item.get("caption") or "").strip() or None
            body = str(item.get("text") or "").strip() or None
            return caption, body
    return None, None


def _text_extent_pads(
    image_bgr: np.ndarray,
    axes_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Estimate extra padding (left, top, right, bottom) from dark text / OCR
    outside the axes frame.
    """
    height, width = image_bgr.shape[:2]
    ax0, ay0, ax1, ay1 = axes_bbox
    axes_w = max(1, ax1 - ax0)
    axes_h = max(1, ay1 - ay0)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Bottom band: tick numbers + axis title.
    bottom_y0 = ay1
    bottom_y1 = min(height, ay1 + max(40, int(axes_h * 0.35)))
    bottom = gray[bottom_y0:bottom_y1, max(0, ax0 - 10) : min(width, ax1 + 10)]
    extra_bottom = 0
    if bottom.size:
        dark_rows = (bottom < 140).mean(axis=1)
        active = np.flatnonzero(dark_rows > 0.01)
        if active.size:
            extra_bottom = int(active[-1]) + 12

    # Left band: y-axis title / tick stubs.
    left_x0 = max(0, ax0 - max(40, int(axes_w * 0.25)))
    left_x1 = ax0
    left = gray[max(0, ay0) : min(height, ay1), left_x0:left_x1]
    extra_left = 0
    if left.size and left_x1 > left_x0:
        dark_cols = (left < 140).mean(axis=0)
        active = np.flatnonzero(dark_cols > 0.01)
        if active.size:
            # Distance from axes left edge back to leftmost dark ink.
            extra_left = int((left_x1 - left_x0) - active[0]) + 8

    # Top band: panel letter / title.
    top_y0 = max(0, ay0 - max(24, int(axes_h * 0.18)))
    top_y1 = ay0
    top = gray[top_y0:top_y1, max(0, ax0) : min(width, ax1)]
    extra_top = 0
    if top.size and top_y1 > top_y0:
        dark_rows = (top < 140).mean(axis=1)
        active = np.flatnonzero(dark_rows > 0.008)
        if active.size:
            extra_top = int((top_y1 - top_y0) - active[0]) + 6

    # Right band: legends often sit inside axes; still grab overhanging labels.
    right_x0 = ax1
    right_x1 = min(width, ax1 + max(24, int(axes_w * 0.12)))
    right = gray[max(0, ay0) : min(height, ay1), right_x0:right_x1]
    extra_right = 0
    if right.size and right_x1 > right_x0:
        dark_cols = (right < 140).mean(axis=0)
        active = np.flatnonzero(dark_cols > 0.01)
        if active.size:
            extra_right = int(active[-1]) + 8

    # OCR boost for numeric tick labels below the axes when available.
    try:
        import pytesseract
        from pytesseract import Output

        band = image_bgr[
            bottom_y0:bottom_y1,
            max(0, ax0) : min(width, ax1),
        ]
        if band.size:
            data = pytesseract.image_to_data(
                band,
                config="--psm 6",
                output_type=Output.DICT,
            )
            max_bottom = extra_bottom
            for i, text in enumerate(data.get("text") or []):
                token = str(text or "").strip()
                if not token:
                    continue
                if not any(ch.isdigit() for ch in token) and "theta" not in token.lower() and "θ" not in token:
                    # Keep axis-title-like tokens too.
                    if not any(k in token.lower() for k in ("degree", "2", "deg")):
                        continue
                try:
                    top_i = int(data["top"][i])
                    h_i = int(data["height"][i])
                except (TypeError, ValueError, KeyError):
                    continue
                max_bottom = max(max_bottom, top_i + h_i + 10)
            extra_bottom = max(extra_bottom, max_bottom)
    except Exception:
        pass

    return (
        max(0, extra_left),
        max(0, extra_top),
        max(0, extra_right),
        max(0, extra_bottom),
    )


def expand_axes_to_panel_source_bbox(
    axes_bbox: tuple[int, int, int, int],
    *,
    image_bgr: np.ndarray,
    neighbors: list[tuple[int, int, int, int]] | None = None,
    pad_left_frac: float = 0.10,
    pad_right_frac: float = 0.08,
    pad_top_frac: float = 0.08,
    pad_bottom_frac: float = 0.15,
) -> tuple[int, int, int, int]:
    """
    Build a padded ``panel_source_bbox`` around ``axes_bbox``.

    Asymmetric percentage pads plus detected text extents; clamped to the
    figure and midpoints between neighboring panels.
    """
    height, width = image_bgr.shape[:2]
    ax0, ay0, ax1, ay1 = axes_bbox
    axes_w = max(1, ax1 - ax0)
    axes_h = max(1, ay1 - ay0)

    pad_left = max(12, int(axes_w * pad_left_frac))
    pad_right = max(10, int(axes_w * pad_right_frac))
    pad_top = max(10, int(axes_h * pad_top_frac))
    pad_bottom = max(18, int(axes_h * pad_bottom_frac))

    t_left, t_top, t_right, t_bottom = _text_extent_pads(image_bgr, axes_bbox)
    pad_left = max(pad_left, t_left)
    pad_top = max(pad_top, t_top)
    pad_right = max(pad_right, t_right)
    pad_bottom = max(pad_bottom, t_bottom)

    nx0 = max(0, ax0 - pad_left)
    ny0 = max(0, ay0 - pad_top)
    nx1 = min(width, ax1 + pad_right)
    ny1 = min(height, ay1 + pad_bottom)

    for ox0, oy0, ox1, oy1 in neighbors or []:
        if (ox0, oy0, ox1, oy1) == axes_bbox:
            continue
        # Side-by-side vs stacked.
        if abs(((ox0 + ox1) / 2) - ((ax0 + ax1) / 2)) > abs(
            ((oy0 + oy1) / 2) - ((ay0 + ay1) / 2)
        ):
            if ox0 >= ax1:
                mid = (ax1 + ox0) // 2
                nx1 = min(nx1, mid)
            elif ox1 <= ax0:
                mid = (ox1 + ax0) // 2
                nx0 = max(nx0, mid)
        else:
            if oy0 >= ay1:
                mid = (ay1 + oy0) // 2
                ny1 = min(ny1, mid)
            elif oy1 <= ay0:
                mid = (oy1 + ay0) // 2
                ny0 = max(ny0, mid)

    return (int(nx0), int(ny0), int(nx1), int(ny1))


def write_panel_crops(
    source_png: Path,
    panels: list[PlotPanel],
    *,
    figure_id: str | None = None,
    figure_caption: str | None = None,
    figure_body_text: str | None = None,
    output_dir: Path | None = None,
) -> FigurePanelSplitResult:
    """
    Crop each panel to ``{figure_id}_{label}.png`` beside the source figure.

    Detected ``axes_bbox`` is padded into ``panel_source_bbox`` so tick labels
    and axis titles are preserved. Leaves ``{figure_id}.png`` untouched.
    """
    source_png = Path(source_png)
    image = cv2.imread(str(source_png))
    if image is None:
        raise FileNotFoundError(f"Could not read figure image: {source_png}")

    figure_id = figure_id or source_png.stem
    out_dir = Path(output_dir) if output_dir is not None else source_png.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    method = panels[0].detection_method if panels else "unknown"
    # Axes boxes before padding (detect_plot_panels returns frame boxes).
    axes_boxes = [tuple(p.axes_bbox or p.bbox) for p in panels]
    records: list[PanelCropRecord] = []
    for panel, axes_bbox in zip(panels, axes_boxes):
        label = (panel.label or str(panel.index)).strip()
        axes_bbox_i = (
            int(axes_bbox[0]),
            int(axes_bbox[1]),
            int(axes_bbox[2]),
            int(axes_bbox[3]),
        )
        source_bbox = expand_axes_to_panel_source_bbox(
            axes_bbox_i,
            image_bgr=image,
            neighbors=list(axes_boxes),
        )
        x0, y0, x1, y1 = source_bbox
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop_path = out_dir / f"{figure_id}_{label}.png"
        cv2.imwrite(str(crop_path), crop)

        panel.axes_bbox = axes_bbox_i
        panel.panel_source_bbox = source_bbox
        panel.bbox = source_bbox  # public bbox is the source crop used downstream

        body_ctx = extract_panel_body_context(
            panel_label=label,
            figure_caption=figure_caption,
            figure_body_text=figure_body_text,
        )
        records.append(
            PanelCropRecord(
                panel=panel,
                image_path=crop_path,
                figure_caption=figure_caption,
                panel_body_context=body_ctx,
                axes_bbox=axes_bbox_i,
                panel_source_bbox=source_bbox,
            )
        )

    result = FigurePanelSplitResult(
        figure_id=figure_id,
        source_png=source_png,
        panels=records,
        method=str(method or "geometric"),
    )
    panels_json = out_dir / f"{figure_id}.panels.json"
    panels_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    result.panels_json_path = panels_json
    return result


def split_figure_into_panels(
    source_png: Path,
    *,
    figure_id: str | None = None,
    figure_caption: str | None = None,
    figure_body_text: str | None = None,
    output_dir: Path | None = None,
    force_vision: bool = False,
    model: str | None = None,
    http_post: HttpPost | None = None,
) -> FigurePanelSplitResult:
    """
    Detect panels geometrically; if fewer than 2 panels are found (or
    ``force_vision``), fall back to OpenAI bbox localization.
    """
    source_png = Path(source_png)
    image = cv2.imread(str(source_png))
    if image is None:
        raise FileNotFoundError(f"Could not read figure image: {source_png}")

    panels = [] if force_vision else detect_plot_panels(image)
    method = "geometric"
    if len(panels) < 2:
        LOGGER.info(
            "Geometric panel detection found %d panel(s); trying OpenAI locator",
            len(panels),
        )
        try:
            panels = call_openai_panel_locator(
                image,
                model=model,
                http_post=http_post,
            )
            method = "openai_vision"
        except Exception as exc:
            LOGGER.warning("OpenAI panel locator failed: %s", exc)
            if len(panels) < 2:
                raise RuntimeError(
                    f"Could not split multipanel figure {source_png.name}: {exc}"
                ) from exc

    # Ensure labels exist
    used = {p.label for p in panels if p.label}
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    next_i = 0
    for panel in panels:
        if panel.label:
            continue
        while next_i < len(alphabet) and alphabet[next_i] in used:
            next_i += 1
        panel.label = alphabet[next_i] if next_i < len(alphabet) else str(panel.index)
        used.add(panel.label)
        panel.detection_method = panel.detection_method or method
        next_i += 1

    result = write_panel_crops(
        source_png,
        panels,
        figure_id=figure_id,
        figure_caption=figure_caption,
        figure_body_text=figure_body_text,
        output_dir=output_dir,
    )
    result.method = method
    if result.panels_json_path is not None:
        result.panels_json_path.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
    return result
