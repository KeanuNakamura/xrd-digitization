"""Optional, non-blocking extraction of in-plot scientific annotations."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import cv2

LOGGER = logging.getLogger(__name__)

_MILLER_RE = re.compile(
    r"[\(\[]\s*-?\d\s*-?\d\s*-?\d\s*[\)\]]|(?:×|x)\s*\d+",
    re.IGNORECASE,
)


def extract_plot_annotations(
    image_path: Path,
    *,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Best-effort OCR of Miller indices / peak multipliers.

    Failures return an empty list and never raise to callers that digitize.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Annotation OCR unavailable: %s", exc)
        return []

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        data = pytesseract.image_to_data(
            gray,
            config="--psm 11",
            output_type=Output.DICT,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Annotation OCR failed for %s: %s", image_path.name, exc)
        return []

    annotations: list[dict[str, Any]] = []
    n = len(data.get("text") or [])
    for i in range(n):
        text = str(data["text"][i] or "").strip()
        if not text or not _MILLER_RE.search(text):
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 40:
            continue
        annotations.append(
            {
                "text": text,
                "confidence": conf / 100.0,
                "bbox": [
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["left"][i]) + int(data["width"][i]),
                    int(data["top"][i]) + int(data["height"][i]),
                ],
            }
        )

    if output_path is not None:
        try:
            Path(output_path).write_text(
                json.dumps({"annotations": annotations}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.debug("Could not write annotations for %s: %s", image_path.name, exc)
    return annotations
