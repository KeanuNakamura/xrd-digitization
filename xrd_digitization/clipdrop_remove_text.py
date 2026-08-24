"""ClipDrop Remove Text: clean only the plot interior, preserve axes pixels.

Flow:
  original → detect/inset plot interior → ClipDrop Remove Text on crop →
  paste cleaned crop into a copy of the original → write ``*_clean.png``

Everything outside the inset crop remains byte-identical to the original.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np
import requests

from xrd_digitization.plot_interior_crop import (
    DEFAULT_INSET_FRAC,
    PlotInteriorCrop,
    load_image_bgr,
    resolve_plot_interior_bbox,
)

LOGGER = logging.getLogger(__name__)

CLIPDROP_REMOVE_TEXT_URL = "https://clipdrop-api.co/remove-text/v1"
CLIPDROP_API_KEY_ENV = "CLIPDROP_API_KEY"
DEFAULT_TIMEOUT_S = 120.0


class ClipdropError(RuntimeError):
    """ClipDrop API or response handling failure."""


class HttpPost(Protocol):
    def __call__(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ClipdropCleanResult:
    cleaned_bgr: Any
    output_path: Path | None
    interior: PlotInteriorCrop
    warnings: list[str] = field(default_factory=list)


def get_clipdrop_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get(CLIPDROP_API_KEY_ENV)
    if not key:
        raise ClipdropError(
            f"Missing ClipDrop API key; set {CLIPDROP_API_KEY_ENV} or pass api_key="
        )
    return key


def encode_image_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ClipdropError("Failed to encode crop as PNG for ClipDrop")
    return buf.tobytes()


def decode_image_bytes(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ClipdropError("ClipDrop response was not a decodable image")
    return image


def call_clipdrop_remove_text(
    image_bgr: np.ndarray,
    *,
    api_key: str | None = None,
    url: str = CLIPDROP_REMOVE_TEXT_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    http_post: HttpPost | None = None,
    filename: str = "crop.png",
) -> np.ndarray:
    """
    Send an image crop to ClipDrop Remove Text and return the cleaned BGR image.

    ``http_post`` is injectable for tests (same pattern as agent_guidance).
    """
    key = get_clipdrop_api_key(api_key)
    png_bytes = encode_image_png(image_bgr)
    headers = {"x-api-key": key}
    files = {
        "image_file": (filename, png_bytes, "image/png"),
    }
    post = http_post or requests.post
    try:
        response = post(url, headers=headers, files=files, timeout=timeout_s)
    except requests.RequestException as exc:
        raise ClipdropError(f"ClipDrop request failed: {exc}") from exc

    status = getattr(response, "status_code", None)
    if status is None or int(status) >= 400:
        body = getattr(response, "text", "") or ""
        raise ClipdropError(
            f"ClipDrop Remove Text failed (HTTP {status}): {body[:500]}"
        )

    content = getattr(response, "content", None)
    if not content:
        raise ClipdropError("ClipDrop returned an empty response body")
    return decode_image_bytes(content)


def match_crop_size(
    cleaned_bgr: np.ndarray,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    """Ensure cleaned crop matches ``(height, width)`` of the source crop."""
    warnings: list[str] = []
    target_h, target_w = target_hw
    h, w = cleaned_bgr.shape[:2]
    if (h, w) == (target_h, target_w):
        return cleaned_bgr, warnings

    warnings.append(f"clipdrop_crop_resized:{(w, h)}->{(target_w, target_h)}")
    resized = cv2.resize(
        cleaned_bgr,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA if (h > target_h or w > target_w) else cv2.INTER_LINEAR,
    )
    return resized, warnings


def paste_crop_into_image(
    original_bgr: np.ndarray,
    cleaned_crop_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[str]]:
    """
    Paste ``cleaned_crop_bgr`` into a copy of ``original_bgr`` at ``bbox``.

    Outside the bbox, pixels are unchanged copies of the original.
    """
    x0, y0, x1, y1 = bbox
    target_h = y1 - y0
    target_w = x1 - x0
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid paste bbox: {bbox}")

    matched, warnings = match_crop_size(cleaned_crop_bgr, (target_h, target_w))
    composite = original_bgr.copy()
    composite[y0:y1, x0:x1] = matched
    return composite, warnings


def assert_exterior_pixel_identical(
    original_bgr: np.ndarray,
    composite_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> None:
    """Raise if any pixel outside ``bbox`` differs from the original."""
    if original_bgr.shape != composite_bgr.shape:
        raise ClipdropError(
            f"Composite shape {composite_bgr.shape} != original {original_bgr.shape}"
        )
    x0, y0, x1, y1 = bbox
    mask = np.ones(original_bgr.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = False
    if not np.array_equal(original_bgr[mask], composite_bgr[mask]):
        raise ClipdropError("Composite exterior pixels differ from original")


def default_clean_output_path(image_path: Path, *, dry_run: bool = False) -> Path:
    """``figure_1.png`` → ``figure_1_clean.png`` (never overwrites the source).

    Dry-run writes ``*_clean_dryrun.png`` so existing ClipDrop cleans are not clobbered.
    """
    image_path = Path(image_path)
    stem = image_path.stem
    if dry_run:
        base = stem[: -len("_clean")] if stem.endswith("_clean") else stem
        return image_path.with_name(f"{base}_clean_dryrun.png")
    if stem.endswith("_clean"):
        return image_path.with_name(f"{stem}_clipdrop.png")
    return image_path.with_name(f"{stem}_clean.png")


def clean_figure_preserve_axes(
    image: str | Path | np.ndarray,
    *,
    output_path: str | Path | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    inset_px: int | None = None,
    inset_frac: float = DEFAULT_INSET_FRAC,
    apply_inset_to_manual: bool = True,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    http_post: HttpPost | None = None,
    dry_run: bool = False,
    verify_exterior: bool = True,
    remove_text_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ClipdropCleanResult:
    """
    Remove in-plot text via ClipDrop while preserving axis labels from the original.

    When ``dry_run`` is True, the crop is pasted back unchanged (no API call) —
    useful for validating bbox / exterior identity without spending credits.
    """
    source_path: Path | None
    if isinstance(image, (str, Path)):
        source_path = Path(image)
        original_bgr = load_image_bgr(source_path)
    else:
        source_path = None
        original_bgr = load_image_bgr(image)

    interior = resolve_plot_interior_bbox(
        original_bgr,
        bbox=bbox,
        inset_px=inset_px,
        inset_frac=inset_frac,
        apply_inset_to_manual=apply_inset_to_manual,
    )
    warnings = list(interior.warnings)

    if dry_run:
        cleaned_crop = interior.cropped_bgr.copy()
        warnings.append("clipdrop_dry_run")
    elif remove_text_fn is not None:
        cleaned_crop = remove_text_fn(interior.cropped_bgr)
    else:
        cleaned_crop = call_clipdrop_remove_text(
            interior.cropped_bgr,
            api_key=api_key,
            timeout_s=timeout_s,
            http_post=http_post,
            filename=(source_path.stem + "_interior.png") if source_path else "interior.png",
        )

    composite, paste_warnings = paste_crop_into_image(
        original_bgr, cleaned_crop, interior.bbox
    )
    warnings.extend(paste_warnings)

    if verify_exterior:
        assert_exterior_pixel_identical(original_bgr, composite, interior.bbox)

    out: Path | None = None
    if output_path is not None:
        out = Path(output_path)
    elif source_path is not None:
        out = default_clean_output_path(source_path, dry_run=dry_run)

    if out is not None:
        if source_path is not None and out.resolve() == source_path.resolve():
            raise ClipdropError(
                f"Refusing to overwrite original image: {source_path}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), composite):
            raise ClipdropError(f"Failed to write cleaned image: {out}")
        LOGGER.info(
            "Wrote ClipDrop composite %s (interior=%s method=%s)",
            out.name,
            interior.bbox,
            interior.method,
        )

    return ClipdropCleanResult(
        cleaned_bgr=composite,
        output_path=out,
        interior=interior,
        warnings=warnings,
    )
