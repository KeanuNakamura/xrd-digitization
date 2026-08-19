"""Coordinate-space transforms for agent metadata ↔ plot-crop pixels."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


BBox = Sequence[float]


def transform_bbox(
    bbox: BBox,
    source_image_size: tuple[int, int],
    target_crop_bbox: tuple[int, int, int, int],
    target_image_size: tuple[int, int],
    *,
    actual_image_size: tuple[int, int] | None = None,
) -> list[float]:
    """
    Map a bbox from agent coordinate space into plot-crop image pixels.

    Parameters
    ----------
    bbox:
        ``[x1, y1, x2, y2]`` in the agent's coordinate space.
    source_image_size:
        ``(width, height)`` of the image the agent annotated.
    target_crop_bbox:
        ``(x0, y0, x1, y1)`` of the plot crop in *actual* original-image pixels.
    target_image_size:
        ``(width, height)`` of the cropped array being cleaned / digitized.
    actual_image_size:
        ``(width, height)`` of the real original image. When this differs from
        ``source_image_size``, the bbox is scaled into actual image space first.
    """
    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]

    src_w, src_h = (max(int(source_image_size[0]), 1), max(int(source_image_size[1]), 1))
    act_w, act_h = actual_image_size if actual_image_size is not None else (src_w, src_h)
    act_w, act_h = max(int(act_w), 1), max(int(act_h), 1)

    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Agent space → actual original-image pixels.
    sx = act_w / src_w
    sy = act_h / src_h
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy

    crop_x0, crop_y0, crop_x1, crop_y1 = (int(v) for v in target_crop_bbox)
    crop_w = max(crop_x1 - crop_x0, 1)
    crop_h = max(crop_y1 - crop_y0, 1)
    tgt_w, tgt_h = (max(int(target_image_size[0]), 1), max(int(target_image_size[1]), 1))

    # Crop-local, then scale if the cropped array was resized.
    local_x1 = (x1 - crop_x0) * (tgt_w / crop_w)
    local_y1 = (y1 - crop_y0) * (tgt_h / crop_h)
    local_x2 = (x2 - crop_x0) * (tgt_w / crop_w)
    local_y2 = (y2 - crop_y0) * (tgt_h / crop_h)

    local_x1 = float(np.clip(local_x1, 0, tgt_w))
    local_y1 = float(np.clip(local_y1, 0, tgt_h))
    local_x2 = float(np.clip(local_x2, 0, tgt_w))
    local_y2 = float(np.clip(local_y2, 0, tgt_h))
    return [local_x1, local_y1, local_x2, local_y2]


def bbox_dark_ink_fraction(
    gray: np.ndarray,
    bbox: BBox,
    *,
    thresh: int = 180,
) -> float:
    """Fraction of dark pixels inside ``bbox`` (crop-local gray image)."""
    if gray.ndim != 2 or not bbox or len(bbox) < 4:
        return 0.0
    h, w = gray.shape
    x1 = max(0, int(np.floor(bbox[0])))
    y1 = max(0, int(np.floor(bbox[1])))
    x2 = min(w, int(np.ceil(bbox[2])))
    y2 = min(h, int(np.ceil(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float((roi < thresh).mean())


def boxes_overlap_annotation_text(
    image_bgr: np.ndarray,
    bboxes: Sequence[BBox],
    *,
    min_dark_fraction: float = 0.05,
    min_overlapping_boxes: float = 0.5,
) -> tuple[bool, dict]:
    """
    Gate text removal: transformed boxes must overlap annotation ink.

    Returns ``(ok, stats)``.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, {"reason": "empty_image", "fractions": []}
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else image_bgr
    )
    fractions = [bbox_dark_ink_fraction(gray, box) for box in bboxes]
    if not fractions:
        return False, {"reason": "no_boxes", "fractions": []}
    overlapping = sum(1 for f in fractions if f >= min_dark_fraction)
    ratio = overlapping / len(fractions)
    ok = ratio >= min_overlapping_boxes and overlapping >= 1
    return ok, {
        "overlapping_boxes": overlapping,
        "total_boxes": len(fractions),
        "overlap_ratio": ratio,
        "fractions": fractions,
        "mean_dark_fraction": float(np.mean(fractions)),
    }


def render_bbox_debug_overlay(
    image_bgr: np.ndarray,
    bboxes: Sequence[BBox],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw boxes on a copy of the exact image that will be cleaned."""
    overlay = image_bgr.copy()
    for index, box in enumerate(bboxes):
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and index < len(labels) and labels[index]:
            cv2.putText(
                overlay,
                str(labels[index])[:24],
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return overlay
