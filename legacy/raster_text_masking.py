"""
Geometry-first raster text masking for scientific figure digitization.

Goal: find and mask text-like annotations inside the plotting rectangle so
curves are easier to digitize. OCR transcription is optional supporting
evidence — acceptance does not require a correct alphanumeric string.

Preserve axis ticks/titles via spatial bands. Prefer glyph masks over large
rectangular erasures so plotted curves are not wiped out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Coverage / safety.
MAX_REMOVABLE_PLOT_COVERAGE = 0.12
MAX_AXIS_BAND_REMOVABLE_OVERLAP = 0.08
MAX_CURVE_LIKE_SPAN_FRAC = 0.35
REJECT_DEBUG_MAX = 50
PROPOSAL_NMS_IOU = 0.55

# Component geometry (relative to image).
MIN_COMP_AREA = 8
MAX_COMP_AREA_FRAC = 0.004
MAX_COMP_HEIGHT_FRAC = 0.07
MAX_COMP_WIDTH_FRAC = 0.06
MAX_COMP_ASPECT = 6.0
MIN_COMP_ASPECT = 1.0 / 6.0

# Grouping.
VERT_X_TOL_FRAC = 0.012
HORZ_Y_TOL_FRAC = 0.018
MAX_GAP_FRAC = 0.022
MIN_GROUP_COMPONENTS = 1

# OCR is supporting evidence only.
OCR_SUPPORT_MIN_CONF = 20.0
IN_PLOT_OCR_SUPPORT_MIN_CONF = 15.0

REMOVABLE_ROLE_PREFIX = "removable_"
PRESERVED_ROLES = frozenset(
    {
        "preserved_axis_tick",
        "preserved_axis_title",
        "caption_or_external_text",
    }
)
REMOVABLE_ROLES = frozenset(
    {
        "removable_peak_annotation",
        "removable_curve_label",
        "removable_legend_text",
        "removable_panel_label",
        "removable_in_plot_text",
    }
)


@dataclass
class RasterTextDetectionResult:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    by_role: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    plot_bbox: tuple[int, int, int, int] | None = None
    caption_bbox: tuple[int, int, int, int] | None = None
    inner_plot_bbox: tuple[int, int, int, int] | None = None
    axis_bands: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    panels: list[tuple[int, int, int, int]] = field(default_factory=list)
    # Digitization default: glyph mask of removable in-plot text.
    mask: np.ndarray | None = None
    all_text_candidate_mask: np.ndarray | None = None
    removable_region_mask: np.ndarray | None = None
    removable_glyph_mask: np.ndarray | None = None
    preserved_axis_mask: np.ndarray | None = None
    coverage: float = 0.0
    failed: bool = False
    failure_reason: str | None = None
    debug_accepted_bgr: np.ndarray | None = None
    debug_preserved_bgr: np.ndarray | None = None
    debug_rejected_bgr: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_width(box: Sequence[float]) -> float:
    return max(0.0, float(box[2] - box[0]))


def _box_height(box: Sequence[float]) -> float:
    return max(0.0, float(box[3] - box[1]))


def _clamp_box(box: Sequence[float], width: int, height: int) -> list[float]:
    x0 = float(max(0, min(width, box[0])))
    y0 = float(max(0, min(height, box[1])))
    x1 = float(max(0, min(width, box[2])))
    y1 = float(max(0, min(height, box[3])))
    return [x0, y0, x1, y1]


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _iou_box(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _center(box: Sequence[float]) -> tuple[float, float]:
    return 0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3]))


def _point_in_box(x: float, y: float, box: Sequence[float], *, pad: float = 0.0) -> bool:
    return (
        float(box[0]) - pad <= x <= float(box[2]) + pad
        and float(box[1]) - pad <= y <= float(box[3]) + pad
    )


def mask_coverage_fraction(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return float((mask > 0).mean())


def build_box_mask_array(
    boxes: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    padding_px: int = 1,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        if len(box) < 4:
            continue
        x0 = max(0, int(box[0]) - padding_px)
        y0 = max(0, int(box[1]) - padding_px)
        x1 = min(width, int(box[2]) + padding_px)
        y1 = min(height, int(box[3]) + padding_px)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _nms_xywh_proposals(
    proposals: Sequence[tuple[int, int, int, int]],
    *,
    iou_threshold: float = PROPOSAL_NMS_IOU,
) -> tuple[list[tuple[int, int, int, int]], int]:
    if not proposals:
        return [], 0
    boxes = []
    for x, y, w, h in proposals:
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    order = sorted(range(len(boxes)), key=lambda i: _box_area(boxes[i]), reverse=True)
    kept_idx: list[int] = []
    suppressed = 0
    while order:
        i = order.pop(0)
        kept_idx.append(i)
        remaining: list[int] = []
        for j in order:
            if _iou_box(boxes[i], boxes[j]) >= iou_threshold:
                suppressed += 1
            else:
                remaining.append(j)
        order = remaining
    kept = [proposals[i] for i in kept_idx]
    return kept, suppressed


def _select_rejected_for_debug(
    rejected: Sequence[dict[str, Any]],
    *,
    max_count: int = REJECT_DEBUG_MAX,
) -> list[dict[str, Any]]:
    scored = []
    for det in rejected:
        box = det.get("bbox") or [0, 0, 0, 0]
        conf = det.get("detection_confidence")
        if conf is None:
            conf = det.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf_f = 0.0
        scored.append((conf_f, _box_area(box), det))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [t[2] for t in scored[:max_count]]


def _dedupe_detections(
    detections: Sequence[dict[str, Any]],
    *,
    iou_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    if not detections:
        return []

    def _score(det: dict[str, Any]) -> tuple[float, float, float]:
        n_comp = float(det.get("n_components") or 1)
        det_conf = float(det.get("detection_confidence") or 0.0)
        area = _box_area(det.get("bbox") or [0, 0, 0, 0])
        # Prefer multi-character groups over OCR fragments / singletons.
        return (n_comp, det_conf, area)

    order = sorted(range(len(detections)), key=lambda i: _score(detections[i]), reverse=True)
    kept: list[dict[str, Any]] = []
    kept_boxes: list[Sequence[float]] = []
    for i in order:
        det = detections[i]
        box = det.get("bbox")
        if not box:
            continue
        suppress = False
        for kb in kept_boxes:
            iou = _iou_box(box, kb)
            if iou >= iou_threshold:
                suppress = True
                break
            # Also suppress near-duplicates that share center and similar width
            # (rotated + upright views of the same vertical label).
            inter = _intersection_area(box, kb)
            smaller = min(_box_area(box), _box_area(kb))
            if smaller > 0 and inter / smaller >= 0.65:
                suppress = True
                break
            cx0, cy0 = _center(box)
            cx1, cy1 = _center(kb)
            if abs(cx0 - cx1) <= max(4.0, 0.4 * min(_box_width(box), _box_width(kb))) and (
                _intersection_area(
                    [box[0], min(box[1], kb[1]), box[2], max(box[3], kb[3])],
                    kb,
                )
                / max(_box_area(kb), 1.0)
                >= 0.5
            ):
                # Same vertical column, heavily overlapping in y-extent.
                suppress = True
                break
        if suppress:
            continue
        kept.append(det)
        kept_boxes.append(box)
    return kept


# ---------------------------------------------------------------------------
# Plot frame + axis bands
# ---------------------------------------------------------------------------


def detect_inner_plot_frame(
    image_bgr: np.ndarray,
    *,
    plot_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Detect the inner axes rectangle used as the plotting region."""
    height, width = image_bgr.shape[:2]
    if plot_bbox is None:
        rx0, ry0, rx1, ry1 = 0, 0, width, height
    else:
        rx0, ry0, rx1, ry1 = plot_bbox

    region = image_bgr[ry0:ry1, rx0:rx1]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    left = right = top = bottom = None
    try:
        from xrd_digitization.crop_plot_area import _detect_axis_lines

        left, right, top, bottom = _detect_axis_lines(gray)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("axis-line detection unavailable: %s", exc)

    rh, rw = gray.shape
    if left is None or right is None or top is None or bottom is None:
        # Fallback margins inside the plot-only crop.
        left = int(rw * 0.10)
        right = int(rw * 0.98)
        top = int(rh * 0.05)
        bottom = int(rh * 0.90)
    if right <= left + 20 or bottom <= top + 20:
        left, top, right, bottom = int(rw * 0.08), int(rh * 0.04), int(rw * 0.98), int(rh * 0.92)

    return (
        int(rx0 + left),
        int(ry0 + top),
        int(rx0 + right),
        int(ry0 + bottom),
    )


def define_axis_bands(
    *,
    image_width: int,
    image_height: int,
    inner_plot: tuple[int, int, int, int],
    plot_bbox: tuple[int, int, int, int],
    caption_bbox: tuple[int, int, int, int] | None,
) -> dict[str, tuple[int, int, int, int]]:
    """
    Explicit axis bands outside the inner plotting rectangle.

    Text whose center falls in these bands is preserved (not removable).
    """
    ix0, iy0, ix1, iy1 = inner_plot
    px0, py0, px1, py1 = plot_bbox
    plot_w = max(1, ix1 - ix0)
    plot_h = max(1, iy1 - iy0)

    y_tick_w = max(18, int(plot_w * 0.045))
    y_title_w = max(28, int(plot_w * 0.08))
    x_tick_h = max(18, int(plot_h * 0.06))
    x_title_h = max(22, int(plot_h * 0.08))

    y_tick = (
        max(px0, ix0 - y_tick_w),
        iy0,
        ix0,
        iy1,
    )
    y_title = (
        max(px0, ix0 - y_tick_w - y_title_w),
        iy0,
        max(px0, ix0 - y_tick_w),
        iy1,
    )
    x_tick = (
        ix0,
        iy1,
        ix1,
        min(py1 if caption_bbox is None else caption_bbox[1], iy1 + x_tick_h),
    )
    x_title = (
        ix0,
        min(image_height, x_tick[3]),
        ix1,
        min(
            py1 if caption_bbox is None else caption_bbox[1],
            x_tick[3] + x_title_h,
        ),
    )
    # Slightly inset inner plot so axis strokes themselves are not treated as
    # removable text.
    inset = max(2, int(min(plot_w, plot_h) * 0.008))
    inner = (ix0 + inset, iy0 + inset, ix1 - inset, iy1 - inset)
    return {
        "inner_plot": inner,
        "y_tick": y_tick,
        "y_title": y_title,
        "x_tick": x_tick,
        "x_title": x_title,
    }


# ---------------------------------------------------------------------------
# Character-like components + grouping
# ---------------------------------------------------------------------------


def _long_line_mask(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 15), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 15)))
    return cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk),
    )


def _component_stroke_width(comp_mask: np.ndarray) -> float:
    if comp_mask.size == 0 or not np.any(comp_mask):
        return 0.0
    dist = cv2.distanceTransform(comp_mask.astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[comp_mask > 0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals) * 2.0)


def _is_curve_like_component(
    *,
    width: int,
    height: int,
    area: int,
    image_width: int,
    image_height: int,
    span_x: int | None = None,
    span_y: int | None = None,
) -> bool:
    """Reject long thin strokes that are part of plotted curves / axes."""
    aspect = width / max(height, 1)
    if width >= image_width * MAX_CURVE_LIKE_SPAN_FRAC and height <= max(6, image_height * 0.015):
        return True
    if height >= image_height * MAX_CURVE_LIKE_SPAN_FRAC and width <= max(6, image_width * 0.015):
        return True
    if aspect >= 10 and height <= 8:
        return True
    if aspect <= 0.1 and width <= 8:
        return True
    fill = area / max(1, width * height)
    if fill < 0.15 and max(width, height) > max(40, image_width * 0.05):
        return True
    if span_x is not None and span_x >= image_width * 0.25 and height <= 10:
        return True
    if span_y is not None and span_y >= image_height * 0.25 and width <= 10:
        return True
    return False


def extract_char_like_components(
    image_bgr: np.ndarray,
    *,
    origin_xy: tuple[int, int] = (0, 0),
    source: str = "cc",
) -> list[dict[str, Any]]:
    """
    Extract compact character-like connected components as text candidates.

    Does not require OCR. Long axis/curve strokes are filtered out.
    """
    if image_bgr.size == 0:
        return []
    gray = (
        image_bgr
        if image_bgr.ndim == 2
        else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    )
    h, w = gray.shape
    ox, oy = origin_xy
    binary = (gray < 145).astype(np.uint8) * 255
    lines = _long_line_mask(binary)
    ink = cv2.bitwise_and(binary, cv2.bitwise_not(lines))
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    image_area = float(max(1, h * w))
    out: list[dict[str, Any]] = []
    for i in range(1, num):
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        if area < MIN_COMP_AREA:
            continue
        if area / image_area > MAX_COMP_AREA_FRAC:
            continue
        if ch > h * MAX_COMP_HEIGHT_FRAC or cw > w * MAX_COMP_WIDTH_FRAC:
            continue
        aspect = cw / max(ch, 1)
        if aspect > MAX_COMP_ASPECT or aspect < MIN_COMP_ASPECT:
            continue
        if _is_curve_like_component(
            width=cw,
            height=ch,
            area=area,
            image_width=w,
            image_height=h,
        ):
            continue
        comp = ink[y : y + ch, x : x + cw] > 0
        stroke = _component_stroke_width(comp.astype(np.uint8) * 255)
        # Characters usually have modest stroke width vs size.
        if stroke > max(cw, ch) * 0.85 and min(cw, ch) > 12:
            continue
        fill = area / max(1.0, cw * ch)
        # Detection confidence from compactness / glyph-like geometry.
        det_conf = 55.0
        if 0.2 <= fill <= 0.85:
            det_conf += 15.0
        if 0.35 <= aspect <= 2.8:
            det_conf += 15.0
        if 1.5 <= stroke <= max(cw, ch) * 0.55:
            det_conf += 10.0
        box = [float(ox + x), float(oy + y), float(ox + x + cw), float(oy + y + ch)]
        out.append(
            {
                "bbox": box,
                "text": None,
                "recognized_text": None,
                "detection_confidence": float(min(98.0, det_conf)),
                "ocr_confidence": None,
                "confidence": float(min(98.0, det_conf)),
                "source": source,
                "area": area,
                "stroke_width": stroke,
                "aspect_ratio": aspect,
                "orientation": "upright",
                "n_components": 1,
            }
        )
    return out


def _merge_box_list(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        float(min(b[0] for b in boxes)),
        float(min(b[1] for b in boxes)),
        float(max(b[2] for b in boxes)),
        float(max(b[3] for b in boxes)),
    ]


def group_text_components(
    components: Sequence[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """
    Group nearby character-like components into horizontal or vertical labels.

    Groups are accepted as text candidates even when OCR fails.
    """
    if not components:
        return []

    x_tol = max(4.0, image_width * VERT_X_TOL_FRAC)
    y_tol = max(4.0, image_height * HORZ_Y_TOL_FRAC)
    gap_tol = max(4.0, min(image_width, image_height) * MAX_GAP_FRAC)

    def _height(i: int) -> float:
        b = components[i]["bbox"]
        return max(1.0, float(b[3] - b[1]))

    groups: list[dict[str, Any]] = []
    used: set[int] = set()

    # Vertical groups (peak annotations): similar x, stacked in y.
    by_x = sorted(range(len(components)), key=lambda i: _center(components[i]["bbox"])[0])
    for seed in by_x:
        if seed in used:
            continue
        members = [seed]
        sx, _ = _center(components[seed]["bbox"])
        sh = _height(seed)
        for idx in by_x:
            if idx in used or idx == seed:
                continue
            _bx0, by0, _bx1, by1 = components[idx]["bbox"]
            cx, _cy = _center(components[idx]["bbox"])
            if abs(cx - sx) > x_tol:
                continue
            if abs(_height(idx) - sh) > max(sh, _height(idx)) * 0.9:
                continue
            stack_y0 = min(components[m]["bbox"][1] for m in members)
            stack_y1 = max(components[m]["bbox"][3] for m in members)
            if by0 > stack_y1 + gap_tol or by1 < stack_y0 - gap_tol:
                continue
            members.append(idx)
            sx = float(np.mean([_center(components[m]["bbox"])[0] for m in members]))
            sh = float(np.mean([_height(m) for m in members]))
        if len(members) >= 2:
            boxes = [components[m]["bbox"] for m in members]
            union = _merge_box_list(boxes)
            if _box_height(union) >= _box_width(union) * 1.2:
                for m in members:
                    used.add(m)
                det_conf = float(
                    np.mean(
                        [
                            float(components[m].get("detection_confidence") or 50)
                            for m in members
                        ]
                    )
                )
                det_conf = min(99.0, det_conf + 10.0 * min(4, len(members) - 1))
                groups.append(
                    {
                        "bbox": union,
                        "text": None,
                        "recognized_text": None,
                        "detection_confidence": det_conf,
                        "ocr_confidence": None,
                        "confidence": det_conf,
                        "source": "vertical_group",
                        "orientation": "vertical",
                        "n_components": len(members),
                        "member_boxes": boxes,
                        "alignment": "vertical",
                    }
                )

    # Horizontal groups from remaining components.
    remaining = [i for i in range(len(components)) if i not in used]
    by_y = sorted(remaining, key=lambda i: _center(components[i]["bbox"])[1])
    used_h: set[int] = set()
    for seed in by_y:
        if seed in used_h:
            continue
        members = [seed]
        _sx, sy = _center(components[seed]["bbox"])
        sh = _height(seed)
        for idx in by_y:
            if idx in used_h or idx == seed:
                continue
            bx0, _by0, bx1, _by1 = components[idx]["bbox"]
            _cx, cy = _center(components[idx]["bbox"])
            if abs(cy - sy) > y_tol:
                continue
            if abs(_height(idx) - sh) > max(sh, _height(idx)) * 0.9:
                continue
            stack_x0 = min(components[m]["bbox"][0] for m in members)
            stack_x1 = max(components[m]["bbox"][2] for m in members)
            if bx0 > stack_x1 + gap_tol or bx1 < stack_x0 - gap_tol:
                continue
            members.append(idx)
            sy = float(np.mean([_center(components[m]["bbox"])[1] for m in members]))
            sh = float(np.mean([_height(m) for m in members]))
        if len(members) >= 2:
            for m in members:
                used_h.add(m)
                used.add(m)
            boxes = [components[m]["bbox"] for m in members]
            union = _merge_box_list(boxes)
            det_conf = float(
                np.mean(
                    [float(components[m].get("detection_confidence") or 50) for m in members]
                )
            )
            det_conf = min(99.0, det_conf + 6.0 * min(4, len(members) - 1))
            groups.append(
                {
                    "bbox": union,
                    "text": None,
                    "recognized_text": None,
                    "detection_confidence": det_conf,
                    "ocr_confidence": None,
                    "confidence": det_conf,
                    "source": "horizontal_group",
                    "orientation": "horizontal",
                    "n_components": len(members),
                    "member_boxes": boxes,
                    "alignment": "horizontal",
                }
            )

    # Singletons that look text-like enough on their own.
    for i in range(len(components)):
        if i in used:
            continue
        det = dict(components[i])
        det["n_components"] = 1
        det["member_boxes"] = [det["bbox"]]
        det["alignment"] = "singleton"
        groups.append(det)

    return groups


# ---------------------------------------------------------------------------
# Orientation mapping + optional OCR
# ---------------------------------------------------------------------------


def _map_box_from_rotate90_cw(
    box: Sequence[float],
    *,
    region_width: int,
    region_height: int,
    origin_xy: tuple[int, int] = (0, 0),
) -> list[float]:
    """Map a box from ROTATE_90_CLOCKWISE image coords back to original."""
    # rotated size: (region_width, region_height) -> (region_height, region_width)
    # old_x = new_y; old_y = region_height - new_x  (using region_height of original)
    nx0, ny0, nx1, ny1 = box
    corners = [(nx0, ny0), (nx1, ny0), (nx1, ny1), (nx0, ny1)]
    mapped = []
    for nx, ny in corners:
        ox = float(ny)
        oy = float(region_height - nx)
        mapped.append((ox, oy))
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    ox, oy = origin_xy
    return [min(xs) + ox, min(ys) + oy, max(xs) + ox, max(ys) + oy]


def _map_box_from_rotate90_ccw(
    box: Sequence[float],
    *,
    region_width: int,
    region_height: int,
    origin_xy: tuple[int, int] = (0, 0),
) -> list[float]:
    """Map a box from ROTATE_90_COUNTERCLOCKWISE image coords back to original."""
    # old_x = region_width - new_y; old_y = new_x
    nx0, ny0, nx1, ny1 = box
    corners = [(nx0, ny0), (nx1, ny0), (nx1, ny1), (nx0, ny1)]
    mapped = []
    for nx, ny in corners:
        ox = float(region_width - ny)
        oy = float(nx)
        mapped.append((ox, oy))
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    ox, oy = origin_xy
    return [min(xs) + ox, min(ys) + oy, max(xs) + ox, max(ys) + oy]


def _optional_ocr_crop(
    image_bgr: np.ndarray,
    box: Sequence[float],
    *,
    config: str = "--psm 7",
    whitelist: str | None = None,
) -> tuple[str | None, float | None]:
    try:
        import pytesseract
    except Exception:
        return None, None
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image_bgr.shape[1], x1)
    y1 = min(image_bgr.shape[0], y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    # Upscale tiny crops.
    scale = 3 if max(crop.shape[:2]) < 40 else 2
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    cfg = config
    if whitelist:
        cfg = f"{config} -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT,
            config=cfg,
        )
    except Exception:
        return None, None
    texts = []
    confs = []
    for i, raw in enumerate(data.get("text", [])):
        t = str(raw or "").strip()
        if not t:
            continue
        try:
            c = float(data["conf"][i])
        except (TypeError, ValueError, KeyError):
            c = -1.0
        texts.append(t)
        if c >= 0:
            confs.append(c)
    if not texts:
        return None, None
    text = "".join(texts) if all(len(t) <= 2 for t in texts) else " ".join(texts)
    conf = float(np.mean(confs)) if confs else None
    return text, conf


def attach_optional_ocr(
    image_bgr: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    *,
    in_plot: bool,
) -> list[dict[str, Any]]:
    """Attach recognized_text when OCR works; never reject on OCR failure."""
    out: list[dict[str, Any]] = []
    for det in candidates:
        det = dict(det)
        orientation = str(det.get("orientation") or "horizontal")
        box = det["bbox"]
        # For vertical groups, OCR on a 90° CW rotation of the crop.
        text = None
        conf = None
        if orientation == "vertical":
            x0, y0, x1, y1 = [int(v) for v in box]
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(image_bgr.shape[1], x1)
            y1 = min(image_bgr.shape[0], y1)
            crop = image_bgr[y0:y1, x0:x1]
            if crop.size:
                rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                # OCR on rotated crop directly (local coords).
                text, conf = _optional_ocr_crop(
                    rot,
                    [0, 0, rot.shape[1], rot.shape[0]],
                    config="--psm 7",
                    whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789().-",
                )
        else:
            text, conf = _optional_ocr_crop(
                image_bgr,
                box,
                config="--psm 7" if (det.get("n_components") or 1) <= 6 else "--psm 6",
            )
        if text:
            det["recognized_text"] = text
            det["text"] = text
            det["ocr_confidence"] = conf
            # Soft boost only.
            if conf is not None and conf >= (
                IN_PLOT_OCR_SUPPORT_MIN_CONF if in_plot else OCR_SUPPORT_MIN_CONF
            ):
                det["detection_confidence"] = float(
                    min(99.0, float(det.get("detection_confidence") or 50) + 5.0)
                )
                det["confidence"] = det["detection_confidence"]
        out.append(det)
    return out


def _collect_rotated_region_candidates(
    image_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    *,
    rotation: str,
    source: str,
) -> list[dict[str, Any]]:
    """Detect components in a rotated crop and map boxes back."""
    x0, y0, x1, y1 = region
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    rh, rw = crop.shape[:2]
    if rotation == "cw":
        rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        mapper = _map_box_from_rotate90_cw
    else:
        rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mapper = _map_box_from_rotate90_ccw

    comps = extract_char_like_components(rotated, origin_xy=(0, 0), source=source)
    # Group in rotated space then map.
    groups = group_text_components(
        comps,
        image_width=rotated.shape[1],
        image_height=rotated.shape[0],
    )
    mapped: list[dict[str, Any]] = []
    for g in groups:
        box = mapper(
            g["bbox"],
            region_width=rw,
            region_height=rh,
            origin_xy=(x0, y0),
        )
        members = [
            mapper(mb, region_width=rw, region_height=rh, origin_xy=(x0, y0))
            for mb in (g.get("member_boxes") or [g["bbox"]])
        ]
        det = dict(g)
        det["bbox"] = box
        det["member_boxes"] = members
        det["source"] = source
        det["orientation"] = "vertical" if rotation in {"cw", "ccw"} else det.get("orientation")
        # After mapping, vertical labels from CCW/CW become upright in original;
        # keep orientation tag based on mapped aspect.
        if _box_height(box) >= _box_width(box) * 1.3:
            det["orientation"] = "vertical"
            det["alignment"] = "vertical"
        mapped.append(det)
    return mapped


# ---------------------------------------------------------------------------
# Roles, glyph mask, validation helpers
# ---------------------------------------------------------------------------


def assign_spatial_role(
    det: dict[str, Any],
    *,
    bands: dict[str, tuple[int, int, int, int]],
    caption_bbox: tuple[int, int, int, int] | None,
) -> str:
    box = det.get("bbox") or [0, 0, 0, 0]
    cx, cy = _center(box)
    text = str(det.get("recognized_text") or det.get("text") or "")
    orientation = str(det.get("orientation") or "")
    n_comp = int(det.get("n_components") or 1)
    tall = _box_height(box) >= _box_width(box) * 2.5

    if caption_bbox is not None and cy >= caption_bbox[1]:
        return "caption_or_external_text"

    # Tall vertical titles in the left margin are axis titles even if the
    # geometric center drifts into the tick band.
    if (_point_in_box(cx, cy, bands["y_title"]) or _point_in_box(cx, cy, bands["y_tick"])) and (
        tall or orientation == "vertical"
    ):
        if "intensity" in text.lower() or n_comp >= 4 or tall:
            return "preserved_axis_title"

    if _point_in_box(cx, cy, bands["y_title"]):
        return "preserved_axis_title"
    if _point_in_box(cx, cy, bands["y_tick"]):
        return "preserved_axis_tick"
    if _point_in_box(cx, cy, bands["x_title"]):
        if any(ch.isdigit() for ch in text) and len(text) <= 4:
            return "preserved_axis_tick"
        return "preserved_axis_title"
    if _point_in_box(cx, cy, bands["x_tick"]):
        return "preserved_axis_tick"

    if _point_in_box(cx, cy, bands["inner_plot"]):
        if orientation == "vertical" or (
            n_comp >= 3 and _box_height(box) >= _box_width(box) * 1.5
        ):
            return "removable_peak_annotation"
        if text and any(ch.isdigit() for ch in text) and "(" in text:
            return "removable_peak_annotation"
        ix0, _iy0, ix1, _iy1 = bands["inner_plot"]
        if cx >= ix0 + 0.72 * (ix1 - ix0):
            return "removable_curve_label"
        return "removable_in_plot_text"

    return "uncertain_text_candidate"


def build_glyph_mask(
    image_bgr: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    padding_px: int = 1,
) -> np.ndarray:
    """
    Mask only ink pixels that look like text glyphs inside accepted regions.

    Avoids filling large rectangles that would erase overlapping peak curves.
    """
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if not detections:
        return mask
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binary = (gray < 150).astype(np.uint8) * 255
    lines = _long_line_mask(binary)

    for det in detections:
        box = det.get("bbox")
        if not box:
            continue
        members = det.get("member_boxes") or [box]
        for mb in members:
            x0 = max(0, int(mb[0]) - padding_px)
            y0 = max(0, int(mb[1]) - padding_px)
            x1 = min(width, int(mb[2]) + padding_px)
            y1 = min(height, int(mb[3]) + padding_px)
            if x1 <= x0 or y1 <= y0:
                continue
            region = binary[y0:y1, x0:x1]
            line_region = lines[y0:y1, x0:x1]
            ink = cv2.bitwise_and(region, cv2.bitwise_not(line_region))
            num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            rh, rw = ink.shape
            for i in range(1, num):
                x, y, cw, ch, area = [int(v) for v in stats[i]]
                if area < 4:
                    continue
                if _is_curve_like_component(
                    width=cw,
                    height=ch,
                    area=area,
                    image_width=max(rw, width),
                    image_height=max(rh, height),
                ):
                    continue
                # Skip components that nearly fill a wide region (curve segments).
                if cw >= rw * 0.85 and ch <= max(4, rh * 0.25):
                    continue
                mask[y0 + y : y0 + y + ch, x0 + x : x0 + x + cw][
                    labels[y : y + ch, x : x + cw] == i
                ] = 255
    # Tiny dilate for soft-penalty usability without large boxes.
    if np.any(mask):
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    return mask


def render_detection_debug_image(
    image_bgr: np.ndarray,
    detections: Sequence[dict[str, Any]],
    *,
    title: str = "",
    show_reject_reason: bool = False,
) -> np.ndarray:
    canvas = image_bgr.copy()
    for det in detections:
        box = det.get("bbox")
        if not box or len(box) < 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        role = str(det.get("role") or "")
        if show_reject_reason:
            color = (0, 0, 220)
        elif role.startswith("preserved_"):
            color = (220, 140, 0)
        elif role.startswith("removable_"):
            color = (0, 180, 0)
        elif role == "caption_or_external_text":
            color = (0, 180, 180)
        else:
            color = (160, 160, 0)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        text = str(det.get("recognized_text") or det.get("text") or "")
        conf = det.get("detection_confidence", det.get("confidence"))
        label_parts = [str(det.get("source") or ""), role]
        if text:
            label_parts.append(text[:24])
        if conf is not None:
            try:
                label_parts.append(f"{float(conf):.0f}")
            except (TypeError, ValueError):
                pass
        reason = str(det.get("reject_reason") or "")
        if show_reject_reason and reason:
            label_parts.append(reason)
        label = " | ".join(p for p in label_parts if p)
        cv2.putText(
            canvas,
            label[:120],
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    if title:
        cv2.putText(
            canvas,
            title,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _normalize_label_token(text: str) -> str:
    import re

    t = str(text or "").strip().upper().replace(" ", "")
    # Common OCR confusions in crystallographic labels.
    t = (
        t.replace("O", "0")
        .replace("Q", "0")
        .replace("—", "-")
        .replace("_", "")
    )
    t = "".join(ch for ch in t if ch.isalnum() or ch in "()")
    # Prefer canonical peak form when a 2–3 digit index is present.
    match = re.search(r"\(?([0-9]{2,3})\)?", t)
    if match and ("(" in str(text) or ")" in str(text) or match.group(1)):
        digits = match.group(1)
        if len(digits) in {2, 3}:
            return f"({digits})"
    return t


def _missed_expected_labels(
    accepted: Sequence[dict[str, Any]],
    expected: Sequence[str] | None,
) -> list[str]:
    """Diagnostics only — never used to bias detection."""
    if not expected:
        return []
    found_raw: set[str] = set()
    found_norm: set[str] = set()
    for det in accepted:
        for key in ("recognized_text", "text"):
            text = str(det.get(key) or "").strip()
            if not text:
                continue
            found_raw.add(text)
            found_raw.add(text.upper())
            found_raw.add(text.lower())
            compact = text.replace(" ", "")
            found_raw.add(compact)
            found_raw.add(compact.upper())
            found_norm.add(_normalize_label_token(text))
    missed: list[str] = []
    for label in expected:
        target = str(label).strip()
        if not target:
            continue
        compact = target.replace(" ", "")
        norm = _normalize_label_token(target)
        if (
            target in found_raw
            or target.upper() in found_raw
            or compact in found_raw
            or compact.upper() in found_raw
            or norm in found_norm
        ):
            continue
        # Fuzzy contains after normalization.
        if any(norm and norm in token for token in found_norm):
            continue
        if any(
            norm
            and norm in _normalize_label_token(str(d.get("recognized_text") or d.get("text") or ""))
            for d in accepted
        ):
            continue
        # Degree / intensity titles.
        if "deg" in target.lower() or "θ" in target or "theta" in target.lower():
            if any("deg" in str(d.get("text") or d.get("recognized_text") or "").lower() for d in accepted):
                continue
        if "intensity" in target.lower():
            if any(
                "intensity" in str(d.get("text") or d.get("recognized_text") or "").lower()
                for d in accepted
            ):
                continue
        missed.append(target)
    return missed


def validate_mask_for_combine(
    mask: np.ndarray,
    *,
    max_coverage: float,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[bool, float, str | None]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if region is not None:
        x0, y0, x1, y1 = region
        view = mask[y0:y1, x0:x1]
    else:
        view = mask
    coverage = mask_coverage_fraction(view)
    if coverage > max_coverage:
        return False, coverage, f"coverage_{coverage:.3f}_exceeds_{max_coverage:.3f}"
    return True, coverage, None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def detect_raster_text_regions(
    image_bgr: np.ndarray,
    *,
    use_tesseract: bool = True,
    use_mser_proposals: bool = True,
    min_confidence: float = 35.0,
    max_mask_coverage: float = MAX_REMOVABLE_PLOT_COVERAGE,
    pdf_text_boxes_px: Sequence[Sequence[float]] | None = None,
    padding_px: int = 1,
    expected_labels: Sequence[str] | None = None,
    remove_legend_graphics: bool = False,
) -> RasterTextDetectionResult:
    """
    Detect text-like regions for masking (not for accurate transcription).

    Acceptance is driven by character-like geometry, grouping, and spatial
    role. OCR may attach ``recognized_text`` but is not required.

    The returned ``mask`` is the removable in-plot **glyph** mask.
    """
    del remove_legend_graphics  # reserved; text-only by default

    result = RasterTextDetectionResult()
    if image_bgr is None or image_bgr.size == 0:
        result.failed = True
        result.failure_reason = "empty_image"
        result.mask = np.zeros((1, 1), dtype=np.uint8)
        return result

    height, width = image_bgr.shape[:2]
    empty = np.zeros((height, width), dtype=np.uint8)
    result.mask = empty.copy()
    result.all_text_candidate_mask = empty.copy()
    result.removable_region_mask = empty.copy()
    result.removable_glyph_mask = empty.copy()
    result.preserved_axis_mask = empty.copy()

    # Delayed import: caller module owns caption/panel helpers.
    import raster_text_detection as _rtd  # noqa: WPS433

    plot_bbox, caption_bbox = _rtd.split_figure_caption_region(
        image_bgr,
        pdf_text_boxes_px=pdf_text_boxes_px,
    )
    result.plot_bbox = plot_bbox
    result.caption_bbox = caption_bbox
    inner = detect_inner_plot_frame(image_bgr, plot_bbox=plot_bbox)
    result.inner_plot_bbox = inner
    bands = define_axis_bands(
        image_width=width,
        image_height=height,
        inner_plot=inner,
        plot_bbox=plot_bbox,
        caption_bbox=caption_bbox,
    )
    result.axis_bands = bands
    panels = _rtd.detect_chart_panels(image_bgr, plot_bbox=plot_bbox)
    result.panels = panels
    result.notes.extend(
        [
            "Geometry-first text masking: OCR is optional supporting evidence.",
            "Removal mask includes only roles beginning with removable_.",
            "Axis-band text is preserved regardless of OCR confidence.",
            "Default digitization mask is the removable glyph mask.",
        ]
    )

    # 1) Upright character-like components over the full figure.
    comps = extract_char_like_components(image_bgr, source="cc")
    if use_mser_proposals:
        # MSER proposals as additional components (geometry only).
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            mser = cv2.MSER_create()
            mser.setMinArea(max(12, int((height * width) // 80000)))
            mser.setMaxArea(max(200, int(height * width * 0.004)))
            regions, _ = mser.detectRegions(gray)
            mser_boxes = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                if w < 3 or h < 3:
                    continue
                mser_boxes.append((x, y, w, h))
            mser_boxes, dup_removed = _nms_xywh_proposals(mser_boxes)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("MSER failed: %s", exc)
            mser_boxes, dup_removed = [], 0
    else:
        mser_boxes, dup_removed = [], 0

    for x, y, w, h in mser_boxes:
        aspect = w / max(h, 1)
        if aspect > MAX_COMP_ASPECT or aspect < MIN_COMP_ASPECT:
            continue
        if _is_curve_like_component(
            width=w, height=h, area=w * h, image_width=width, image_height=height
        ):
            continue
        comps.append(
            {
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "text": None,
                "recognized_text": None,
                "detection_confidence": 50.0,
                "ocr_confidence": None,
                "confidence": 50.0,
                "source": "mser_component",
                "area": int(w * h),
                "stroke_width": 0.0,
                "aspect_ratio": aspect,
                "orientation": "upright",
                "n_components": 1,
            }
        )

    # Deduplicate raw components before grouping.
    comps = _dedupe_detections(comps, iou_threshold=0.7)

    groups = group_text_components(comps, image_width=width, image_height=height)

    # 2) Rotated passes over the inner plot (vertical peak labels) and left axis.
    ix0, iy0, ix1, iy1 = bands["inner_plot"]
    rotated_groups: list[dict[str, Any]] = []
    for rotation, source in (
        ("ccw", "rotated_ccw"),
        ("cw", "rotated_cw"),
    ):
        rotated_groups.extend(
            _collect_rotated_region_candidates(
                image_bgr,
                (ix0, iy0, ix1, iy1),
                rotation=rotation,
                source=source,
            )
        )
        # Left margin for vertical axis titles.
        yx0, yy0, yx1, yy1 = bands["y_title"]
        if yx1 > yx0 and yy1 > yy0:
            rotated_groups.extend(
                _collect_rotated_region_candidates(
                    image_bgr,
                    (yx0, yy0, max(yx1, ix0), yy1),
                    rotation=rotation,
                    source=f"{source}_axis",
                )
            )

    all_candidates = groups + rotated_groups
    all_candidates = _dedupe_detections(all_candidates, iou_threshold=0.55)

    # Optional OCR attachment (never a hard gate).
    if use_tesseract:
        in_plot_cands = []
        other_cands = []
        for det in all_candidates:
            cx, cy = _center(det["bbox"])
            if _point_in_box(cx, cy, bands["inner_plot"]):
                in_plot_cands.append(det)
            else:
                other_cands.append(det)
        all_candidates = attach_optional_ocr(
            image_bgr, in_plot_cands, in_plot=True
        ) + attach_optional_ocr(image_bgr, other_cands, in_plot=False)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for det in all_candidates:
        box = _clamp_box(det["bbox"], width, height)
        det = {**det, "bbox": box}
        role = assign_spatial_role(det, bands=bands, caption_bbox=caption_bbox)
        det["role"] = role
        det_conf = float(det.get("detection_confidence") or 0.0)

        # Size sanity: never accept plot-sized boxes.
        area_frac = _box_area(box) / float(max(1, width * height))
        if area_frac > 0.08:
            rejected.append({**det, "reject_reason": "box_too_large"})
            continue
        if _box_width(box) > width * 0.55 and _box_height(box) < height * 0.05:
            rejected.append({**det, "reject_reason": "long_thin_stroke"})
            continue

        # Inside plot: lower geometric threshold; allow single chars / punctuation.
        if role.startswith(REMOVABLE_ROLE_PREFIX):
            n_comp = int(det.get("n_components") or 1)
            # Prefer grouped text; allow singletons only with decent confidence
            # or OCR support (legend letters, split glyphs).
            if n_comp >= 2:
                accepted.append(det)
            elif det.get("recognized_text") and det_conf >= 25.0:
                accepted.append(det)
            elif det_conf >= max(40.0, min_confidence):
                accepted.append(det)
            else:
                rejected.append({**det, "reject_reason": "weak_singleton_in_plot"})
            continue

        if role in PRESERVED_ROLES:
            # Preserve axis text with weaker thresholds — location matters.
            if det_conf >= 20.0 or int(det.get("n_components") or 1) >= 2:
                accepted.append(det)
            else:
                # Still keep as accepted preserved if clearly in axis band.
                accepted.append(det)
            continue

        if role == "uncertain_text_candidate":
            if det_conf >= min_confidence + 15:
                accepted.append(det)
            else:
                rejected.append({**det, "reject_reason": "uncertain_low_confidence"})
            continue

        rejected.append({**det, "reject_reason": "unhandled_role"})

    accepted = _dedupe_detections(accepted, iou_threshold=0.5)
    rejected_unique = _dedupe_detections(rejected, iou_threshold=0.7)

    by_role: dict[str, list[dict[str, Any]]] = {}
    for det in accepted:
        by_role.setdefault(str(det.get("role") or "uncertain_text_candidate"), []).append(det)

    removable = [d for d in accepted if str(d.get("role") or "").startswith(REMOVABLE_ROLE_PREFIX)]
    preserved = [
        d
        for d in accepted
        if str(d.get("role") or "") in {"preserved_axis_tick", "preserved_axis_title"}
    ]
    all_for_candidate_mask = accepted + [
        d for d in rejected_unique if float(d.get("detection_confidence") or 0) >= 40
    ]

    all_text_candidate_mask = build_box_mask_array(
        [d["bbox"] for d in all_for_candidate_mask],
        width=width,
        height=height,
        padding_px=padding_px,
    )
    removable_region_mask = build_box_mask_array(
        [d["bbox"] for d in removable],
        width=width,
        height=height,
        padding_px=padding_px,
    )
    preserved_axis_mask = build_box_mask_array(
        [d["bbox"] for d in preserved],
        width=width,
        height=height,
        padding_px=padding_px,
    )
    removable_glyph_mask = build_glyph_mask(image_bgr, removable, padding_px=padding_px)

    # Coverage diagnostics on inner plot.
    ipx0, ipy0, ipx1, ipy1 = bands["inner_plot"]
    plot_view = removable_glyph_mask[ipy0:ipy1, ipx0:ipx1]
    coverage = mask_coverage_fraction(plot_view) if plot_view.size else 0.0

    # Axis-band overlap with removable mask must stay low.
    axis_overlap_scores = []
    for name in ("x_tick", "x_title", "y_tick", "y_title"):
        bx0, by0, bx1, by1 = bands[name]
        view = removable_glyph_mask[by0:by1, bx0:bx1]
        if view.size:
            axis_overlap_scores.append(mask_coverage_fraction(view))
    axis_overlap = float(max(axis_overlap_scores) if axis_overlap_scores else 0.0)

    # Preserved vs removable region overlap.
    overlap = np.logical_and(removable_region_mask > 0, preserved_axis_mask > 0)
    role_overlap_frac = float(overlap.mean()) if overlap.size else 0.0

    failed = False
    failure_reason = None
    if coverage > max_mask_coverage:
        failed = True
        failure_reason = (
            f"removable_plot_coverage_{coverage:.3f}_exceeds_{max_mask_coverage:.3f}"
        )
    elif axis_overlap > MAX_AXIS_BAND_REMOVABLE_OVERLAP:
        failed = True
        failure_reason = f"axis_band_removable_overlap_{axis_overlap:.3f}"
    elif role_overlap_frac > 0.02:
        failed = True
        failure_reason = f"preserved_removable_overlap_{role_overlap_frac:.3f}"

    if failed:
        result.notes.append(
            f"Mask validation failed ({failure_reason}); emitting empty removable masks."
        )
        removable_glyph_mask = empty.copy()
        removable_region_mask = empty.copy()
        coverage = 0.0

    n_vertical = sum(
        1
        for d in removable
        if str(d.get("orientation") or "") == "vertical"
        or str(d.get("alignment") or "") == "vertical"
        or str(d.get("source") or "").startswith("rotated_")
        or str(d.get("source") or "") == "vertical_group"
    )

    missed = _missed_expected_labels(accepted, expected_labels)
    diagnostics = {
        "accepted_count": len(accepted),
        "accepted_by_role": {k: len(v) for k, v in by_role.items()},
        "rejected_count": len(rejected),
        "rejected_unique_count": len(rejected_unique),
        "duplicate_proposals_removed": dup_removed,
        "mask_coverage": coverage,
        "removable_plot_coverage": coverage,
        "axis_band_removable_overlap": axis_overlap,
        "preserved_removable_overlap": role_overlap_frac,
        "in_plot_text_groups": len(removable),
        "vertical_groups": n_vertical,
        "preserved_axis_labels": len(preserved),
        "uncertain_candidates": len(by_role.get("uncertain_text_candidate") or []),
        "rotated_ocr_detections": sum(
            1 for d in accepted if str(d.get("source") or "").startswith("rotated_")
        ),
        "missed_expected_labels": missed,
    }

    result.accepted = accepted
    result.rejected = rejected_unique
    result.by_role = by_role
    result.mask = removable_glyph_mask
    result.all_text_candidate_mask = all_text_candidate_mask
    result.removable_region_mask = removable_region_mask
    result.removable_glyph_mask = removable_glyph_mask
    result.preserved_axis_mask = preserved_axis_mask
    result.coverage = coverage
    result.failed = failed
    result.failure_reason = failure_reason
    result.diagnostics = diagnostics
    result.debug_accepted_bgr = render_detection_debug_image(
        image_bgr,
        removable,
        title=(
            f"removable n={len(removable)} vert={n_vertical} "
            f"cov={coverage:.3f}"
        ),
    )
    result.debug_preserved_bgr = render_detection_debug_image(
        image_bgr,
        preserved,
        title=f"preserved axes n={len(preserved)}",
    )
    result.debug_rejected_bgr = render_detection_debug_image(
        image_bgr,
        _select_rejected_for_debug(rejected_unique),
        title=(
            f"rejected debug "
            f"{min(REJECT_DEBUG_MAX, len(rejected_unique))}/{len(rejected_unique)}"
        ),
        show_reject_reason=True,
    )
    return result
