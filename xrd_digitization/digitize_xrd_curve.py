from __future__ import annotations

import cv2
import numpy as np

from xrd_digitization.detect_panels import detect_stacked_curve_bands
from xrd_digitization.simplify_curve import simplify_single_curve
from xrd_digitization.types import AxisCalibrationResult, CurveData, PlotCropResult

CURVE_COLOR_TOLERANCE = 35
MIN_CURVE_COVERAGE_FRACTION = 0.04

COLOR_MASKS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "blue": (np.array([90, 50, 50]), np.array([130, 255, 255])),
    "cyan": (np.array([80, 40, 50]), np.array([100, 255, 255])),
    "red": (np.array([0, 50, 50]), np.array([12, 255, 255])),
    "orange": (np.array([10, 70, 70]), np.array([28, 255, 255])),
    "green": (np.array([35, 45, 45]), np.array([85, 255, 255])),
    "purple": (np.array([125, 40, 40]), np.array([155, 255, 255])),
    "magenta": (np.array([145, 40, 40]), np.array([175, 255, 255])),
}


def _blue_curve_mask(cropped_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(cropped_bgr)
    hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    channel = ((b > 45) & (b > g + 8) & (b > r + 8) & (g < 160) & (r < 160)).astype(np.uint8) * 255
    hsv_mask = cv2.inRange(hsv, *COLOR_MASKS["blue"])
    return cv2.bitwise_or(channel, hsv_mask)


def _red_curve_mask(cropped_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    lower2 = np.array([168, 50, 50])
    upper2 = np.array([179, 255, 255])
    hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, *COLOR_MASKS["red"]),
        cv2.inRange(hsv, lower2, upper2),
    )
    b, g, r = cv2.split(cropped_bgr)
    channel = ((r > 120) & (r > g + 25) & (r > b + 25) & (g < 140) & (b < 140)).astype(np.uint8) * 255
    return cv2.bitwise_or(hsv_mask, channel)


def _orange_curve_mask(cropped_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, *COLOR_MASKS["orange"])


def _green_curve_mask(cropped_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, *COLOR_MASKS["green"])


def _purple_curve_mask(cropped_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, *COLOR_MASKS["purple"])


def _magenta_curve_mask(cropped_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, *COLOR_MASKS["magenta"])


def _black_curve_mask(cropped_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = dark.shape
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, height // 25)))
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, width // 25), 1))
    v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vertical)
    h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal)
    axes = cv2.bitwise_or(v_lines, h_lines)
    curve = cv2.bitwise_and(dark, cv2.bitwise_not(axes))
    return cv2.morphologyEx(curve, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _detect_curve_hsv_ranges(cropped_bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    mask = (pixels[:, 1] > 25) & (pixels[:, 2] > 30) & (pixels[:, 2] < 250)
    colored = pixels[mask]
    if len(colored) == 0:
        return []

    h = int(np.median(colored[:, 0]))
    s = int(np.median(colored[:, 1]))
    v = int(np.median(colored[:, 2]))
    tol = CURVE_COLOR_TOLERANCE
    lower = np.array([max(0, h - tol), max(0, s - tol), max(0, v - tol)])
    upper = np.array([min(179, h + tol), min(255, s + tol), min(255, v + tol)])
    return [(lower, upper)]


def _mask_column_coverage(mask: np.ndarray) -> int:
    return int(np.sum(mask.sum(axis=0) > 0))


def _clean_curve_mask(
    mask: np.ndarray,
    *,
    plot_top: int | None = None,
    plot_bottom: int | None = None,
    plot_left: int | None = None,
    plot_right: int | None = None,
) -> np.ndarray:
    """Remove axis lines, legend swatches, and text speckle from a curve mask."""
    if mask.size == 0:
        return mask

    height, width = mask.shape
    cleaned = mask.copy()

    if None not in (plot_top, plot_bottom, plot_left, plot_right):
        label_margin = int((plot_bottom - plot_top) * 0.12)
        if label_margin > 0:
            cleaned[plot_top : plot_top + label_margin, plot_left:plot_right] = 0

    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 8), 1))
    h_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horizontal)
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(h_lines))

    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 2)))
    v_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vertical)
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(v_lines))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    kept = np.zeros_like(cleaned)
    min_area = max(8, (height * width) // 50000)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        if area < min_area:
            continue
        if h > w * 4 and w <= 3:
            continue
        if h > height * 0.55 and w < max(8, width * 0.04):
            continue
        kept[labels == label] = 255

    if plot_left is not None and plot_right is not None:
        min_span = max(20, int((plot_right - plot_left) * 0.4))
        if _mask_column_coverage(kept) < min_span:
            # CC filtering removed the trace; keep morphologically cleaned mask instead.
            return cleaned

    return kept


def _build_curve_masks(
    cropped_bgr: np.ndarray,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    band_top: int | None = None,
    band_bottom: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Return all plausible curve color masks inside the plot area."""
    band_top = plot_top if band_top is None else band_top
    band_bottom = plot_bottom if band_bottom is None else band_bottom
    roi = cropped_bgr[band_top:band_bottom, plot_left:plot_right]
    if roi.size == 0:
        return []

    plot_width = max(plot_right - plot_left, 1)
    min_coverage = max(20, int(plot_width * MIN_CURVE_COVERAGE_FRACTION))

    candidates: list[tuple[str, np.ndarray, int]] = []
    mask_fns = (
        ("blue", _blue_curve_mask),
        ("red", _red_curve_mask),
        ("orange", _orange_curve_mask),
        ("green", _green_curve_mask),
        ("purple", _purple_curve_mask),
        ("magenta", _magenta_curve_mask),
        ("black", _black_curve_mask),
    )
    for name, fn in mask_fns:
        full_mask = _clean_curve_mask(
            fn(cropped_bgr),
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            plot_left=plot_left,
            plot_right=plot_right,
        )
        sub = full_mask[band_top:band_bottom, plot_left:plot_right]
        coverage = _mask_column_coverage(sub)
        if coverage >= min_coverage:
            candidates.append((name, full_mask, coverage))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for lower, upper in _detect_curve_hsv_ranges(roi):
        auto = _clean_curve_mask(
            cv2.inRange(hsv, lower, upper),
            plot_top=band_top,
            plot_bottom=band_bottom,
            plot_left=plot_left,
            plot_right=plot_right,
        )
        padded = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)
        padded[band_top:band_bottom, plot_left:plot_right] = auto
        coverage = _mask_column_coverage(auto)
        if coverage >= min_coverage:
            candidates.append(("auto", padded, coverage))

    if not candidates:
        return []

    def _median_row(mask: np.ndarray) -> float:
        sub = mask[band_top:band_bottom, plot_left:plot_right]
        ys, _ = np.where(sub > 0)
        return float(np.median(ys)) if len(ys) else -1.0

    # Drop near-duplicate masks (same columns covered at similar vertical position).
    selected: list[tuple[str, np.ndarray]] = []
    for name, mask, coverage in sorted(candidates, key=lambda item: -item[2]):
        sub = mask[band_top:band_bottom, plot_left:plot_right]
        cols = sub.sum(axis=0) > 0
        med_y = _median_row(mask)
        duplicate = False
        for existing_name, existing in selected:
            existing_sub = existing[band_top:band_bottom, plot_left:plot_right]
            existing_cols = existing_sub.sum(axis=0) > 0
            overlap = np.logical_and(cols, existing_cols).sum()
            union = np.logical_or(cols, existing_cols).sum()
            existing_med_y = _median_row(existing)
            vertically_separated = (
                med_y >= 0
                and existing_med_y >= 0
                and abs(med_y - existing_med_y) >= max(12, (band_bottom - band_top) // 12)
            )
            if union > 0 and overlap / union > 0.75 and not vertically_separated:
                duplicate = True
                break
        if not duplicate:
            selected.append((name, mask))

    if selected:
        return selected

    # Last resort: best single mask.
    best_name, best_mask, _ = max(candidates, key=lambda item: item[2])
    return [(best_name, best_mask)]


def _extract_trace_from_mask(
    mask: np.ndarray,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    *,
    band_top: int | None = None,
    band_bottom: int | None = None,
    stacked: bool = False,
    lower_envelope: bool = False,
) -> np.ndarray:
    band_top = plot_top if band_top is None else band_top
    band_bottom = plot_bottom if band_bottom is None else band_bottom
    sub = mask[band_top:band_bottom, plot_left:plot_right]
    height, width = sub.shape
    if width == 0 or height == 0:
        return np.array([]).reshape(0, 2)

    label_margin = int(height * 0.12)
    scan_top = label_margin if label_margin < height else 0
    points: list[tuple[int, float]] = []
    prev_y: float | None = None

    for x in range(width):
        ys = np.where(sub[scan_top:, x] > 0)[0]
        if len(ys) == 0:
            ys = np.where(sub[:, x] > 0)[0]
            offset = 0
        else:
            offset = scan_top

        if len(ys) == 0:
            continue

        if lower_envelope:
            y = float(np.max(ys) + offset)
        elif len(ys) == 1:
            y = float(ys[0] + offset)
        else:
            y = float(np.median(ys) + offset)
            if prev_y is not None and abs(y - prev_y) > height * 0.35:
                y = float(ys[np.argmin(np.abs(ys + offset - prev_y))] + offset)

        if prev_y is not None and abs(y - prev_y) > height * 0.35:
            continue
        prev_y = y
        points.append((x, y))

    if not points:
        return np.array([]).reshape(0, 2)

    pts = np.array(points, dtype=float)
    if len(pts) >= 5 and not stacked:
        window = max(3, len(pts) // 120)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        pts[:, 1] = np.convolve(pts[:, 1], kernel, mode="same")

    if len(pts) >= 2:
        x0, x1 = pts[0, 0], pts[-1, 0]
        span_x = np.arange(int(x0), int(x1) + 1, dtype=float)
        span_y = np.interp(span_x, pts[:, 0], pts[:, 1])
        return np.column_stack((span_x, span_y))
    return pts


def _pixel_to_two_theta(
    x_pixels: np.ndarray,
    calibration: AxisCalibrationResult,
) -> np.ndarray:
    span = calibration.plot_right - calibration.plot_left
    if span <= 0:
        span = 1
    frac = (x_pixels - calibration.plot_left) / span
    return calibration.x_min + frac * (calibration.x_max - calibration.x_min)


def _pixel_y_to_intensity(
    y_pixels: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    band_top: int,
    band_bottom: int,
    stacked: bool = False,
) -> np.ndarray:
    if calibration.has_y_calibration and len(calibration.y_tick_pairs) >= 2:
        ys = np.array([p[0] for p in calibration.y_tick_pairs], dtype=float)
        vals = np.array([p[1] for p in calibration.y_tick_pairs], dtype=float)
        coeffs = np.polyfit(ys, vals, 1)
        intensity = coeffs[0] * y_pixels + coeffs[1]
        y_lo = min(calibration.y_min or 0.0, calibration.y_max or 0.0)
        y_hi = max(calibration.y_min or 0.0, calibration.y_max or 0.0)
        return np.clip(intensity, y_lo, y_hi)

    if stacked and len(y_pixels):
        baseline = float(np.percentile(y_pixels, 92))
        span = max(baseline - band_top, 1.0)
        return np.clip((baseline - y_pixels) / span, 0.0, None)

    span = band_bottom - band_top
    if span <= 0:
        span = 1
    frac = (band_bottom - y_pixels) / span
    return np.clip(frac, 0.0, None)


def _resample_uniform(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    num_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    if len(two_theta) < 2:
        return two_theta, intensity

    order = np.argsort(two_theta)
    x = two_theta[order]
    y = intensity[order]

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if len(x_unique) < 2:
        return x_unique, y_unique

    grid = np.linspace(x_unique.min(), x_unique.max(), num_points)
    y_grid = np.interp(grid, x_unique, y_unique)
    return grid, y_grid


def _normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    if len(intensity) == 0:
        return intensity
    peak = float(np.percentile(intensity, 99))
    if peak <= 0:
        peak = float(np.max(intensity))
    if peak <= 0:
        return intensity
    return np.clip(intensity / peak * 100.0, 0.0, 100.0)


def _is_valid_curve(intensity: np.ndarray) -> bool:
    if len(intensity) < 20:
        return False
    y = np.asarray(intensity, dtype=float)
    if float(np.max(y)) < 12.0:
        return False
    if float(np.mean(y > 4.0)) < 0.005:
        return False
    peaks = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 8.0)
    return bool(peaks.sum() >= 1)


def _dedupe_similar_curves(curves: list[CurveData]) -> list[CurveData]:
    if len(curves) <= 1:
        return curves

    kept: list[CurveData] = []
    for curve in curves:
        y = np.asarray(curve.intensity, dtype=float)
        duplicate = False
        for existing in kept:
            ey = np.asarray(existing.intensity, dtype=float)
            if len(y) != len(ey):
                continue
            corr = np.corrcoef(y, ey)[0, 1]
            if np.isfinite(corr) and corr > 0.92:
                duplicate = True
                break
        if not duplicate:
            kept.append(curve)
    return kept


def _masks_are_vertically_stacked(
    masks: list[tuple[str, np.ndarray]],
    plot_top: int,
    plot_bottom: int,
    plot_left: int,
    plot_right: int,
) -> bool:
    medians: list[float] = []
    span = max(plot_bottom - plot_top, 1)
    for _, mask in masks:
        ys, _ = np.where(mask[plot_top:plot_bottom, plot_left:plot_right] > 0)
        if len(ys) < 20:
            continue
        medians.append(float(np.median(ys)))
    if len(medians) < 2:
        return False
    medians.sort()
    cluster_tol = max(12.0, span / 15.0)
    clusters: list[float] = []
    for med in medians:
        if not clusters or med - clusters[-1] > cluster_tol:
            clusters.append(med)
        else:
            clusters[-1] = (clusters[-1] + med) / 2.0
    if len(clusters) < 2:
        return False
    gaps = [clusters[index + 1] - clusters[index] for index in range(len(clusters) - 1)]
    return bool(gaps and min(gaps) >= span / 8.0)


def _assign_masks_to_horizontal_bands(
    masks: list[tuple[str, np.ndarray]],
    plot_top: int,
    plot_bottom: int,
    plot_left: int,
    plot_right: int,
) -> list[tuple[str, np.ndarray, int, int]]:
    """Group color masks into horizontal bands by median row of each trace."""
    entries: list[tuple[str, np.ndarray, float]] = []
    for name, mask in masks:
        sub = mask[plot_top:plot_bottom, plot_left:plot_right]
        ys, xs = np.where(sub > 0)
        if len(ys) < 20:
            continue
        entries.append((name, mask, float(np.median(ys))))

    if len(entries) <= 1:
        return [
            (name, mask, plot_top, plot_bottom)
            for name, mask, _ in entries
        ]

    entries.sort(key=lambda item: item[2])
    groups: list[list[tuple[str, np.ndarray, float]]] = [[entries[0]]]
    for entry in entries[1:]:
        if entry[2] - groups[-1][-1][2] < max(12, (plot_bottom - plot_top) // 20):
            groups[-1].append(entry)
        else:
            groups.append([entry])

    band_height = (plot_bottom - plot_top) / max(len(groups), 1)
    assigned: list[tuple[str, np.ndarray, int, int]] = []
    for index, group in enumerate(groups):
        best_name, best_mask, _ = max(
            group,
            key=lambda item: item[1][plot_top:plot_bottom, plot_left:plot_right].sum(),
        )
        band_top = int(plot_top + index * band_height)
        band_bottom = int(plot_top + (index + 1) * band_height)
        if index == len(groups) - 1:
            band_bottom = plot_bottom
        assigned.append((best_name, best_mask, band_top, band_bottom))
    return assigned


def _mask_column_overlap_fraction(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    plot_top: int,
    plot_bottom: int,
    plot_left: int,
    plot_right: int,
) -> float:
    a = mask_a[plot_top:plot_bottom, plot_left:plot_right].sum(axis=0) > 0
    b = mask_b[plot_top:plot_bottom, plot_left:plot_right].sum(axis=0) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _filter_spurious_colored_masks(
    masks: list[tuple[str, np.ndarray]],
    calibration: AxisCalibrationResult,
) -> list[tuple[str, np.ndarray]]:
    """Drop colored masks that mostly duplicate a strong black trace."""
    black_masks = [(n, m) for n, m in masks if n == "black"]
    if not black_masks:
        return masks

    _, black = max(
        black_masks,
        key=lambda item: _mask_column_coverage(
            item[1][
                calibration.plot_top : calibration.plot_bottom,
                calibration.plot_left : calibration.plot_right,
            ]
        ),
    )
    black_cov = _mask_column_coverage(
        black[
            calibration.plot_top : calibration.plot_bottom,
            calibration.plot_left : calibration.plot_right,
        ]
    )
    plot_width = max(calibration.plot_right - calibration.plot_left, 1)
    if black_cov < max(20, int(plot_width * 0.12)):
        return masks

    kept: list[tuple[str, np.ndarray]] = []
    for name, mask in masks:
        if name == "black":
            kept.append((name, mask))
            continue
        overlap = _mask_column_overlap_fraction(
            mask,
            black,
            plot_top=calibration.plot_top,
            plot_bottom=calibration.plot_bottom,
            plot_left=calibration.plot_left,
            plot_right=calibration.plot_right,
        )
        if overlap < 0.35:
            kept.append((name, mask))
        elif name != "auto":
            colored_cov = _mask_column_coverage(
                mask[
                    calibration.plot_top : calibration.plot_bottom,
                    calibration.plot_left : calibration.plot_right,
                ]
            )
            if colored_cov >= int(black_cov * 0.85):
                kept.append((name, mask))
    if black_cov >= max(20, int(plot_width * MIN_CURVE_COVERAGE_FRACTION)):
        kept = [
            (name, mask)
            for name, mask in kept
            if name == "black"
            or _mask_column_coverage(
                mask[
                    calibration.plot_top : calibration.plot_bottom,
                    calibration.plot_left : calibration.plot_right,
                ]
            )
            >= int(black_cov * 0.85)
        ]
    return kept if kept else masks


def _pick_best_single_mask(
    masks: list[tuple[str, np.ndarray]],
    calibration: AxisCalibrationResult,
) -> tuple[str, np.ndarray] | None:
    if not masks:
        return None
    plot_sub = lambda m: m[
        calibration.plot_top : calibration.plot_bottom,
        calibration.plot_left : calibration.plot_right,
    ]
    plot_width = max(calibration.plot_right - calibration.plot_left, 1)

    def score(item: tuple[str, np.ndarray]) -> tuple[int, int]:
        name, mask = item
        coverage = _mask_column_coverage(plot_sub(mask))
        if name in ("blue", "red", "green", "orange", "cyan", "purple", "magenta"):
            return (3, coverage)
        if name == "black" and coverage >= max(20, int(plot_width * 0.15)):
            return (2, coverage)
        if name == "auto":
            return (1, coverage)
        black_is_axis = (
            name == "black"
            and coverage < max(20, int(plot_width * 0.15))
            and any(n != "black" for n, _ in masks)
        )
        if black_is_axis:
            return (-1, coverage)
        return (1, coverage)

    return max(masks, key=score)


def _colored_masks_co_located(
    colored: list[tuple[str, np.ndarray]],
    plot_top: int,
    plot_bottom: int,
    plot_left: int,
    plot_right: int,
) -> bool:
    """True when multiple color masks trace the same curve on a shared baseline."""
    medians: list[float] = []
    span = max(plot_bottom - plot_top, 1)
    for _, mask in colored:
        ys, _ = np.where(mask[plot_top:plot_bottom, plot_left:plot_right] > 0)
        if len(ys) >= 20:
            medians.append(float(np.median(ys)))
    if len(medians) < 2:
        return True
    cluster_tol = max(20, span // 10)
    return max(medians) - min(medians) <= cluster_tol


def _simplify_from_best_mask(
    cropped: np.ndarray,
    calibration: AxisCalibrationResult,
    masks: list[tuple[str, np.ndarray]],
    *,
    num_points: int,
) -> CurveData | None:
    picked = _pick_best_single_mask(masks, calibration)
    if not picked:
        return None
    color_name, mask = picked
    use_mask: np.ndarray | None = None
    if color_name == "auto":
        colored = [(n, m) for n, m in masks if n not in ("black", "auto")]
        if colored:
            color_name, mask = max(
                colored,
                key=lambda item: _mask_column_coverage(
                    item[1][
                        calibration.plot_top : calibration.plot_bottom,
                        calibration.plot_left : calibration.plot_right,
                    ]
                ),
            )
            use_mask = mask
    elif color_name not in ("black",):
        use_mask = mask
    simplified, peak_records = simplify_single_curve(
        cropped,
        calibration,
        band_top=calibration.plot_top,
        band_bottom=calibration.plot_bottom,
        num_points=num_points,
        curve_mask=use_mask,
    )
    if not simplified.two_theta:
        return None
    return CurveData(
        two_theta=simplified.two_theta,
        intensity=simplified.intensity,
        curve_id="curve_1",
        color=color_name if color_name not in ("auto", "black") else None,
        label=color_name if color_name not in ("auto", "black") else None,
        warnings=sorted(set(simplified.warnings)),
        detected_peaks=peak_records,
    )


def _classify_plot_layout(
    cropped: np.ndarray,
    calibration: AxisCalibrationResult,
    masks: list[tuple[str, np.ndarray]],
) -> str:
    masks = _filter_spurious_colored_masks(masks, calibration)
    plot_width = max(calibration.plot_right - calibration.plot_left, 1)
    min_coverage = max(20, int(plot_width * MIN_CURVE_COVERAGE_FRACTION))

    def coverage(mask: np.ndarray) -> int:
        return _mask_column_coverage(
            mask[
                calibration.plot_top : calibration.plot_bottom,
                calibration.plot_left : calibration.plot_right,
            ]
        )

    colored = [
        (name, mask)
        for name, mask in masks
        if name != "black" and coverage(mask) >= min_coverage
    ]
    black_masks = [(name, mask) for name, mask in masks if name == "black"]
    black_cov = max((coverage(m) for _, m in black_masks), default=0)

    if len(colored) == 0 and black_cov >= min_coverage:
        return "single"

    if len(colored) >= 2:
        if _colored_masks_co_located(
            colored,
            calibration.plot_top,
            calibration.plot_bottom,
            calibration.plot_left,
            calibration.plot_right,
        ):
            return "single"
        if _masks_are_vertically_stacked(
            colored,
            calibration.plot_top,
            calibration.plot_bottom,
            calibration.plot_left,
            calibration.plot_right,
        ):
            return "stacked"
        return "overlay"

    if len(colored) == 1:
        return "single"

    bands = detect_stacked_curve_bands(
        cropped,
        calibration.plot_top,
        calibration.plot_bottom,
        calibration.plot_left,
        calibration.plot_right,
    )
    if len(bands) > 1 and len(colored) >= 1:
        return "stacked"
    return "single"


def digitize_xrd_curves(
    plot_crop: PlotCropResult,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
) -> list[CurveData]:
    """
    Extract one or more XRD curves from a cropped plot.

    Returns a single curve for simple plots, multiple curves for overlay or
    stacked multi-trace figures.
    """
    cropped = plot_crop.cropped_bgr
    masks = _build_curve_masks(
        cropped,
        plot_left=calibration.plot_left,
        plot_right=calibration.plot_right,
        plot_top=calibration.plot_top,
        plot_bottom=calibration.plot_bottom,
    )
    masks = _filter_spurious_colored_masks(masks, calibration)
    layout = _classify_plot_layout(cropped, calibration, masks)
    curves: list[CurveData] = []

    if layout == "single":
        curve = _simplify_from_best_mask(
            cropped,
            calibration,
            masks,
            num_points=num_points,
        )
        if curve is not None:
            curves.append(curve)
    elif layout == "stacked":
        colored_masks = [
            (name, mask)
            for name, mask in masks
            if name != "black"
            and _mask_column_coverage(
                mask[
                    calibration.plot_top : calibration.plot_bottom,
                    calibration.plot_left : calibration.plot_right,
                ]
            )
            >= max(20, int((calibration.plot_right - calibration.plot_left) * MIN_CURVE_COVERAGE_FRACTION))
        ]
        if len(colored_masks) >= 2 and _masks_are_vertically_stacked(
            colored_masks,
            calibration.plot_top,
            calibration.plot_bottom,
            calibration.plot_left,
            calibration.plot_right,
        ):
            assigned = _assign_masks_to_horizontal_bands(
                colored_masks,
                calibration.plot_top,
                calibration.plot_bottom,
                calibration.plot_left,
                calibration.plot_right,
            )
            for curve_index, (color_name, mask, band_top, band_bottom) in enumerate(assigned, start=1):
                curve = _curve_from_mask(
                    mask,
                    calibration,
                    band_top=band_top,
                    band_bottom=band_bottom,
                    curve_id=f"curve_{curve_index}_{color_name}",
                    color_name=color_name,
                    stacked=True,
                    num_points=num_points,
                )
                if curve is not None:
                    curves.append(curve)
        else:
            bands = detect_stacked_curve_bands(
                cropped,
                calibration.plot_top,
                calibration.plot_bottom,
                calibration.plot_left,
                calibration.plot_right,
            )
            for band_index, (band_top, band_bottom) in enumerate(bands, start=1):
                band_masks = _build_curve_masks(
                    cropped,
                    plot_left=calibration.plot_left,
                    plot_right=calibration.plot_right,
                    plot_top=calibration.plot_top,
                    plot_bottom=calibration.plot_bottom,
                    band_top=band_top,
                    band_bottom=band_bottom,
                )
                picked = _pick_best_single_mask(band_masks, calibration)
                if not picked:
                    continue
                color_name, mask = picked
                curve = _curve_from_mask(
                    mask,
                    calibration,
                    band_top=band_top,
                    band_bottom=band_bottom,
                    curve_id=f"curve_{band_index}",
                    color_name=color_name,
                    stacked=True,
                    num_points=num_points,
                )
                if curve is not None:
                    curves.append(curve)
    else:
        colored_masks = [
            (name, mask)
            for name, mask in masks
            if name != "black"
            and _mask_column_coverage(
                mask[
                    calibration.plot_top : calibration.plot_bottom,
                    calibration.plot_left : calibration.plot_right,
                ]
            )
            >= max(20, int((calibration.plot_right - calibration.plot_left) * MIN_CURVE_COVERAGE_FRACTION))
        ]
        pool = colored_masks if colored_masks else masks
        for curve_index, (color_name, mask) in enumerate(pool, start=1):
            curve = _curve_from_mask(
                mask,
                calibration,
                band_top=calibration.plot_top,
                band_bottom=calibration.plot_bottom,
                curve_id=f"curve_{curve_index}_{color_name}",
                color_name=color_name,
                stacked=False,
                num_points=num_points,
            )
            if curve is not None:
                curves.append(curve)
        if len(curves) > 1:
            curves = _dedupe_similar_curves(curves)

    if len(curves) > 1:
        curves = _dedupe_similar_curves(curves)

    valid_curves = [curve for curve in curves if curve.two_theta]
    if not valid_curves:
        fallback = _simplify_from_best_mask(
            cropped,
            calibration,
            masks,
            num_points=num_points,
        )
        if fallback is not None:
            curves = [fallback]

    if not curves:
        return [
            CurveData(
                two_theta=[],
                intensity=[],
                warnings=["no_curve_pixels_detected"],
            )
        ]
    return curves


def _curve_from_mask(
    mask: np.ndarray,
    calibration: AxisCalibrationResult,
    *,
    band_top: int,
    band_bottom: int,
    curve_id: str,
    color_name: str,
    stacked: bool,
    num_points: int,
) -> CurveData | None:
    curve_warnings: list[str] = []
    if color_name == "black":
        curve_warnings.append("black_curve_extraction")
    if stacked:
        curve_warnings.append("stacked_curve_band")

    points = _extract_trace_from_mask(
        mask,
        calibration.plot_left,
        calibration.plot_right,
        calibration.plot_top,
        calibration.plot_bottom,
        band_top=band_top,
        band_bottom=band_bottom,
        stacked=stacked,
        lower_envelope=color_name == "black" and not stacked,
    )
    if points.size == 0:
        return None

    x_pixels = points[:, 0] + calibration.plot_left
    y_pixels = points[:, 1] + band_top

    two_theta = _pixel_to_two_theta(x_pixels, calibration)
    intensity = _pixel_y_to_intensity(
        y_pixels,
        calibration,
        band_top=band_top,
        band_bottom=band_bottom,
        stacked=stacked,
    )

    two_theta, intensity = _resample_uniform(two_theta, intensity, num_points=num_points)
    if not calibration.has_y_calibration:
        intensity = _normalize_intensity(intensity)

    if len(two_theta) > 10:
        coverage = (two_theta.max() - two_theta.min()) / max(
            calibration.x_max - calibration.x_min,
            1e-6,
        )
        if coverage < 0.5:
            curve_warnings.append("partial_curve_coverage")

    if float(np.max(intensity)) <= 0:
        curve_warnings.append("zero_intensity_curve")
        return None

    if not _is_valid_curve(intensity):
        return None

    return CurveData(
        two_theta=two_theta.tolist(),
        intensity=intensity.tolist(),
        curve_id=curve_id,
        color=color_name if color_name != "auto" else None,
        label=color_name if color_name != "auto" else None,
        warnings=sorted(set(curve_warnings)),
    )


def digitize_xrd_curve(
    plot_crop: PlotCropResult,
    calibration: AxisCalibrationResult,
    *,
    num_points: int = 2000,
) -> CurveData:
    """Backward-compatible single-curve entry point."""
    curves = digitize_xrd_curves(plot_crop, calibration, num_points=num_points)
    if not curves:
        return CurveData(two_theta=[], intensity=[], warnings=["no_curve_pixels_detected"])
    return curves[0]
