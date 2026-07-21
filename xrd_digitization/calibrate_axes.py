from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from xrd_digitization.types import AxisCalibrationResult, PlotCropResult

DEFAULT_X_RANGES: list[tuple[float, float]] = [
    (5.0, 80.0),
    (10.0, 80.0),
    (5.0, 90.0),
]

NUMBER_PATTERN = re.compile(r"^\d{1,3}(?:\.\d+)?$")
Y_NUMBER_PATTERN = re.compile(r"^\d{1,6}(?:\.\d+)?$")
NUMBER_FIND_PATTERN = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\b")
Y_NUMBER_FIND_PATTERN = re.compile(r"\b(\d{1,6}(?:\.\d+)?)\b")
MAX_Y_TICK_VALUE = 200_000.0
MIN_Y_TICK_VALUE = 0.0
NORMALIZED_AXIS_MAX = 1.5


def _tick_dedup_key(value: float) -> float | int:
    """Avoid collapsing fractional tick labels (0.2, 0.4, …) onto the same key."""
    if value <= NORMALIZED_AXIS_MAX:
        return round(value, 3)
    return int(round(value))


def _is_normalized_unit_axis(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    lo, hi = min(values), max(values)
    return hi <= NORMALIZED_AXIS_MAX and lo >= -0.05


COMMON_XRD_TICK_STEPS = (1.0, 2.0, 5.0, 10.0, 20.0, 25.0)


def _nearest_common_tick_step(value: float) -> float:
    """Snap an inferred spacing to a standard 2θ tick step."""
    return min(COMMON_XRD_TICK_STEPS, key=lambda step: abs(step - value))


def _minimum_arithmetic_step(values: list[float]) -> float:
    if _is_normalized_unit_axis(values):
        return 0.04
    if len(values) >= 2:
        ordered = sorted(values)
        diffs = [
            ordered[index + 1] - ordered[index]
            for index in range(len(ordered) - 1)
            if ordered[index + 1] - ordered[index] > 0
        ]
        if diffs:
            median_diff = float(np.median(diffs))
            if median_diff >= 0.9:
                return _nearest_common_tick_step(median_diff)
            return max(0.5, median_diff)
    if values and max(values) <= 20.0:
        return 0.5
    return 5.0


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _preprocess_label_band(gray: np.ndarray) -> np.ndarray:
    upscaled = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(upscaled, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(upscaled, 180, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_or(otsu, fixed)


def _ocr_numeric_boxes(
    binary: np.ndarray,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
    scale: float = 1.0,
) -> list[tuple[int, float, str]]:
    import pytesseract

    labels: list[tuple[int, float, str]] = []
    for psm in (11, 6, 7):
        config = f"--psm {psm} -c tessedit_char_whitelist=0123456789."
        data = pytesseract.image_to_data(
            binary,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        for i, text in enumerate(data["text"]):
            raw = (text or "").strip()
            if not raw or not NUMBER_PATTERN.match(raw):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if not (5.0 <= value <= 120.0):
                continue
            conf = float(data["conf"][i])
            if conf < 20:
                continue
            center_x = int((data["left"][i] + data["width"][i] / 2) / scale + x_offset)
            labels.append((center_x, value, raw))

    dedup: dict[int, tuple[int, float, str]] = {}
    for center_x, value, raw in labels:
        key = int(round(value))
        existing = dedup.get(key)
        if existing is None or abs(existing[0] - center_x) > 5:
            dedup[key] = (center_x, value, raw)
    return list(dedup.values())


def _ocr_numeric_boxes_y(
    binary: np.ndarray,
    *,
    y_offset: int = 0,
    scale: float = 1.0,
    value_min: float = 0.0,
    value_max: float = MAX_Y_TICK_VALUE,
) -> list[tuple[int, float, str]]:
    """OCR numeric tick labels from a vertical (y-axis) band."""
    import pytesseract

    labels: list[tuple[int, float, str]] = []
    for psm in (11, 6, 7):
        config = f"--psm {psm} -c tessedit_char_whitelist=0123456789."
        data = pytesseract.image_to_data(
            binary,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        for i, text in enumerate(data["text"]):
            raw = (text or "").strip()
            if not raw or not Y_NUMBER_PATTERN.match(raw):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if not (value_min <= value <= value_max):
                continue
            conf = float(data["conf"][i])
            if conf < 20:
                continue
            center_y = int((data["top"][i] + data["height"][i] / 2) / scale + y_offset)
            labels.append((center_y, value, raw))

    dedup: dict[float | int, tuple[int, float, str]] = {}
    for center_y, value, raw in labels:
        key = _tick_dedup_key(value)
        existing = dedup.get(key)
        if existing is None or abs(existing[0] - center_y) > 5:
            dedup[key] = (center_y, value, raw)
    return list(dedup.values())


def _ocr_y_tick_labels_from_image(
    image_bgr: np.ndarray,
    *,
    band_left_ratio: float = 0.0,
    band_right_ratio: float = 0.22,
) -> list[tuple[int, float, str]]:
    """OCR numeric y-axis tick labels from the left margin band."""
    height, width = image_bgr.shape[:2]
    band_left = int(width * band_left_ratio)
    band_right = int(width * band_right_ratio)
    band = image_bgr[:, band_left:band_right]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    binary = _preprocess_label_band(gray)
    labels = _ocr_numeric_boxes_y(
        binary,
        y_offset=0,
        scale=4.0,
        value_min=MIN_Y_TICK_VALUE,
        value_max=MAX_Y_TICK_VALUE,
    )
    if len(labels) >= 2:
        return labels

    import pytesseract

    text = pytesseract.image_to_string(binary, config="--psm 6")
    numbers = [float(m.group(1)) for m in Y_NUMBER_FIND_PATTERN.finditer(text)]
    numbers = sorted({n for n in numbers if MIN_Y_TICK_VALUE <= n <= MAX_Y_TICK_VALUE})
    if len(numbers) >= 2:
        step = height / max(len(numbers) - 1, 1)
        return [
            (int(round(i * step)), value, str(int(value) if value.is_integer() else value))
            for i, value in enumerate(reversed(numbers))
        ]
    return labels


def _ocr_y_tick_labels_full_image(
    full_image_bgr: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
) -> list[tuple[int, float, str]]:
    """OCR y-axis labels using the crop and full-image left margin."""
    x0, y0, x1, y1 = crop_bbox
    crop = full_image_bgr[y0:y1, x0:x1]

    labels = _ocr_y_tick_labels_from_image(crop)
    if len(labels) >= 2:
        return labels

    height, width = full_image_bgr.shape[:2]
    band_right = max(int(width * 0.18), x0)
    band = full_image_bgr[y0:y1, 0:band_right]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    binary = _preprocess_label_band(gray)
    raw_labels = _ocr_numeric_boxes_y(
        binary,
        y_offset=y0,
        scale=4.0,
        value_min=0.0,
        value_max=MAX_Y_TICK_VALUE,
    )

    crop_labels: list[tuple[int, float, str]] = []
    for center_y, value, raw in raw_labels:
        rel_y = center_y - y0
        if 0 <= rel_y <= (y1 - y0):
            crop_labels.append((rel_y, value, raw))

    import pytesseract

    text = pytesseract.image_to_string(binary, config="--psm 6")
    numbers = sorted(
        {
            float(m.group(1))
            for m in Y_NUMBER_FIND_PATTERN.finditer(text)
            if MIN_Y_TICK_VALUE <= float(m.group(1)) <= MAX_Y_TICK_VALUE
        }
    )
    plot_height = y1 - y0
    if len(numbers) >= 2 and len(numbers) > len({round(v) for _, v, _ in crop_labels}):
        step = plot_height / max(len(numbers) - 1, 1)
        return [
            (int(round(i * step)), value, str(int(value) if value.is_integer() else value))
            for i, value in enumerate(reversed(numbers))
        ]

    if len(crop_labels) >= 2:
        return crop_labels
    return crop_labels


def _prune_y_tick_outliers(
    y_tick_pairs: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Drop spurious y ticks (e.g. OCR '2' from 2θ labels) that break scale."""
    if len(y_tick_pairs) < 2:
        return y_tick_pairs

    ordered = sorted(y_tick_pairs, key=lambda pair: pair[0])
    values = [value for _, value in ordered]
    positive = [value for value in values if value > 0.0]
    if positive:
        scale_ref = max(positive)
        if scale_ref >= 100.0:
            ordered = [
                pair
                for pair in ordered
                if pair[1] == 0.0 or pair[1] >= scale_ref * 0.05
            ]

    if len(ordered) < 2:
        return y_tick_pairs

    fit = _fit_linear_calibration([(row, value) for row, value in ordered])
    if fit is None or len(ordered) < 3:
        return ordered

    intercept, slope, rmse = fit
    value_span = abs(ordered[-1][1] - ordered[0][1])
    rmse_limit = max(0.02, value_span * 0.08) if value_span <= NORMALIZED_AXIS_MAX else max(50.0, value_span * 0.08)
    if rmse <= rmse_limit:
        return ordered

    worst = max(
        range(len(ordered)),
        key=lambda index: abs(
            slope * ordered[index][0] + intercept - ordered[index][1]
        ),
    )
    pruned = ordered[:worst] + ordered[worst + 1 :]
    return pruned if len(pruned) >= 2 else ordered


def _filter_y_tick_pairs(
    y_tick_pairs: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    if len(y_tick_pairs) < 2:
        return y_tick_pairs

    y_tick_pairs = _prune_y_tick_outliers(y_tick_pairs)
    if len(y_tick_pairs) < 2:
        return y_tick_pairs

    values = [v for _, v in y_tick_pairs]
    sequence = _select_arithmetic_number_sequence(values)
    if len(sequence) >= 2:
        if _is_normalized_unit_axis(values):
            allowed = sequence
            filtered = [
                (y, v)
                for y, v in y_tick_pairs
                if any(abs(v - candidate) <= 0.06 for candidate in allowed)
            ]
        else:
            allowed = {round(v) for v in sequence}
            filtered = [(y, v) for y, v in y_tick_pairs if round(v) in allowed]
        if len(filtered) >= 2:
            return sorted(filtered, key=lambda pair: pair[0])

    ordered = sorted(y_tick_pairs, key=lambda pair: pair[0])
    if len(ordered) >= 2:
        step = abs(ordered[-1][1] - ordered[0][1]) / max(len(ordered) - 1, 1)
        if step >= 100 or (_is_normalized_unit_axis(values) and step >= 0.04):
            kept = [ordered[0]]
            for pair in ordered[1:]:
                if abs(pair[1] - kept[-1][1]) >= step * 0.75:
                    kept.append(pair)
            if len(kept) >= 2:
                return kept
    return y_tick_pairs


def _fit_y_calibration(
    y_tick_pairs: list[tuple[int, float]],
    plot_top: int,
    plot_bottom: int,
) -> tuple[float, float, str, float] | None:
    """Map pixel y to intensity using OCR tick pairs (y increases downward)."""
    if len(y_tick_pairs) < 2:
        return None

    fit = _fit_linear_calibration(y_tick_pairs)
    if fit is None:
        return None

    intercept, slope, rmse = fit
    if abs(slope) < 1e-6:
        return None

    y_at_top = slope * plot_top + intercept
    y_at_bottom = slope * plot_bottom + intercept
    y_min = min(y_at_top, y_at_bottom)
    y_max = max(y_at_top, y_at_bottom)
    confidence = max(0.55, min(0.95, 1.0 - rmse / max(abs(y_max - y_min) * 0.05, 1.0)))
    return float(y_min), float(y_max), "ocr_linear_regression", confidence


def _ocr_tick_labels_from_image(
    image_bgr: np.ndarray,
    *,
    band_top_ratio: float = 0.55,
    band_bottom_ratio: float = 0.98,
) -> list[tuple[int, float, str]]:
    """OCR numeric tick labels from a horizontal band of an image."""
    height, _ = image_bgr.shape[:2]
    band_top = int(height * band_top_ratio)
    band_bottom = int(height * band_bottom_ratio)
    band = image_bgr[band_top:band_bottom, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    binary = _preprocess_label_band(gray)
    labels = _ocr_numeric_boxes(binary, x_offset=0, y_offset=band_top, scale=4.0)
    if len(labels) >= 2:
        return labels

    import pytesseract

    text = pytesseract.image_to_string(binary, config="--psm 6")
    numbers = [float(m.group(1)) for m in NUMBER_FIND_PATTERN.finditer(text)]
    numbers = sorted({n for n in numbers if 5.0 <= n <= 120.0})
    if len(numbers) >= 2:
        width = image_bgr.shape[1]
        step = width / max(len(numbers) - 1, 1)
        return [
            (int(round(i * step)), value, str(int(value) if value.is_integer() else value))
            for i, value in enumerate(numbers)
        ]
    return labels


def _select_arithmetic_number_sequence(numbers: list[float]) -> list[float]:
    """Pick the longest evenly spaced subsequence from OCR numbers."""
    numbers = sorted(set(numbers))
    if len(numbers) < 2:
        return numbers

    best = numbers[:2]
    best_step = numbers[1] - numbers[0] if len(numbers) >= 2 else 0.0
    min_step = _minimum_arithmetic_step(numbers)
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            step = numbers[j] - numbers[i]
            if step < min_step:
                continue
            seq = [numbers[i], numbers[j]]
            current = numbers[j]
            tolerance = max(0.05, step * 0.05) if max(numbers) <= NORMALIZED_AXIS_MAX else max(0.5, step * 0.05)
            for value in numbers[j + 1 :]:
                if abs(value - (current + step)) <= tolerance:
                    seq.append(value)
                    current = value
            if len(seq) > len(best) or (
                len(seq) == len(best)
                and seq
                and best
                and seq[0] > best[0]
            ):
                best = seq
                best_step = step
    return best


def _value_span_calibration(tick_pairs: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Map plot edges to arithmetic tick labels when numeric values are reliable."""
    if len(tick_pairs) < 2:
        return None

    values = [v for _, v in tick_pairs]
    sequence = _select_arithmetic_number_sequence(values)
    if len(sequence) < 2:
        return None

    step = sequence[1] - sequence[0]
    if step <= 0:
        return None

    x_min = float(sequence[0])
    x_max = float(sequence[-1] + step)
    return x_min, x_max


def _ocr_tick_labels_full_image(
    full_image_bgr: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
) -> list[tuple[int, float, str]]:
    """
    OCR x-axis labels using the crop and full-image bands, mapped to crop coordinates.
    """
    x0, y0, x1, y1 = crop_bbox
    crop = full_image_bgr[y0:y1, x0:x1]

    labels = _ocr_tick_labels_from_image(crop)
    if len(labels) >= 2:
        return labels

    height, width = full_image_bgr.shape[:2]
    band_top = max(int(height * 0.55), y0 + int((y1 - y0) * 0.55))
    band = full_image_bgr[band_top:, max(0, x0 - 10) : min(width, x1 + 10)]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    binary = _preprocess_label_band(gray)
    raw_labels = _ocr_numeric_boxes(binary, x_offset=max(0, x0 - 10), scale=4.0)

    crop_labels: list[tuple[int, float, str]] = []
    for center_x, value, raw in raw_labels:
        rel_x = center_x - x0
        if 0 <= rel_x <= (x1 - x0):
            crop_labels.append((rel_x, value, raw))

    import pytesseract

    text = pytesseract.image_to_string(binary, config="--psm 6")
    numbers = [float(m.group(1)) for m in NUMBER_FIND_PATTERN.finditer(text)]
    numbers = _select_arithmetic_number_sequence([n for n in numbers if 5.0 <= n <= 120.0])
    plot_width = x1 - x0

    if len(numbers) >= 3 and len(numbers) > len({round(v) for _, v, _ in crop_labels}):
        step = plot_width / max(len(numbers) - 1, 1)
        return [
            (int(round(i * step)), value, str(int(value) if value.is_integer() else value))
            for i, value in enumerate(numbers)
        ]

    if len(crop_labels) >= 2:
        return crop_labels

    return crop_labels


def _fill_missing_arithmetic_ticks(
    tick_pairs: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Insert missing evenly spaced tick labels between OCR pairs with value gaps."""
    if len(tick_pairs) < 2:
        return tick_pairs

    ordered = sorted(tick_pairs, key=lambda pair: pair[0])
    small_steps = [
        round(ordered[i + 1][1] - ordered[i][1], 6)
        for i in range(len(ordered) - 1)
        if 0 < ordered[i + 1][1] - ordered[i][1] <= 20
    ]
    if not small_steps:
        return tick_pairs

    step = float(np.median(small_steps))
    if step <= 0:
        return tick_pairs

    filled: list[tuple[int, float]] = [ordered[0]]
    for (x0, v0), (x1, v1) in zip(ordered, ordered[1:]):
        delta_v = v1 - v0
        n_steps = int(round(delta_v / step))
        if n_steps > 1 and abs(delta_v - n_steps * step) < step * 0.15:
            for k in range(1, n_steps):
                frac = k / n_steps
                px = int(round(x0 + frac * (x1 - x0)))
                val = v0 + k * step
                filled.append((px, val))
        filled.append((x1, v1))

    dedup: dict[int, tuple[int, float]] = {}
    for px, val in filled:
        key = int(round(val))
        dedup.setdefault(key, (px, val))
    return sorted(dedup.values(), key=lambda pair: pair[0])


def _infer_arithmetic_ticks(
    tick_pairs: list[tuple[int, float]],
    plot_width: int,
) -> list[tuple[int, float]]:
    """Extend tick pairs when labels form an evenly spaced arithmetic sequence."""
    if len(tick_pairs) < 2:
        return tick_pairs

    ordered = sorted(tick_pairs, key=lambda p: p[0])
    values = [v for _, v in ordered]
    steps = [round(values[i + 1] - values[i], 6) for i in range(len(values) - 1)]
    if not steps:
        return tick_pairs

    median_step = float(np.median(steps))
    if median_step <= 0:
        return tick_pairs

    if max(steps) - min(steps) > max(1.0, median_step * 0.35):
        return tick_pairs

    extended = list(ordered)
    # Extrapolate one tick to the right if the axis likely continues.
    last_x, last_v = ordered[-1]
    expected_next = last_v + median_step
    if expected_next <= 120.0:
        step_px = ordered[-1][0] - ordered[-2][0]
        next_x = int(last_x + step_px)
        if next_x <= int(plot_width * 1.08):
            extended.append((next_x, expected_next))

    # Extrapolate one tick to the left when needed.
    first_x, first_v = ordered[0]
    expected_prev = first_v - median_step
    if expected_prev >= 5.0 and len(ordered) >= 2:
        prev_x = int(first_x - (ordered[1][0] - ordered[0][0]))
        if prev_x >= 0:
            extended.insert(0, (prev_x, expected_prev))

    return sorted(extended, key=lambda p: p[0])


def _repair_short_tick_labels(tick_pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Fix OCR reads like 6.0 on an otherwise 10-degree tick ladder."""
    if len(tick_pairs) < 3:
        return tick_pairs

    ordered = sorted(tick_pairs, key=lambda pair: pair[0])
    values = [value for _, value in ordered]
    sequence = _select_arithmetic_number_sequence(values)
    if len(sequence) < 3:
        return tick_pairs

    step = sequence[1] - sequence[0]
    if step < 1.0:
        return tick_pairs

    expected = {round(sequence[0] + index * step, 6) for index in range(len(sequence) + 2)}
    repaired: list[tuple[int, float]] = []
    for px, value in ordered:
        if value in expected:
            repaired.append((px, value))
            continue
        candidates = [candidate for candidate in expected if abs(candidate - value) <= step * 0.25]
        if len(candidates) == 1:
            repaired.append((px, candidates[0]))
        else:
            repaired.append((px, value))
    return repaired


def _filter_monotonic_x_tick_pairs(
    tick_pairs: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Drop OCR ticks whose values fall while pixel x increases (misreads like 45→10)."""
    if len(tick_pairs) < 2:
        return tick_pairs

    ordered = sorted(tick_pairs, key=lambda pair: pair[0])
    kept = [ordered[0]]
    for px, val in ordered[1:]:
        if val > kept[-1][1] + 1e-6:
            kept.append((px, val))
    return kept


def _prune_x_tick_outliers_by_rmse(
    tick_pairs: list[tuple[int, float]],
    *,
    max_rmse: float = 1.0,
) -> list[tuple[int, float]]:
    """Iteratively remove x tick labels with the largest regression residual."""
    pairs = list(tick_pairs)
    while len(pairs) >= 3:
        fit = _fit_linear_calibration(pairs)
        if fit is None:
            break
        intercept, slope, rmse = fit
        if rmse <= max_rmse:
            break
        worst = max(
            range(len(pairs)),
            key=lambda index: abs(slope * pairs[index][0] + intercept - pairs[index][1]),
        )
        pairs.pop(worst)
    return pairs


def _fit_linear_calibration(pairs: list[tuple[int, float]]) -> tuple[float, float, float] | None:
    if len(pairs) < 2:
        return None

    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    coeffs = np.polyfit(xs, ys, 1)
    slope, intercept = coeffs
    if abs(slope) < 1e-6:
        return None
    predicted = slope * xs + intercept
    rmse = float(np.sqrt(np.mean((predicted - ys) ** 2)))
    return float(intercept), float(slope), rmse


def _detect_plot_bounds_from_axes(cropped_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the inner plotting area from the frame / axis lines."""
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    dark = gray < 120

    left = int(width * 0.08)
    right = width - 1
    top = int(height * 0.05)
    bottom = int(height * 0.92)

    # Left y-axis: darkest vertical column in the left margin.
    left_band = dark[:, : int(width * 0.22)]
    if left_band.size:
        col_scores = left_band.sum(axis=0)
        if col_scores.max() > height * 0.15:
            left = int(np.argmax(col_scores))

    # Bottom x-axis: strongest horizontal line in the lower third of the crop.
    search_top = int(height * 0.55)
    bottom_region = dark[search_top:, :]
    if bottom_region.size:
        row_scores = bottom_region.sum(axis=1)
        min_score = max(width * 0.12, 20)
        strong = np.where(row_scores >= min_score)[0]
        if len(strong):
            bottom = int(search_top + strong[-1])
        elif row_scores.max() > 0:
            bottom = int(search_top + int(np.argmax(row_scores)))

    # Top boundary: first strong horizontal line in the upper margin.
    top_region = dark[: int(height * 0.22), left : int(width * 0.95)]
    if top_region.size:
        row_scores = top_region.sum(axis=1)
        if row_scores.max() > top_region.shape[1] * 0.25:
            top = max(top, int(np.argmax(row_scores)) + 1)

    # Right frame (optional).
    right_band = dark[top:bottom, int(width * 0.92) :]
    if right_band.size:
        col_scores = right_band.sum(axis=0)
        if col_scores.max() > (bottom - top) * 0.25:
            right = int(width * 0.92 + np.argmax(col_scores))

    left = min(max(left, 0), width - 2)
    right = max(min(right, width - 1), left + 1)
    top = min(max(top, 0), height - 2)
    bottom = max(min(bottom, height - 1), top + 1)
    return left, top, right, bottom


def _extend_plot_right_to_curve(
    cropped_bgr: np.ndarray,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
) -> int:
    """Extend the plot width when trace pixels continue past the detected frame."""
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[plot_top:plot_bottom, plot_left:]
    if roi.size == 0:
        return plot_right

    last_curve_col = plot_right - plot_left
    for col in range(roi.shape[1] - 1, last_curve_col, -1):
        column = roi[:, col]
        if float(column.min()) > 200:
            continue
        dark_frac = float(np.count_nonzero(column < 200)) / float(len(column))
        if dark_frac > 0.5:
            continue
        last_curve_col = col
        break
    return plot_left + last_curve_col


def _choose_default_range(
    tick_pairs: list[tuple[int, float]],
) -> tuple[float, float, str]:
    if tick_pairs:
        values = [v for _, v in tick_pairs]
        return min(values) - 5.0, max(values) + 5.0, "partial_ocr_guess"

    x_min, x_max = DEFAULT_X_RANGES[0]
    return x_min, x_max, "default_range"


def merge_panel_x_calibration(
    figure_calibration: AxisCalibrationResult,
    panel_calibration: AxisCalibrationResult,
    *,
    figure_crop_bbox: tuple[int, int, int, int],
    panel_crop_bbox: tuple[int, int, int, int],
) -> AxisCalibrationResult:
    """
    Apply figure-level x-axis calibration to a subplot panel.

    Maps panel-local x pixels onto the figure crop coordinate system so OCR
    tick labels taken from the full figure still apply to upper panels.
    """
    fit = _fit_linear_calibration(figure_calibration.tick_pairs)
    if fit and len(figure_calibration.tick_pairs) >= 2:
        intercept, slope, _ = fit
        fx0, _, fx1, _ = figure_crop_bbox
        px0, _, px1, _ = panel_crop_bbox
        panel_left_full = float(panel_calibration.plot_left + (px0 - fx0))
        panel_right_full = float(panel_calibration.plot_right + (px0 - fx0))
        x_min = slope * panel_left_full + intercept
        x_max = slope * panel_right_full + intercept
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if figure_calibration.method.startswith("ocr"):
            x_min = figure_calibration.x_min
            x_max = figure_calibration.x_max
        return AxisCalibrationResult(
            x_min=float(x_min),
            x_max=float(x_max),
            plot_left=panel_calibration.plot_left,
            plot_right=panel_calibration.plot_right,
            plot_top=panel_calibration.plot_top,
            plot_bottom=panel_calibration.plot_bottom,
            method=figure_calibration.method,
            confidence=figure_calibration.confidence,
            tick_pairs=figure_calibration.tick_pairs,
            warnings=sorted(set(figure_calibration.warnings + panel_calibration.warnings)),
        )

    return AxisCalibrationResult(
        x_min=figure_calibration.x_min,
        x_max=figure_calibration.x_max,
        plot_left=panel_calibration.plot_left,
        plot_right=panel_calibration.plot_right,
        plot_top=panel_calibration.plot_top,
        plot_bottom=panel_calibration.plot_bottom,
        method=figure_calibration.method,
        confidence=min(figure_calibration.confidence, panel_calibration.confidence),
        tick_pairs=figure_calibration.tick_pairs,
        warnings=sorted(set(figure_calibration.warnings + panel_calibration.warnings)),
    )


def _infer_snap_step(tick_pairs: list[tuple[int, float]]) -> float:
    values = sorted({round(value, 4) for _, value in tick_pairs})
    if len(values) < 2:
        return 5.0

    diffs = [
        values[index + 1] - values[index]
        for index in range(len(values) - 1)
        if values[index + 1] - values[index] > 0
    ]
    if not diffs:
        return 5.0

    median_diff = float(np.median(diffs))
    if median_diff >= 0.9:
        common_step = _nearest_common_tick_step(median_diff)
        if abs(common_step - median_diff) <= max(0.15, common_step * 0.12):
            return common_step
    return max(0.5, median_diff)


def _resolve_nearby_tick_conflicts(
    tick_pairs: list[tuple[int, float]],
    *,
    max_px_gap: int = 20,
) -> list[tuple[int, float]]:
    """Drop OCR ghost reads like 75.0 beside the correct 7.5 label."""
    if len(tick_pairs) < 2:
        return tick_pairs

    ordered = sorted(tick_pairs, key=lambda pair: pair[0])
    resolved: list[tuple[int, float]] = []
    index = 0
    while index < len(ordered):
        cluster = [ordered[index]]
        next_index = index + 1
        while (
            next_index < len(ordered)
            and ordered[next_index][0] - cluster[0][0] <= max_px_gap
        ):
            cluster.append(ordered[next_index])
            next_index += 1

        if len(cluster) == 1:
            resolved.append(cluster[0])
        else:
            values = [value for _, value in cluster]
            ratio = max(values) / max(min(values), 1e-6)
            if ratio > 5.0:
                resolved.append(min(cluster, key=lambda pair: pair[1]))
            else:
                global_values = sorted({value for _, value in ordered})
                best = cluster[0]
                best_score = -1
                for candidate in cluster:
                    candidate_values = sorted(
                        {value for _, value in resolved}
                        | {value for pair in cluster if pair != candidate for value in [pair[1]]}
                        | {candidate[1]}
                    )
                    score = len(_select_arithmetic_number_sequence(candidate_values))
                    if score > best_score:
                        best_score = score
                        best = candidate
                resolved.append(best)
        index = next_index

    return resolved


def _snap_tick_values(
    tick_pairs: list[tuple[int, float]],
    *,
    step: float | None = None,
) -> list[tuple[int, float]]:
    """Snap OCR x tick values to a regular degree grid."""
    if not tick_pairs:
        return tick_pairs
    if step is None:
        step = _infer_snap_step(tick_pairs)

    degree_axis = not _is_normalized_unit_axis([value for _, value in tick_pairs])
    snapped: list[tuple[int, float]] = []
    for px, val in tick_pairs:
        snapped_val = round(val / step) * step
        if degree_axis and step >= 1.0:
            snapped_val = float(round(snapped_val))
        snapped.append((px, snapped_val))
    dedup: dict[int, tuple[int, float]] = {}
    for px, val in snapped:
        key = int(round(val * 100))
        if key not in dedup:
            dedup[key] = (px, val)
    return sorted(dedup.values(), key=lambda pair: pair[0])


def calibrate_axes(
    plot_crop: PlotCropResult,
    *,
    full_image_bgr: np.ndarray | None = None,
) -> AxisCalibrationResult:
    """
    Detect plot bounds and calibrate pixel x positions to 2θ values.

    Uses OCR on x-axis tick labels when possible; otherwise falls back to
    configurable default ranges.
    """
    warnings: list[str] = []
    cropped = plot_crop.cropped_bgr
    x0, y0, x1, y1 = plot_crop.bbox
    plot_left, plot_top, plot_right, plot_bottom = _detect_plot_bounds_from_axes(cropped)
    plot_right = _extend_plot_right_to_curve(
        cropped, plot_left, plot_right, plot_top, plot_bottom
    )

    tick_pairs: list[tuple[int, float]] = []
    method = "default_fallback"
    confidence = 0.35
    x_min, x_max = DEFAULT_X_RANGES[0]

    if _tesseract_available():
        try:
            if full_image_bgr is not None:
                ocr_labels = _ocr_tick_labels_full_image(full_image_bgr, plot_crop.bbox)
            else:
                raw = _ocr_tick_labels_from_image(cropped)
                ocr_labels = raw

            tick_pairs = [(x, val) for x, val, _ in ocr_labels]
            tick_pairs = _resolve_nearby_tick_conflicts(tick_pairs)
            tick_pairs = _filter_monotonic_x_tick_pairs(tick_pairs)
            tick_pairs = _prune_x_tick_outliers_by_rmse(tick_pairs)
            tick_pairs = _repair_short_tick_labels(tick_pairs)
            tick_pairs = _snap_tick_values(tick_pairs)
            tick_pairs = _filter_monotonic_x_tick_pairs(tick_pairs)
            tick_pairs = _fill_missing_arithmetic_ticks(tick_pairs)
            tick_pairs = _infer_arithmetic_ticks(tick_pairs, plot_right - plot_left)
            fit = _fit_linear_calibration(tick_pairs)

            if fit and len(tick_pairs) >= 2:
                intercept, slope, rmse = fit
                x_min = slope * plot_left + intercept
                x_max = slope * plot_right + intercept
                if x_min > x_max:
                    x_min, x_max = x_max, x_min
                method = "ocr_linear_regression"
                confidence = max(0.55, min(0.95, 1.0 - rmse / 8.0))
            elif len(tick_pairs) >= 2:
                x_min, x_max, method = _choose_default_range(tick_pairs)
                confidence = 0.45
            elif len(tick_pairs) == 1:
                warnings.append("single_tick_label_ocr")
                x_min, x_max, method = _choose_default_range(tick_pairs)
                method = "ocr_single_label_with_default_span"
                confidence = 0.45
            else:
                warnings.append("missing_tick_labels")
                x_min, x_max, method = _choose_default_range(tick_pairs)
        except Exception:
            warnings.append("ocr_failed")
            x_min, x_max, method = _choose_default_range(tick_pairs)
    else:
        warnings.append("tesseract_not_available")
        x_min, x_max, method = _choose_default_range(tick_pairs)

    if method == "default_range" and not tick_pairs and "missing_tick_labels" not in warnings:
        warnings.append("missing_tick_labels")

    if x_max - x_min < 5:
        x_min, x_max = DEFAULT_X_RANGES[0]
        warnings.append("invalid_calibrated_range_reset_to_default")

    y_tick_pairs: list[tuple[int, float]] = []
    y_min: float | None = None
    y_max: float | None = None
    y_method = "relative"

    if _tesseract_available():
        try:
            if full_image_bgr is not None:
                y_labels = _ocr_y_tick_labels_full_image(full_image_bgr, plot_crop.bbox)
            else:
                y_labels = _ocr_y_tick_labels_from_image(cropped)
            y_tick_pairs = [(y, val) for y, val, _ in y_labels]
            y_tick_pairs = _prune_y_tick_outliers(y_tick_pairs)
            y_tick_pairs = _filter_y_tick_pairs(y_tick_pairs)
            y_fit = _fit_y_calibration(y_tick_pairs, plot_top, plot_bottom)
            if y_fit is not None:
                y_min, y_max, y_method, y_conf = y_fit
                confidence = float(np.mean([confidence, y_conf]))
            elif y_tick_pairs:
                warnings.append("partial_y_axis_ocr")
        except Exception:
            warnings.append("y_ocr_failed")

    return AxisCalibrationResult(
        x_min=float(x_min),
        x_max=float(x_max),
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        method=method,
        confidence=confidence,
        tick_pairs=tick_pairs,
        y_min=y_min,
        y_max=y_max,
        y_tick_pairs=y_tick_pairs,
        y_method=y_method,
        warnings=warnings,
    )
