from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from xrd_digitization.types import CurveData, PeakRecord


def detect_peaks(
    curve: CurveData,
    *,
    min_prominence: float = 3.0,
    min_distance_points: int = 8,
    max_peaks: int = 30,
) -> list[PeakRecord]:
    """
    Detect major peaks in a digitized XRD curve.

    Simplified constant-width curves reuse the substantial-peak detector.
    """
    if "simplified_constant_width_peaks" in curve.warnings and curve.detected_peaks:
        return list(curve.detected_peaks)

    if "simplified_constant_width_peaks" in curve.warnings:
        from xrd_digitization.simplify_curve import detect_substantial_peaks

        x = np.array(curve.two_theta, dtype=float)
        y = np.array(curve.intensity, dtype=float)
        return detect_substantial_peaks(x, y, max_peaks=max_peaks)

    if len(curve.two_theta) < 5 or len(curve.intensity) < 5:
        return []

    x = np.array(curve.two_theta, dtype=float)
    y = np.array(curve.intensity, dtype=float)

    if len(x) > 15:
        window = min(15, len(y) // 20 * 2 + 1)
        if window >= 5:
            kernel = np.ones(window) / window
            y = np.convolve(y, kernel, mode="same")

    min_distance = max(1, min_distance_points // 2)
    if len(x) > 1:
        dx = float(np.median(np.diff(x)))
        if dx > 0:
            min_distance = max(min_distance, int(0.3 / dx))

    peaks, properties = find_peaks(
        y,
        prominence=min_prominence,
        distance=min_distance,
        height=5.0,
    )

    if len(peaks) == 0:
        peaks, properties = find_peaks(
            y,
            prominence=max(1.0, min_prominence * 0.5),
            distance=max(1, min_distance // 2),
        )

    prominences = properties.get("prominences", np.zeros(len(peaks)))
    order = np.argsort(y[peaks])[::-1]
    peaks = peaks[order][:max_peaks]
    prominences = prominences[order][:max_peaks] if len(prominences) else np.zeros(len(peaks))

    records: list[PeakRecord] = []
    for idx, prom in zip(peaks, prominences):
        records.append(
            PeakRecord(
                two_theta=float(x[idx]),
                relative_intensity=float(y[idx]),
                prominence=float(prom),
            )
        )

    records.sort(key=lambda p: p.two_theta)
    return records
