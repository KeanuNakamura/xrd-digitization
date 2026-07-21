from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xrd_digitization.calibrate_axes import calibrate_axes
from xrd_digitization.classify_xrd_figures import classify_from_context, classify_xrd_figure
from xrd_digitization.crop_plot_area import crop_plot_area
from xrd_digitization.detect_peaks import detect_peaks
from xrd_digitization.digitize_xrd_curve import digitize_xrd_curves
from xrd_digitization.plot_digitized_curve import plot_from_curves, save_multi_column_xy
from xrd_digitization.types import (
    AxisCalibrationResult,
    DigitizationResult,
    FigureContext,
    PanelDigitizationResult,
    PeakRecord,
    PlotCropResult,
    PlotPanel,
)


def _peaks_for_json(
    peaks: list[PeakRecord],
    calibration: AxisCalibrationResult,
) -> list[dict[str, float]]:
    """Write peak heights on a 0–100 scale; y-calibrated runs use raw units internally."""
    if not peaks:
        return []
    max_amp = max(max(float(peak.relative_intensity), 0.0) for peak in peaks)
    if max_amp <= 0:
        scale = 1.0
    else:
        scale = 100.0 / max_amp
    return [
        {
            "two_theta": peak.two_theta,
            "relative_intensity": float(peak.relative_intensity) * scale,
            "prominence": peak.prominence,
        }
        for peak in peaks
    ]

LOGGER = logging.getLogger(__name__)

FIGURE_NUMBER_PATTERN = re.compile(r"(?:figure[_-]?|fig[_-]?|sample_figure[_-]?)(\d+)", re.IGNORECASE)
STALE_OUTPUT_PATTERN = re.compile(
    r"^(?P<stem>.+?)_(?:\d+_)?digitized\.(?:png|xy)$"
)


def infer_output_stem(image_path: Path) -> str:
    match = FIGURE_NUMBER_PATTERN.search(image_path.stem)
    if match:
        return f"figure_{int(match.group(1))}"
    return f"{image_path.stem}_digitized"


def load_figure_context(
    image_path: Path,
    *,
    caption: str | None = None,
    figure_id: str | None = None,
    source_pdf: str | None = None,
    page: int | None = None,
    crop_bbox: list[float] | None = None,
) -> FigureContext:
    return FigureContext(
        image_path=image_path,
        figure_id=figure_id or image_path.stem,
        caption=caption,
        source_pdf=source_pdf,
        page=page,
        crop_bbox=crop_bbox,
    )


def _cleanup_stale_outputs(figure_dir: Path, stem: str) -> None:
    """Remove numbered digitized artifacts from prior multi-panel / multi-curve runs."""
    for path in figure_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name in {
            f"{stem}_digitized.png",
            f"{stem}_digitized.xy",
            f"{stem}.peaks.json",
            f"{stem}.json",
            f"{stem}.metadata.json",
        }:
            continue
        if re.match(rf"^{re.escape(stem)}(_\d+)+_digitized\.(png|xy)$", name):
            path.unlink()
            LOGGER.debug("Removed stale output: %s", path.name)
        elif re.match(rf"^{re.escape(stem)}_\d+\.peaks\.json$", name):
            path.unlink()
            LOGGER.debug("Removed stale output: %s", path.name)


def digitize_figure(
    context: FigureContext,
    *,
    skip_classification: bool = False,
    num_points: int = 2000,
) -> DigitizationResult | None:
    """Run the full deterministic digitization pipeline on one figure."""
    image_bgr = cv2.imread(str(context.image_path))
    if image_bgr is None:
        LOGGER.error("Could not load image: %s", context.image_path)
        return None

    classification = classify_from_context(context)
    if not skip_classification and not classification.is_xrd:
        LOGGER.info(
            "Skipping %s: not classified as XRD (confidence=%.2f)",
            context.image_path.name,
            classification.confidence,
        )
        return None

    plot_crop = crop_plot_area(image_bgr)
    base_stem = infer_output_stem(context.image_path)

    calibration = calibrate_axes(plot_crop, full_image_bgr=image_bgr)
    curves = digitize_xrd_curves(plot_crop, calibration, num_points=num_points)
    valid_curves = [curve for curve in curves if curve.two_theta]
    peaks = [detect_peaks(curve) for curve in valid_curves]

    warnings: list[str] = []
    warnings.extend(classification.warnings)
    warnings.extend(plot_crop.warnings)
    warnings.extend(calibration.warnings)
    for curve in valid_curves:
        warnings.extend(curve.warnings)

    height, width = image_bgr.shape[:2]
    panel = PlotPanel(index=1, bbox=(0, 0, width, height))
    panel_result = PanelDigitizationResult(
        panel=panel,
        plot_crop=plot_crop,
        calibration=calibration,
        curves=valid_curves,
        peaks=peaks,
        warnings=sorted(set(warnings)),
        output_stem=base_stem,
    )

    curve_scores = [0.8 if valid_curves else 0.1]
    confidence = float(
        np.mean(
            [
                classification.confidence,
                plot_crop.confidence,
                calibration.confidence,
                *curve_scores,
            ]
        )
    )

    return DigitizationResult(
        figure_context=context,
        classification=classification,
        plot_crop=plot_crop,
        panels=[panel_result],
        confidence=confidence,
        warnings=sorted(set(warnings)),
        output_stem=base_stem,
    )


def figure_output_dir(
    result: DigitizationResult,
    output_dir: Path | None = None,
) -> Path:
    """Directory for one figure's digitized outputs (e.g. sample_figures/figure_1/)."""
    base_dir = output_dir or result.figure_context.image_path.parent
    return base_dir / result.output_stem


def save_digitization_outputs(
    result: DigitizationResult,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Write one PNG, one .xy, and peaks JSON per figure."""
    base_dir = output_dir or result.figure_context.image_path.parent
    figure_dir = figure_output_dir(result, base_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = result.output_stem

    _cleanup_stale_outputs(figure_dir, stem)

    source_image = result.figure_context.image_path
    original_copy_path = figure_dir / source_image.name
    if source_image.exists() and not original_copy_path.exists():
        shutil.copy2(source_image, original_copy_path)

    panel_result = result.panels[0]
    curves = panel_result.curves
    calibration = panel_result.calibration

    xy_path = figure_dir / f"{stem}_digitized.xy"
    plot_path = figure_dir / f"{stem}_digitized.png"
    peaks_path = figure_dir / f"{stem}.peaks.json"

    save_multi_column_xy(curves, xy_path)
    plot_from_curves(
        curves,
        plot_path,
        calibration=calibration,
        title=result.figure_context.figure_id,
    )

    peak_entries: list[dict[str, Any]] = []
    for curve_index, curve in enumerate(curves, start=1):
        peak_list = panel_result.peaks[curve_index - 1] if curve_index - 1 < len(panel_result.peaks) else []
        peak_entries.append(
            {
                "curve_id": curve.curve_id,
                "color": curve.color,
                "label": curve.label,
                "num_peaks": len(peak_list),
                "peaks": _peaks_for_json(peak_list, calibration),
                "warnings": curve.warnings,
            }
        )

    peaks_payload = {
        "figure_id": result.figure_context.figure_id,
        "source_image": str(source_image),
        "num_curves": len(curves),
        "curves": peak_entries,
    }
    peaks_path.write_text(json.dumps(peaks_payload, indent=2), encoding="utf-8")

    panel_json_path = figure_dir / f"{stem}.json"
    panel_payload = {
        "figure_id": result.figure_context.figure_id,
        "plot_crop_bbox": list(panel_result.plot_crop.bbox),
        "calibration": {
            "method": calibration.method,
            "x_min": calibration.x_min,
            "x_max": calibration.x_max,
            "y_min": calibration.y_min,
            "y_max": calibration.y_max,
            "y_method": calibration.y_method,
            "confidence": calibration.confidence,
            "tick_pairs": [
                {"pixel_x": px, "two_theta": val}
                for px, val in calibration.tick_pairs
            ],
            "y_tick_pairs": [
                {"pixel_y": py, "intensity": val}
                for py, val in calibration.y_tick_pairs
            ],
        },
        "curves": [
            {
                "curve_id": curve.curve_id,
                "color": curve.color,
                "label": curve.label,
                "warnings": curve.warnings,
            }
            for curve in curves
        ],
        "outputs": {
            "xy": str(xy_path),
            "digitized_plot": str(plot_path),
            "peaks_json": str(peaks_path),
        },
        "warnings": panel_result.warnings,
    }
    panel_json_path.write_text(json.dumps(panel_payload, indent=2), encoding="utf-8")

    metadata_path = figure_dir / f"{stem}.metadata.json"
    metadata: dict[str, Any] = {
        "source_pdf": result.figure_context.source_pdf,
        "figure_id": result.figure_context.figure_id,
        "caption": result.figure_context.caption,
        "page": result.figure_context.page,
        "crop_bbox": result.figure_context.crop_bbox,
        "plot_crop_bbox": list(result.plot_crop.bbox),
        "classification": {
            "is_xrd": result.classification.is_xrd,
            "confidence": result.classification.confidence,
            "reasons": result.classification.reasons,
        },
        "confidence": result.confidence,
        "warnings": result.warnings,
        "outputs": {
            "figure_dir": str(figure_dir),
            "original_image": str(original_copy_path),
            "xy": str(xy_path),
            "digitized_plot": str(plot_path),
            "peaks_json": str(peaks_path),
            "json": str(panel_json_path),
            "metadata_json": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    saved_paths: dict[str, Path] = {
        "figure_dir": figure_dir,
        "original_image": original_copy_path,
        "xy": xy_path,
        "digitized_plot": plot_path,
        "peaks_json": peaks_path,
        "json": panel_json_path,
        "metadata_json": metadata_path,
    }

    LOGGER.info(
        "Saved digitized outputs for %s in %s (%d curve(s))",
        result.figure_context.image_path.name,
        figure_dir.name,
        len(curves),
    )
    return saved_paths


def _is_figure_output_dir(path: Path) -> bool:
    return path.is_dir() and bool(FIGURE_NUMBER_PATTERN.match(path.name))


def process_directory(
    input_dir: Path,
    *,
    pattern: str = "*.png",
    skip_classification: bool = False,
    num_points: int = 2000,
) -> list[DigitizationResult]:
    results: list[DigitizationResult] = []
    for image_path in sorted(input_dir.glob(pattern)):
        if _is_figure_output_dir(image_path.parent):
            continue
        context = load_figure_context(image_path)
        result = digitize_figure(
            context,
            skip_classification=skip_classification,
            num_points=num_points,
        )
        if result is None:
            continue
        save_digitization_outputs(result)
        results.append(result)
    return results


def process_figure_analysis_json(
    analysis_path: Path,
    *,
    skip_classification: bool = False,
    num_points: int = 2000,
) -> list[DigitizationResult]:
    """Process figures listed in a GROBID figure_analysis.json file."""
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    base_dir = analysis_path.parent
    results: list[DigitizationResult] = []

    for entry in payload:
        figure_path = entry.get("figure_path")
        if not figure_path:
            continue
        image_path = (base_dir / Path(figure_path).name) if not Path(figure_path).is_absolute() else Path(figure_path)
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        if not image_path.exists():
            alt = base_dir / "figures" / Path(figure_path).name
            if alt.exists():
                image_path = alt
            else:
                LOGGER.warning("Figure image not found: %s", figure_path)
                continue

        context = load_figure_context(
            image_path,
            caption=entry.get("caption"),
            figure_id=f"fig_{entry.get('figure', image_path.stem)}",
            source_pdf=str(base_dir.name),
        )
        result = digitize_figure(
            context,
            skip_classification=skip_classification,
            num_points=num_points,
        )
        if result is None:
            continue
        save_digitization_outputs(result, output_dir=base_dir)
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic XRD figure digitization pipeline.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="sample_figures",
        help="Image file, directory, or figure_analysis.json (default: sample_figures/)",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern when input is a directory (default: *.png)",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Digitize all images without XRD classification filter",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2000,
        help="Number of points in resampled digitized curve",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional caption for a single image input",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / args.input
        if candidate.exists():
            input_path = candidate
        else:
            parser.error(f"Input path not found: {args.input}")

    if input_path.is_dir():
        results = process_directory(
            input_path,
            pattern=args.pattern,
            skip_classification=args.skip_classification,
            num_points=args.num_points,
        )
        print(f"Digitized {len(results)} figure(s) in {input_path}")
        return

    if input_path.suffix.lower() == ".json":
        results = process_figure_analysis_json(
            input_path,
            skip_classification=args.skip_classification,
            num_points=args.num_points,
        )
        print(f"Digitized {len(results)} figure(s) from {input_path}")
        return

    context = load_figure_context(input_path, caption=args.caption)
    result = digitize_figure(
        context,
        skip_classification=args.skip_classification,
        num_points=args.num_points,
    )
    if result is None:
        print(f"Skipped or failed: {input_path}")
        return

    paths = save_digitization_outputs(result)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
