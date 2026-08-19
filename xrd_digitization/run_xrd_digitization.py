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
from xrd_digitization.hybrid_digitize import run_hybrid_digitization
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
    hybrid: bool = False,
    agent_json: Path | None = None,
) -> DigitizationResult | None:
    """Run the full digitization pipeline on one figure.

    When ``hybrid`` is True, a vision agent supplies text regions / peak priors,
    text is cleaned conservatively, and detailed pixel extractions are fused.
    """
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
    hybrid_payload: dict[str, Any] | None = None

    if hybrid:
        artifacts = run_hybrid_digitization(
            image_bgr,
            plot_crop,
            calibration,
            num_points=num_points,
            agent_metadata_path=agent_json,
        )
        valid_curves = (
            [artifacts.fused_curve] if artifacts.fused_curve.two_theta else []
        )
        hybrid_payload = {
            "agent_meta": artifacts.agent_meta.to_dict(),
            "validation": artifacts.validation,
            "cleaned_bgr": artifacts.cleaned_bgr,
            "text_mask": artifacts.text_mask,
            "removal_mask": artifacts.removal_mask,
            "overlay_bgr": artifacts.overlay_bgr,
            "bbox_debug_bgr": artifacts.bbox_debug_bgr,
            "original_curve": artifacts.original_curve,
            "cleaned_curve": artifacts.cleaned_curve,
            "dp_curve": artifacts.dp_curve,
            "agent_prior": artifacts.agent_prior,
            "hybrid_mode_used": artifacts.hybrid_mode_used,
            "candidate_scores": artifacts.candidate_scores,
            "passthrough_debug": artifacts.passthrough_debug,
        }
        if artifacts.validation.get("flags"):
            plot_crop.warnings = list(plot_crop.warnings) + [
                f"hybrid_{flag}" for flag in artifacts.validation["flags"]
            ]
    else:
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
    if hybrid and hybrid_payload is not None:
        uncertain = float(hybrid_payload["validation"].get("uncertain_fraction") or 0.0)
        curve_scores = [max(0.1, 0.85 - 0.5 * uncertain)]
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
        hybrid=hybrid_payload,
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

    # Write the selected raw curve arrays directly — never rebuild from peaks.json.
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

    if result.hybrid:
        agent_path = figure_dir / f"{stem}.agent.json"
        agent_path.write_text(
            json.dumps(result.hybrid["agent_meta"], indent=2), encoding="utf-8"
        )

        cleaned_path = figure_dir / f"{stem}_cleaned.png"
        overlay_path = figure_dir / f"{stem}_hybrid_overlay.png"
        bbox_debug_path = figure_dir / f"{stem}_bbox_debug.png"
        uncertainty_path = figure_dir / f"{stem}.hybrid_validation.json"
        text_mask_path = figure_dir / f"{stem}_text_mask.png"
        if result.hybrid.get("cleaned_bgr") is not None:
            cv2.imwrite(str(cleaned_path), result.hybrid["cleaned_bgr"])
        if result.hybrid.get("overlay_bgr") is not None:
            cv2.imwrite(str(overlay_path), result.hybrid["overlay_bgr"])
        if result.hybrid.get("bbox_debug_bgr") is not None:
            cv2.imwrite(str(bbox_debug_path), result.hybrid["bbox_debug_bgr"])
        if result.hybrid.get("text_mask") is not None:
            cv2.imwrite(str(text_mask_path), result.hybrid["text_mask"])

        # Candidate debug dumps: original vs final must match on passthrough.
        original_curve = result.hybrid.get("original_curve")
        final_curve = curves[0] if curves else None
        cand_original_xy = figure_dir / f"{stem}_candidate_original.xy"
        cand_original_png = figure_dir / f"{stem}_candidate_original.png"
        cand_final_xy = figure_dir / f"{stem}_candidate_final.xy"
        cand_final_png = figure_dir / f"{stem}_candidate_final.png"
        if original_curve is not None and getattr(original_curve, "two_theta", None):
            save_multi_column_xy([original_curve], cand_original_xy)
            plot_from_curves(
                [original_curve],
                cand_original_png,
                calibration=calibration,
                title=f"{result.figure_context.figure_id} candidate_original",
            )
        if final_curve is not None and final_curve.two_theta:
            save_multi_column_xy([final_curve], cand_final_xy)
            plot_from_curves(
                [final_curve],
                cand_final_png,
                calibration=calibration,
                title=f"{result.figure_context.figure_id} candidate_final",
            )

        validation = dict(result.hybrid.get("validation") or {})
        passthrough_debug = dict(result.hybrid.get("passthrough_debug") or {})
        if passthrough_debug:
            validation.update(passthrough_debug)
        validation.setdefault(
            "selected_candidate",
            result.hybrid.get("hybrid_mode_used"),
        )
        uncertainty_path.write_text(
            json.dumps(validation, indent=2),
            encoding="utf-8",
        )
        metadata["hybrid"] = {
            "validation": validation,
            "hybrid_mode_used": result.hybrid.get("hybrid_mode_used"),
            "selected_candidate": validation.get("selected_candidate"),
            "original_curve_hash": validation.get("original_curve_hash"),
            "final_curve_hash": validation.get("final_curve_hash"),
            "passthrough_arrays_equal": validation.get("passthrough_arrays_equal"),
            "final_curve_source_function": validation.get("final_curve_source_function"),
            "passthrough_debug": passthrough_debug,
            "agent_axis_ranges": {
                "x_axis": result.hybrid["agent_meta"].get("x_axis"),
                "y_axis": result.hybrid["agent_meta"].get("y_axis"),
            },
        }
        metadata["outputs"].update(
            {
                "agent_json": str(agent_path),
                "cleaned_image": str(cleaned_path),
                "hybrid_overlay": str(overlay_path),
                "bbox_debug": str(bbox_debug_path),
                "text_mask": str(text_mask_path),
                "hybrid_validation_json": str(uncertainty_path),
                "candidate_original_xy": str(cand_original_xy),
                "candidate_original_png": str(cand_original_png),
                "candidate_final_xy": str(cand_final_xy),
                "candidate_final_png": str(cand_final_png),
            }
        )

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
    if result.hybrid:
        saved_paths.update(
            {
                "agent_json": figure_dir / f"{stem}.agent.json",
                "cleaned_image": figure_dir / f"{stem}_cleaned.png",
                "hybrid_overlay": figure_dir / f"{stem}_hybrid_overlay.png",
                "bbox_debug": figure_dir / f"{stem}_bbox_debug.png",
                "text_mask": figure_dir / f"{stem}_text_mask.png",
                "hybrid_validation_json": figure_dir / f"{stem}.hybrid_validation.json",
                "candidate_original_xy": figure_dir / f"{stem}_candidate_original.xy",
                "candidate_original_png": figure_dir / f"{stem}_candidate_original.png",
                "candidate_final_xy": figure_dir / f"{stem}_candidate_final.xy",
                "candidate_final_png": figure_dir / f"{stem}_candidate_final.png",
            }
        )

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
    hybrid: bool = False,
    agent_json: Path | None = None,
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
            hybrid=hybrid,
            agent_json=agent_json,
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
    hybrid: bool = False,
    agent_json: Path | None = None,
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
            hybrid=hybrid,
            agent_json=agent_json,
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
        "--hybrid",
        action="store_true",
        help="Enable agent-guided text cleaning + confidence fusion",
    )
    parser.add_argument(
        "--agent-json",
        type=Path,
        default=None,
        help="Offline agent metadata JSON (skips vision API when present)",
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
            hybrid=args.hybrid,
            agent_json=args.agent_json,
        )
        print(f"Digitized {len(results)} figure(s) in {input_path}")
        return

    if input_path.suffix.lower() == ".json":
        results = process_figure_analysis_json(
            input_path,
            skip_classification=args.skip_classification,
            num_points=args.num_points,
            hybrid=args.hybrid,
            agent_json=args.agent_json,
        )
        print(f"Digitized {len(results)} figure(s) from {input_path}")
        return

    context = load_figure_context(input_path, caption=args.caption)
    result = digitize_figure(
        context,
        skip_classification=args.skip_classification,
        num_points=args.num_points,
        hybrid=args.hybrid,
        agent_json=args.agent_json,
    )
    if result is None:
        print(f"Skipped or failed: {input_path}")
        return

    paths = save_digitization_outputs(result)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
