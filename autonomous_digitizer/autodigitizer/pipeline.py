"""End-to-end autonomous digitization pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from autodigitizer.axes.axis_calibrator import fit_axis_calibration, relative_axis_calibration
from autodigitizer.axes.shared_axes import inherit_shared_axes
from autodigitizer.axes.tick_detector import (
    assign_tick_values_from_analysis,
    detect_tick_positions,
    render_ticks,
)
from autodigitizer.config import Config
from autodigitizer.curves.curve_detector import extract_curves
from autodigitizer.models import (
    DigitizationStatus,
    FigureAnalysis,
    PipelineReport,
    PlotExtractionGuide,
    SubplotAnalysis,
    SubplotResult,
)
from autodigitizer.outputs.diagnostics import curve_coverage_metrics
from autodigitizer.outputs.reconstruction import (
    comparison_image,
    reconstruct_full_figure,
    reconstruct_subplot,
)
from autodigitizer.outputs.serializer import (
    write_digitized_figure_json,
    write_report,
    write_subplot_data,
)
from autodigitizer.processing.preprocessing import file_hash, image_hash, load_image, save_image
from autodigitizer.segmentation.plot_area_detector import detect_plot_area, render_plot_area
from autodigitizer.segmentation.subplot_detector import (
    SubplotCrop,
    detect_subplots,
    render_detection_overlay,
)
from autodigitizer.strategy.selector import select_strategy
from autodigitizer.vision.curve_guide import guide_to_plot_space, request_extraction_guide
from autodigitizer.vision.figure_analyzer import analyze_figure, offline_single_panel_analysis
from autodigitizer.vision.openai_client import OpenAIVisionClient
from autodigitizer.vision.qa import visual_qa
from autodigitizer.vision.subplot_analyzer import analyze_subplot

logger = logging.getLogger(__name__)


ProgressFn = Callable[[str], None]


class DigitizationPipeline:
    def __init__(self, config: Config, offline: bool = False):
        self.config = config
        self.offline = offline
        self.client: OpenAIVisionClient | None = None
        if not offline:
            self.client = OpenAIVisionClient(config)

    def run(self, image_path: Path | str, progress: ProgressFn | None = None) -> PipelineReport:
        def say(msg: str) -> None:
            # Console progress only (logging goes to pipeline.log when --debug).
            if progress:
                progress(msg)
            elif self.config.verbose:
                print(msg)
            logger.info(msg)

        image_path = Path(image_path)
        out = Path(self.config.output_dir)
        self._prepare_dirs(out)

        if self.config.debug:
            logging.getLogger("autodigitizer").setLevel(logging.DEBUG)

        say("1/8 load")
        image = load_image(image_path)
        save_image(out / "original" / "figure.png", image)
        cache_key = file_hash(image_path) + "_" + image_hash(image)

        report = PipelineReport(
            source_image=str(image_path),
            status=DigitizationStatus.FAILED,
        )

        # Stage 2
        say("2/8 figure analysis…")
        if self.offline or self.client is None:
            fig_analysis = offline_single_panel_analysis(image)
        else:
            fig_analysis = analyze_figure(self.client, image, cache_key=cache_key)
        (out / "analysis" / "figure_analysis.json").write_text(fig_analysis.model_dump_json(indent=2))
        report.figure_analysis = fig_analysis
        report.stages_completed.append("figure_analysis")
        say(f"   {fig_analysis.subplot_count} subplot(s)")

        if not fig_analysis.digitizable and not any(s.digitizable for s in fig_analysis.subplots):
            report.status = DigitizationStatus.NOT_DIGITIZABLE
            write_report(out / "report.json", report)
            say("not digitizable")
            return report

        # Stage 3
        say("3/8 segment")
        crops = detect_subplots(image, fig_analysis)
        for crop in crops:
            save_image(out / "subplots" / f"subplot_{crop.id}.png", crop.image)
        render_detection_overlay(image, crops, out / "debug" / "subplot_detection.png")
        report.stages_completed.append("subplot_segmentation")
        say(f"   {', '.join(c.id for c in crops)}")

        # Stage 4–10 per subplot (calibrate first pass, then shared axes, then curves)
        results: dict[str, SubplotResult] = {}
        analyses: dict[str, SubplotAnalysis] = {}

        say("4/8 calibrate")
        for crop in crops:
            res = SubplotResult(
                id=crop.id,
                crop_bbox_normalized=crop.bbox_normalized,
                status=DigitizationStatus.FAILED,
            )
            if not crop.digitizable:
                res.status = DigitizationStatus.NOT_DIGITIZABLE
                results[crop.id] = res
                continue

            analysis = self._analyze_subplot(crop, image, cache_key, fig_analysis)
            analyses[crop.id] = analysis
            (out / "analysis" / f"subplot_{crop.id}_analysis.json").write_text(
                analysis.model_dump_json(indent=2)
            )
            res.analysis = analysis
            if not analysis.digitizable:
                res.status = DigitizationStatus.NOT_DIGITIZABLE
                results[crop.id] = res
                continue

            plot_area = detect_plot_area(crop.image)
            res.plot_area = plot_area
            render_plot_area(crop.image, plot_area, out / "debug" / f"{crop.id}_plot_area.png")

            x_ticks, y_ticks = detect_tick_positions(crop.image, plot_area)
            x_ticks, y_ticks = assign_tick_values_from_analysis(
                x_ticks, y_ticks, analysis, plot_area
            )
            render_ticks(crop.image, plot_area, x_ticks, y_ticks, out / "debug" / f"{crop.id}_ticks.png")

            res.x_calibration = fit_axis_calibration(x_ticks, scale=analysis.x_axis.scale, axis_name="x")
            res.y_calibration = fit_axis_calibration(y_ticks, scale=analysis.y_axis.scale, axis_name="y")

            # Unlabeled y-axis (common for XRD intensity): relative units over plot height
            if res.y_calibration is None and res.plot_area is not None:
                ymin = analysis.y_axis.visible_min
                ymax = analysis.y_axis.visible_max
                if ymin is None:
                    ymin = 0.0
                if ymax is None or ymax <= ymin:
                    ymax = 1.0
                res.y_calibration = relative_axis_calibration(
                    float(res.plot_area.top),
                    float(res.plot_area.bottom),
                    data_at_top=float(ymax),
                    data_at_bottom=float(ymin),
                    scale=analysis.y_axis.scale,
                )
                res.warnings.append(
                    f"y-axis unlabeled; using relative intensity [{ymin:g}, {ymax:g}]"
                )
                logger.info("Subplot %s: relative y calibration [%g, %g]", crop.id, ymin, ymax)

            # Unlabeled x-axis fallback (rare)
            if res.x_calibration is None and res.plot_area is not None:
                xmin = analysis.x_axis.visible_min
                xmax = analysis.x_axis.visible_max
                if xmin is not None and xmax is not None and xmax > xmin:
                    from autodigitizer.axes.axis_calibrator import calibration_from_range

                    res.x_calibration = calibration_from_range(
                        float(res.plot_area.left),
                        float(res.plot_area.right),
                        float(xmin),
                        float(xmax),
                        scale=analysis.x_axis.scale,
                    )
                    res.x_calibration.calibration_type = "range_fallback"
                    res.warnings.append("x-axis ticks incomplete; used visible range fallback")

            results[crop.id] = res
            y_note = "rel" if res.y_calibration and res.y_calibration.calibration_type == "relative_unlabeled_axis" else ("ok" if res.y_calibration else "no")
            say(
                f"   {crop.id}: {analysis.estimated_curve_count} curve(s), "
                f"x={'ok' if res.x_calibration else 'no'} y={y_note}"
            )

        report.stages_completed.append("subplot_analysis_calibration")

        # Stage 6 shared axes
        say("5/8 shared axes")
        inherit_shared_axes(results, fig_analysis.axis_relationships)
        report.stages_completed.append("shared_axes")

        # Curve extraction + QA retries
        say("6/8 extract")
        crop_by_id = {c.id: c for c in crops}
        for sid, res in results.items():
            if res.status == DigitizationStatus.NOT_DIGITIZABLE:
                continue
            if res.x_calibration is None or res.y_calibration is None or res.plot_area is None:
                res.status = DigitizationStatus.FAILED
                res.warnings.append("Missing axis calibration after shared-axis resolution")
                say(f"   {sid}: fail (no calibration)")
                continue

            crop = crop_by_id[sid]
            analysis = res.analysis
            assert analysis is not None

            guide: PlotExtractionGuide | None = None
            if self.client is not None and not self.offline:
                try:
                    guide = request_extraction_guide(
                        self.client,
                        crop=crop.image,
                        analysis=analysis,
                        cache_key=cache_key,
                    )
                    (out / "analysis" / f"subplot_{sid}_extraction_guide_raw.json").write_text(
                        guide.model_dump_json(indent=2)
                    )
                    guide = guide_to_plot_space(guide, crop.image.shape, res.plot_area)
                    (out / "analysis" / f"subplot_{sid}_extraction_guide.json").write_text(
                        guide.model_dump_json(indent=2)
                    )
                    say(f"   {sid}: guide {len(guide.curves)} series")
                except Exception as exc:
                    logger.warning("Extraction guide failed for %s: %s", sid, exc)
                    res.warnings.append(f"extraction guide unavailable: {exc}")

            retry_hint = None
            best: SubplotResult | None = None
            best_score = -1.0
            for attempt in range(self.config.max_retries + 1):
                strategy = select_strategy(
                    analysis, attempt=attempt, retry_hint=retry_hint, guide=guide
                )
                res.strategy_name = strategy.name
                res.strategy_reason = strategy.reason
                res.attempts = attempt + 1

                bg_thr = strategy.params.get(
                    "background_threshold", self.config.background_luma_threshold
                )
                curves = extract_curves(
                    crop.image,
                    res.plot_area,
                    analysis,
                    res.x_calibration,
                    res.y_calibration,
                    samples=self.config.samples_per_curve,
                    strategy=strategy.name if strategy.name != "auto" else "auto",
                    color_clusters_override=strategy.params.get("color_clusters_override"),
                    background_threshold=bg_thr,
                    debug_dir=out / "debug",
                    subplot_id=sid,
                    guide=guide,
                )
                res.curves = curves
                metrics = curve_coverage_metrics(crop.image, res.plot_area, curves)
                say(
                    f"   {sid}: {strategy.name} → {len(curves)} curve(s) "
                    f"(cov {metrics.get('coverage', 0):.0%})"
                )

                recon = reconstruct_subplot(
                    res, out / "reconstructions" / f"reconstructed_{sid}.png"
                )
                if recon is not None and self.client is not None and not self.offline:
                    qa = visual_qa(
                        self.client,
                        subplot_id=sid,
                        original=crop.image,
                        reconstruction=recon,
                        cache_key=cache_key,
                        attempt=attempt,
                        local_metrics=metrics,
                    )
                    res.qa = qa
                else:
                    from autodigitizer.models import QAResult

                    passed = metrics.get("coverage", 0) >= 0.25 and len(curves) > 0
                    qa = QAResult(
                        passed=passed,
                        confidence=0.5,
                        issues=[] if passed else ["low local coverage"],
                        suggested_retry_strategy="guided_trace" if not passed else None,
                        metrics=metrics,
                    )
                    res.qa = qa

                score = (
                    (1.0 if qa.passed else 0.0)
                    + 0.5 * metrics.get("coverage", 0.0)
                    + 0.05 * len(curves)
                )
                if score >= best_score:
                    best_score = score
                    best = res.model_copy(deep=True)

                if qa.passed:
                    break
                retry_hint = qa.suggested_retry_strategy or "guided_trace"
                if retry_hint == "none":
                    break

            if best is not None:
                results[sid] = best
                res = best

            if res.curves:
                res.status = (
                    DigitizationStatus.SUCCESS
                    if (res.qa and res.qa.passed)
                    else DigitizationStatus.PARTIAL
                )
            else:
                res.status = DigitizationStatus.FAILED
            say(f"   {sid}: {res.status.value}")

        report.stages_completed.append("curve_extraction")

        # Outputs
        say("7/8 write outputs")
        ordered = [results[c.id] for c in crops if c.id in results]
        for res in ordered:
            write_subplot_data(out / "data", res)
            crop = crop_by_id[res.id]
            recon = reconstruct_subplot(res, out / "reconstructions" / f"reconstructed_{res.id}.png")
            if recon is not None:
                comparison_image(crop.image, recon, out / "comparisons" / f"comparison_{res.id}.png")

        layout = fig_analysis.layout
        reconstruct_full_figure(
            ordered,
            layout.rows if layout else None,
            layout.columns if layout else None,
            out / "reconstructions" / "reconstructed_full_figure.png",
        )

        write_digitized_figure_json(
            out / "digitized_figure.json",
            source_image=str(image_path),
            figure_analysis=fig_analysis.model_dump(),
            subplot_results=ordered,
        )

        report.subplots = ordered
        statuses = [r.status for r in ordered]
        if all(s == DigitizationStatus.NOT_DIGITIZABLE for s in statuses):
            report.status = DigitizationStatus.NOT_DIGITIZABLE
        elif any(s == DigitizationStatus.SUCCESS for s in statuses) and all(
            s in (DigitizationStatus.SUCCESS, DigitizationStatus.NOT_DIGITIZABLE) for s in statuses
        ):
            report.status = DigitizationStatus.SUCCESS
        elif any(s in (DigitizationStatus.SUCCESS, DigitizationStatus.PARTIAL) for s in statuses):
            report.status = DigitizationStatus.PARTIAL
        else:
            report.status = DigitizationStatus.FAILED

        if self.client:
            report.api_calls = self.client.api_calls
            report.cache_hits = self.client.cache_hits

        write_report(out / "report.json", report)
        report.stages_completed.append("outputs")
        say(f"8/8 done → {report.status.value}")
        return report

    def _analyze_subplot(
        self,
        crop: SubplotCrop,
        full: np.ndarray,
        cache_key: str,
        fig_analysis: FigureAnalysis,
    ) -> SubplotAnalysis:
        summary = next((s for s in fig_analysis.subplots if s.id == crop.id), None)
        if summary is None:
            from autodigitizer.models import SubplotSummary

            summary = SubplotSummary(id=crop.id, bbox_normalized=crop.bbox_normalized)

        if self.offline or self.client is None:
            return _offline_subplot_analysis(crop, summary)

        return analyze_subplot(
            self.client,
            subplot=summary,
            crop=crop.image,
            full_figure=full,
            cache_key=cache_key,
        )

    def _prepare_dirs(self, out: Path) -> None:
        for name in (
            "original",
            "analysis",
            "subplots",
            "data",
            "reconstructions",
            "comparisons",
            "debug",
        ):
            (out / name).mkdir(parents=True, exist_ok=True)


def _offline_subplot_analysis(crop: SubplotCrop, summary) -> SubplotAnalysis:
    """Deterministic analysis used when OpenAI is unavailable (tests / --offline)."""
    from autodigitizer.models import AxisInfo, AxisScale, CurveSeparation

    return SubplotAnalysis(
        subplot_id=crop.id,
        digitizable=True,
        confidence=0.4,
        plot_type=summary.plot_type or "line",
        x_axis=AxisInfo(label="x", scale=AxisScale.LINEAR, visible_min=0.0, visible_max=1.0, confidence=0.3),
        y_axis=AxisInfo(label="y", scale=AxisScale.LINEAR, visible_min=0.0, visible_max=1.0, confidence=0.3),
        estimated_curve_count=1,
        curve_separation=CurveSeparation.SINGLE,
    )
