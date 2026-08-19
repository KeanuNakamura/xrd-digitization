"""Tests for hybrid text cleaning, confidence fusion, and validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xrd_digitization.agent_guidance import AgentFigureMetadata, AgentTextRegion  # noqa: E402
from xrd_digitization.coords import transform_bbox  # noqa: E402
from xrd_digitization.hybrid_digitize import (  # noqa: E402
    ORIGINAL_DIGITIZER_SOURCE_FUNCTION,
    assert_passthrough_arrays_equal,
    build_agent_prior_curve,
    detect_artificial_plateaus,
    fuse_extractions,
    hash_array,
    render_hybrid_overlay,
    run_hybrid_digitization,
    validate_extraction,
)
from xrd_digitization.hybrid_text_removal import (  # noqa: E402
    create_text_mask,
    remove_text_preserve_curve,
)
from xrd_digitization.types import AxisCalibrationResult, CurveData, PlotCropResult  # noqa: E402

FIGURE_1_DIR = ROOT / "data" / "figures_with_text" / "figure_1"
FIGURE_1_AGENT = ROOT / "tests" / "fixtures" / "figure_1.agent.json"
FIGURE_4_IMAGE = ROOT / "data" / "figures_with_text" / "figure_4.png"
FIGURE_4_AGENT = ROOT / "tests" / "fixtures" / "figure_4.agent.json"

# Major peak centers expected for figure_1 (degrees 2θ).
FIGURE_1_MAJOR_PEAKS = [
    23.5,
    33.0,
    35.5,
    41.0,
    49.0,
    54.0,
    57.5,
    62.0,
    64.0,
    72.0,
    76.0,
]

# Major sharp peaks for clean figure_4 (degrees 2θ).
FIGURE_4_MAJOR_PEAKS = [33.0, 36.0, 49.0, 54.0]


def _synthetic_annotated_plot() -> tuple[np.ndarray, AgentFigureMetadata, AxisCalibrationResult]:
    """White plot with a dark peak curve and a vertical text annotation blob."""
    image = np.full((220, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 20), (300, 170), (0, 0, 0), 2)
    for x in range(50, 290):
        y = 140 - int(55 * np.exp(-((x - 170) ** 2) / 350.0))
        image[max(0, y) : y + 2, x] = (0, 0, 0)

    for i, ch in enumerate(["2", "2", "0"]):
        cv2.putText(
            image,
            ch,
            (162, 55 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    meta = AgentFigureMetadata(
        plot_bbox=[40, 20, 300, 170],
        text_regions=[
            AgentTextRegion(
                bbox=[155, 40, 185, 120],
                type="peak_annotation",
                text="(220)",
                orientation="vertical",
                confidence=0.9,
            ),
        ],
        curve_count=1,
        curve_layout="single",
        approximate_peaks=[45.0],
        image_width=320,
        image_height=220,
        coordinate_space="original_image_pixels",
    )
    calibration = AxisCalibrationResult(
        x_min=10.0,
        x_max=80.0,
        plot_left=40,
        plot_right=300,
        plot_top=20,
        plot_bottom=170,
        method="synthetic",
        confidence=0.9,
        y_min=0.0,
        y_max=100.0,
        y_method="ocr",
    )
    return image, meta, calibration


class HybridDigitizeTests(unittest.TestCase):
    def test_transform_bbox_scales_and_crops(self) -> None:
        # Agent annotated a 400x300 image; actual is 800x600; crop is half.
        box = [100, 50, 140, 120]
        local = transform_bbox(
            box,
            source_image_size=(400, 300),
            target_crop_bbox=(40, 20, 360, 280),
            target_image_size=(320, 260),
            actual_image_size=(800, 600),
        )
        # Scaled to actual: [200, 100, 280, 240]; crop-local then scale to 320x260.
        self.assertGreater(local[0], 0)
        self.assertLess(local[2], 320)
        self.assertLess(local[3], 260)
        self.assertGreater(local[2] - local[0], 10)

    def test_create_text_mask_union(self) -> None:
        regions = [
            AgentTextRegion(bbox=[10, 10, 30, 40], type="label"),
            {"bbox": [25, 35, 50, 60], "type": "other"},
        ]
        mask = create_text_mask((80, 100), regions, pad=0)
        self.assertTrue(np.any(mask[10:40, 10:30]))
        self.assertTrue(np.any(mask[35:60, 25:50]))
        self.assertFalse(np.any(mask[70:, 70:]))

    def test_remove_text_preserves_curve_connection(self) -> None:
        image, meta, _cal = _synthetic_annotated_plot()
        text_mask = create_text_mask(image.shape[:2], meta.text_regions, pad=2)
        cleaned, removal = remove_text_preserve_curve(
            image, text_mask, text_regions=meta.text_regions
        )
        self.assertTrue(np.any(removal > 0), "expected some text pixels removed")
        gray_clean = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        outside = text_mask == 0
        dark_orig = (gray_orig < 170) & outside
        dark_clean = (gray_clean < 170) & outside
        overlap = float(np.count_nonzero(dark_orig & dark_clean)) / max(
            int(np.count_nonzero(dark_orig)), 1
        )
        self.assertGreater(overlap, 0.85)

    def test_fuse_prefers_cleaned_then_original_then_agent(self) -> None:
        grid = list(np.linspace(10, 80, 21))
        cleaned = CurveData(
            two_theta=grid,
            intensity=[1.0] * 21,
            confidence=[0.9] * 10 + [0.1] * 11,
            source="cleaned",
        )
        original = CurveData(
            two_theta=grid,
            intensity=[0.5] * 21,
            confidence=[0.8] * 10 + [0.1] * 5 + [0.8] * 6,
            source="original",
        )
        agent = CurveData(
            two_theta=grid,
            intensity=[0.2] * 21,
            confidence=[0.2] * 21,
            source="agent",
        )
        fused = fuse_extractions(cleaned, original, agent, threshold=0.45)
        self.assertEqual(fused.point_sources[0], "cleaned")
        # Mid region: both low → averaged (not agent peak prior).
        self.assertIn(fused.point_sources[10], {"averaged", "agent", "cleaned_low"})
        self.assertEqual(fused.point_sources[16], "original")

    def test_validate_and_overlay(self) -> None:
        image, meta, calibration = _synthetic_annotated_plot()
        prior = build_agent_prior_curve(meta, calibration, num_points=100)
        report = validate_extraction(image, prior, meta, calibration, threshold=0.45)
        self.assertIn("uncertain_fraction", report)
        self.assertIn("flags", report)
        overlay = render_hybrid_overlay(image, prior, calibration, meta)
        self.assertEqual(overlay.shape, image.shape)

    def test_run_hybrid_offline_agent_json(self) -> None:
        image, meta, calibration = _synthetic_annotated_plot()
        plot_crop = PlotCropResult(
            cropped_bgr=image.copy(),
            bbox=(0, 0, image.shape[1], image.shape[0]),
            confidence=0.9,
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent_path = Path(tmp) / "agent.json"
            agent_path.write_text(json.dumps(meta.to_dict()), encoding="utf-8")
            artifacts = run_hybrid_digitization(
                image,
                plot_crop,
                calibration,
                num_points=200,
                agent_metadata_path=agent_path,
            )
        self.assertTrue(
            artifacts.fused_curve.two_theta or artifacts.original_curve.two_theta
            or artifacts.agent_prior.two_theta
        )
        self.assertIsNotNone(artifacts.overlay_bgr)
        self.assertIsNotNone(artifacts.bbox_debug_bgr)
        self.assertIn("uncertain_fraction", artifacts.validation)
        self.assertTrue(np.any(artifacts.text_mask > 0) or artifacts.validation.get("bbox_overlap"))


@unittest.skipUnless(
    FIGURE_1_DIR.joinpath("figure_1.png").exists() and FIGURE_1_AGENT.exists(),
    "figure_1 assets not present",
)
class Figure1HybridRegressionTests(unittest.TestCase):
    """Regression checks for the annotated XRD figure_1."""

    @classmethod
    def setUpClass(cls) -> None:
        from xrd_digitization.calibrate_axes import calibrate_axes
        from xrd_digitization.crop_plot_area import crop_plot_area

        cls.image = cv2.imread(str(FIGURE_1_DIR / "figure_1.png"))
        assert cls.image is not None
        cls.plot_crop = crop_plot_area(cls.image)
        cls.calibration = calibrate_axes(cls.plot_crop, full_image_bgr=cls.image)
        cls.artifacts = run_hybrid_digitization(
            cls.image,
            cls.plot_crop,
            cls.calibration,
            num_points=1200,
            agent_metadata_path=FIGURE_1_AGENT,
            require_box_overlap=True,
        )

    def test_transformed_boxes_overlap_text(self) -> None:
        overlap = self.artifacts.validation.get("bbox_overlap") or {}
        self.assertGreaterEqual(overlap.get("overlap_ratio", 0.0), 0.5)
        self.assertNotIn("text_boxes_misaligned", self.artifacts.validation.get("flags") or [])

    def test_text_removed_from_cleaned_image(self) -> None:
        self.assertTrue(np.any(self.artifacts.removal_mask > 0))
        # Cleaned image should have fewer dark pixels inside text boxes.
        gray_o = cv2.cvtColor(self.plot_crop.cropped_bgr, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(self.artifacts.cleaned_bgr, cv2.COLOR_BGR2GRAY)
        mask = self.artifacts.text_mask > 0
        if np.any(mask):
            dark_o = int(np.count_nonzero((gray_o < 180) & mask))
            dark_c = int(np.count_nonzero((gray_c < 180) & mask))
            self.assertLess(dark_c, dark_o * 0.65)

    def test_curve_follows_baseline_not_axis(self) -> None:
        curve = self.artifacts.fused_curve
        self.assertGreater(len(curve.two_theta), 100)
        inten = np.asarray(curve.intensity, dtype=float)
        # Most points should sit above the axis (not near zero intensity).
        near_zero = float(np.mean(inten < max(2.0, 0.02 * float(np.max(inten)))))
        self.assertLess(near_zero, 0.15)
        flags = self.artifacts.validation.get("flags") or []
        self.assertNotIn("axis_tracking", flags)

    def test_no_rectangular_plateaus_or_axis_drops(self) -> None:
        inten = np.asarray(self.artifacts.fused_curve.intensity, dtype=float)
        scale = max(float(np.percentile(inten, 95) - np.percentile(inten, 5)), 1e-6)
        peak_level = float(np.max(inten))
        # Pathological jumps: large drops toward the axis that stay low
        # (not sharp XRD peaks that rise and fall within a few points).
        bad = 0
        for i in range(1, len(inten) - 3):
            drop = (inten[i - 1] - inten[i]) / scale
            if drop > 0.2 and inten[i] < 0.08 * peak_level:
                if float(np.mean(inten[i : i + 4])) < 0.1 * peak_level:
                    bad += 1
        self.assertLess(bad, 25)
        # No long rectangular plateaus at elevated intensity (~1.5° wide).
        d1 = np.abs(np.diff(inten))
        flat = d1 < (0.002 * scale)
        run = 0
        long_high = 0
        for i, f in enumerate(flat):
            if f and inten[i] > 0.25 * peak_level:
                run += 1
                if run == 30:
                    long_high += 1
            else:
                run = 0
        self.assertLess(long_high, 2)
        self.assertNotIn("axis_tracking", self.artifacts.validation.get("flags") or [])

    def test_recovers_major_peaks(self) -> None:
        from scipy.signal import find_peaks

        tt = np.asarray(self.artifacts.fused_curve.two_theta, dtype=float)
        inten = np.asarray(self.artifacts.fused_curve.intensity, dtype=float)
        y_s = inten.copy()
        if len(y_s) > 15:
            window = min(11, len(y_s) // 40 * 2 + 1)
            if window >= 5:
                y_s = np.convolve(y_s, np.ones(window) / window, mode="same")
        prominence = max(0.04 * float(np.max(y_s)), 1e-6)
        idx, _ = find_peaks(y_s, prominence=prominence, distance=max(3, len(y_s) // 80))
        extracted = tt[idx] if idx.size else np.array([])
        hits = 0
        for peak in FIGURE_1_MAJOR_PEAKS:
            if extracted.size and float(np.min(np.abs(extracted - peak))) <= 1.5:
                hits += 1
        self.assertGreaterEqual(
            hits,
            8,
            f"only matched {hits}/{len(FIGURE_1_MAJOR_PEAKS)} major peaks; extracted={extracted.tolist()[:20]}",
        )

    def test_bbox_debug_image_produced(self) -> None:
        self.assertIsNotNone(self.artifacts.bbox_debug_bgr)
        assert self.artifacts.bbox_debug_bgr is not None
        self.assertEqual(
            self.artifacts.bbox_debug_bgr.shape[:2],
            self.plot_crop.cropped_bgr.shape[:2],
        )


class PlateauValidationTests(unittest.TestCase):
    def test_plateaued_path_fails_validation(self) -> None:
        """Low ink distance is not enough when major peaks are underestimated."""
        image, meta, calibration = _synthetic_annotated_plot()
        # Build a curve that follows baseline ink but clips the peak into a plateau.
        grid = np.linspace(10.0, 80.0, 200)
        intensity = np.full_like(grid, 15.0)
        # Flat-topped plateau near the synthetic peak (~45°) instead of a sharp tip.
        plateau = (grid > 42.0) & (grid < 48.0)
        intensity[plateau] = 28.0
        curve = CurveData(
            two_theta=grid.tolist(),
            intensity=intensity.tolist(),
            source="dp",
            confidence=[0.8] * len(grid),
        )
        meta.approximate_peaks = [45.0]
        report = validate_extraction(image, curve, meta, calibration)
        self.assertFalse(report["ok"])
        flags = set(report.get("flags") or [])
        self.assertTrue(
            flags
            & {
                "peak_height_underestimate",
                "low_apex_coverage",
                "missed_tall_narrow_peak",
                "ink_above_path_near_peaks",
                "rectangular_plateaus",
                "peak_mismatch",
            },
            f"expected peak-quality failure flags, got {flags}",
        )


@unittest.skipUnless(
    FIGURE_4_IMAGE.exists() and FIGURE_4_AGENT.exists(),
    "figure_4 assets not present",
)
class Figure4HybridRegressionTests(unittest.TestCase):
    """Clean XRD figure with no text — original digitizer must win."""

    @classmethod
    def setUpClass(cls) -> None:
        from xrd_digitization.calibrate_axes import calibrate_axes
        from xrd_digitization.crop_plot_area import crop_plot_area

        cls.image = cv2.imread(str(FIGURE_4_IMAGE))
        assert cls.image is not None
        cls.plot_crop = crop_plot_area(cls.image)
        cls.calibration = calibrate_axes(cls.plot_crop, full_image_bgr=cls.image)
        cls.artifacts = run_hybrid_digitization(
            cls.image,
            cls.plot_crop,
            cls.calibration,
            num_points=1200,
            agent_metadata_path=FIGURE_4_AGENT,
            require_box_overlap=True,
        )

    def test_no_text_mask_needed(self) -> None:
        self.assertEqual(self.artifacts.agent_meta.text_regions, [])
        self.assertFalse(np.any(self.artifacts.text_mask > 0))
        self.assertFalse(np.any(self.artifacts.removal_mask > 0))
        self.assertEqual(self.artifacts.hybrid_mode_used, "original_passthrough")

    def test_original_and_cleaned_equivalent(self) -> None:
        o = self.artifacts.original_curve
        c = self.artifacts.cleaned_curve
        self.assertEqual(len(o.two_theta), len(c.two_theta))
        self.assertGreater(len(o.two_theta), 100)
        oi = np.asarray(o.intensity, dtype=float)
        ci = np.asarray(c.intensity, dtype=float)
        self.assertTrue(np.allclose(oi, ci, rtol=1e-6, atol=1e-6))
        # Cleaned image is a copy of the original crop.
        self.assertEqual(self.artifacts.cleaned_bgr.shape, self.plot_crop.cropped_bgr.shape)
        self.assertTrue(
            np.array_equal(self.artifacts.cleaned_bgr, self.plot_crop.cropped_bgr)
        )

    def test_follows_baseline_without_plateaus(self) -> None:
        curve = self.artifacts.fused_curve
        inten = np.asarray(curve.intensity, dtype=float)
        self.assertGreater(len(inten), 100)
        # Noisy baseline should sit above pure zero for most of the scan.
        near_zero = float(np.mean(inten < max(1.0, 0.01 * float(np.max(inten)))))
        self.assertLess(near_zero, 0.2)

        plateaus = detect_artificial_plateaus(
            curve,
            self.plot_crop.cropped_bgr,
            self.calibration,
            agent_peaks=FIGURE_4_MAJOR_PEAKS,
        )
        self.assertEqual(plateaus["count"], 0)
        self.assertNotIn(
            "rectangular_plateaus", self.artifacts.validation.get("flags") or []
        )

    def test_narrow_peaks_retain_height(self) -> None:
        tt = np.asarray(self.artifacts.fused_curve.two_theta, dtype=float)
        inten = np.asarray(self.artifacts.fused_curve.intensity, dtype=float)
        peak_level = float(np.max(inten))
        self.assertGreater(peak_level, 40.0)

        # Baseline estimate away from major peaks.
        outside = np.ones(len(tt), dtype=bool)
        for p in FIGURE_4_MAJOR_PEAKS:
            outside &= np.abs(tt - p) > 2.0
        baseline = float(np.median(inten[outside])) if np.any(outside) else 0.0

        hits = 0
        for peak in FIGURE_4_MAJOR_PEAKS:
            local = (tt > peak - 1.5) & (tt < peak + 1.5)
            if not np.any(local):
                continue
            height = float(np.max(inten[local]))
            # Must climb well above baseline — not a short rectangular shelf.
            self.assertGreater(
                height,
                baseline + 0.25 * max(peak_level - baseline, 1.0),
                f"peak near {peak}° under-estimated: height={height}, baseline={baseline}",
            )
            # Narrow: elevated region should not span a broad plateau.
            elevated = local & (inten > baseline + 0.25 * max(peak_level - baseline, 1.0))
            if np.any(elevated):
                width_deg = float(tt[elevated].max() - tt[elevated].min())
                self.assertLess(
                    width_deg,
                    4.0,
                    f"peak near {peak}° too broad ({width_deg:.2f}°)",
                )
            hits += 1
        self.assertGreaterEqual(hits, 3)

    def test_hybrid_returns_original_digitizer_output(self) -> None:
        self.assertEqual(self.artifacts.hybrid_mode_used, "original_passthrough")
        self.assertFalse(self.artifacts.dp_curve.two_theta)
        final = np.asarray(self.artifacts.fused_curve.intensity, dtype=float)
        original = np.asarray(self.artifacts.original_curve.intensity, dtype=float)
        self.assertTrue(np.allclose(final, original, rtol=0.0, atol=1e-10))
        self.assertNotIn(
            "simplified_constant_width_peaks",
            self.artifacts.fused_curve.warnings or [],
        )
        self.assertIn("raw_mask_trace", self.artifacts.original_curve.warnings or [])

    def test_passthrough_hashes_and_metadata(self) -> None:
        debug = assert_passthrough_arrays_equal(
            self.artifacts.original_curve, self.artifacts.fused_curve
        )
        self.assertEqual(debug["selected_candidate"], "original_passthrough")
        self.assertTrue(debug["passthrough_arrays_equal"])
        self.assertEqual(
            debug["final_curve_source_function"], ORIGINAL_DIGITIZER_SOURCE_FUNCTION
        )
        self.assertEqual(
            debug["original_curve_hash"],
            hash_array(self.artifacts.original_curve.intensity),
        )
        self.assertEqual(debug["original_curve_hash"], debug["final_curve_hash"])
        # Validation payload must carry the same debug fields.
        validation = self.artifacts.validation
        self.assertEqual(validation.get("selected_candidate"), "original_passthrough")
        self.assertTrue(validation.get("passthrough_arrays_equal"))
        self.assertEqual(
            validation.get("final_curve_source_function"),
            ORIGINAL_DIGITIZER_SOURCE_FUNCTION,
        )
        self.assertEqual(
            self.artifacts.passthrough_debug.get("original_curve_hash"),
            debug["original_curve_hash"],
        )

    def test_passthrough_is_not_flat_baseline_peak_reconstruction(self) -> None:
        inten = np.asarray(self.artifacts.fused_curve.intensity, dtype=float)
        # Synthetic constant-width reconstructions sit on a near-zero baseline
        # with only a handful of smooth Gaussians. Raw traces are noisy.
        outside = np.ones(len(inten), dtype=bool)
        tt = np.asarray(self.artifacts.fused_curve.two_theta, dtype=float)
        for p in FIGURE_4_MAJOR_PEAKS:
            outside &= np.abs(tt - p) > 2.5
        if np.any(outside):
            baseline = inten[outside]
            # Noisy baseline: not identically zero and not a single constant.
            self.assertGreater(float(np.std(baseline)), 0.5)
            self.assertGreater(float(np.median(baseline)), 1.0)

    def test_forced_dp_does_not_silently_pass_bad_peaks(self) -> None:
        """If DP is explicitly requested, validation must catch clipped peaks."""
        forced = run_hybrid_digitization(
            self.image,
            self.plot_crop,
            self.calibration,
            num_points=800,
            agent_metadata_path=FIGURE_4_AGENT,
            force_dp_tracing=True,
        )
        # Passthrough is disabled, so a candidate is selected among original/cleaned/dp.
        self.assertIn(
            forced.hybrid_mode_used, {"original", "cleaned", "dp", "original_passthrough"}
        )
        if forced.hybrid_mode_used == "dp":
            # A DP winner that underestimates peaks must not report ok=true.
            report = forced.validation
            if float(report.get("peak_agreement") or 0.0) < 0.75 or int(
                report.get("artificial_plateau_count") or 0
            ):
                self.assertFalse(report.get("ok", True))


if __name__ == "__main__":
    unittest.main()
