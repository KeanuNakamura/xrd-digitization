"""Tests for inset plot-interior ClipDrop cleaning (axes preserved)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xrd_digitization.clipdrop_remove_text import (  # noqa: E402
    ClipdropError,
    assert_exterior_pixel_identical,
    call_clipdrop_remove_text,
    clean_figure_preserve_axes,
    default_clean_output_path,
    encode_image_png,
    match_crop_size,
    paste_crop_into_image,
)
from xrd_digitization.plot_interior_crop import (  # noqa: E402
    detect_axes_frame_bbox,
    inset_bbox,
    resolve_plot_interior_bbox,
)

FIGURES_WITH_TEXT = ROOT / "data" / "figures_with_text"


def _synthetic_framed_figure(
    *,
    width: int = 400,
    height: int = 300,
    frame: tuple[int, int, int, int] = (60, 30, 360, 250),
) -> np.ndarray:
    """White figure with a dark rectangular frame, exterior tick glyphs, interior labels."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = frame
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=2)
    # Exterior tick labels (must survive cleaning).
    cv2.putText(image, "50", (8, (y0 + y1) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "40", ((x0 + x1) // 2 - 10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Interior annotation (ClipDrop target).
    cv2.putText(
        image,
        "(104)",
        ((x0 + x1) // 2 - 20, (y0 + y1) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )
    # Simple curve.
    for x in range(x0 + 5, x1 - 5):
        y = y1 - 20 - int(8 * np.sin((x - x0) / 12.0))
        image[y : y + 2, x] = (0, 0, 0)
    return image


class PlotInteriorCropTests(unittest.TestCase):
    def test_inset_shrinks_frame(self) -> None:
        frame = (10, 20, 110, 220)
        out = inset_bbox(frame, inset_px=5, image_width=200, image_height=300)
        self.assertEqual(out, (15, 25, 105, 215))

    def test_manual_bbox_with_inset(self) -> None:
        image = _synthetic_framed_figure()
        interior = resolve_plot_interior_bbox(
            image, bbox=(60, 30, 360, 250), inset_px=3
        )
        self.assertEqual(interior.bbox, (63, 33, 357, 247))
        self.assertEqual(interior.frame_bbox, (60, 30, 360, 250))
        self.assertEqual(interior.method, "manual_inset")

    def test_manual_bbox_without_inset(self) -> None:
        image = _synthetic_framed_figure()
        interior = resolve_plot_interior_bbox(
            image,
            bbox=(60, 30, 360, 250),
            apply_inset_to_manual=False,
        )
        self.assertEqual(interior.bbox, (60, 30, 360, 250))

    def test_auto_detect_excludes_exterior_tick_region(self) -> None:
        frame = (60, 30, 360, 250)
        image = _synthetic_framed_figure(frame=frame)
        interior = resolve_plot_interior_bbox(image, inset_px=4)
        x0, y0, x1, y1 = interior.bbox
        # Exterior label pixels must not be inside the ClipDrop crop.
        self.assertGreater(x0, 20)
        self.assertLess(y1, 290)
        # Interior should still cover the in-plot annotation area.
        self.assertLess(x0, 180)
        self.assertGreater(x1, 220)


class PasteCompositeTests(unittest.TestCase):
    def test_exterior_pixels_identical(self) -> None:
        original = _synthetic_framed_figure()
        bbox = (63, 33, 357, 247)
        crop = original[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
        # Simulate ClipDrop whitening the crop.
        cleaned = np.full_like(crop, 255)
        composite, warnings = paste_crop_into_image(original, cleaned, bbox)
        self.assertEqual(warnings, [])
        assert_exterior_pixel_identical(original, composite, bbox)
        self.assertTrue(np.array_equal(composite[bbox[1] : bbox[3], bbox[0] : bbox[2]], cleaned))
        # Tick label region unchanged.
        self.assertTrue(np.array_equal(original[:, :50], composite[:, :50]))

    def test_resize_mismatch_crop(self) -> None:
        crop = np.zeros((50, 80, 3), dtype=np.uint8)
        resized, warnings = match_crop_size(crop, (40, 70))
        self.assertEqual(resized.shape[:2], (40, 70))
        self.assertTrue(any(w.startswith("clipdrop_crop_resized") for w in warnings))


class ClipdropClientTests(unittest.TestCase):
    def test_call_clipdrop_success(self) -> None:
        crop = np.full((40, 60, 3), 200, dtype=np.uint8)
        response_img = np.full((40, 60, 3), 250, dtype=np.uint8)
        png = encode_image_png(response_img)

        response = MagicMock()
        response.status_code = 200
        response.content = png
        response.text = ""

        def fake_post(url: str, **kwargs):  # noqa: ANN003
            self.assertIn("remove-text", url)
            self.assertEqual(kwargs["headers"]["x-api-key"], "test-key")
            self.assertIn("image_file", kwargs["files"])
            return response

        out = call_clipdrop_remove_text(crop, api_key="test-key", http_post=fake_post)
        self.assertEqual(out.shape, crop.shape)
        self.assertTrue(np.allclose(out.mean(), 250, atol=1))

    def test_call_clipdrop_http_error(self) -> None:
        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        response = MagicMock()
        response.status_code = 401
        response.content = b""
        response.text = "unauthorized"

        with self.assertRaises(ClipdropError):
            call_clipdrop_remove_text(
                crop,
                api_key="bad",
                http_post=lambda url, **kwargs: response,
            )

    def test_missing_api_key(self) -> None:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        old = os.environ.pop("CLIPDROP_API_KEY", None)
        try:
            with self.assertRaises(ClipdropError):
                call_clipdrop_remove_text(
                    crop, api_key=None, http_post=lambda **k: None
                )
        finally:
            if old is not None:
                os.environ["CLIPDROP_API_KEY"] = old


class CleanFigurePreserveAxesTests(unittest.TestCase):
    def test_dry_run_writes_identical_composite(self) -> None:
        image = _synthetic_framed_figure()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "figure_1.png"
            out = Path(tmp) / "figure_1_clean.png"
            cv2.imwrite(str(src), image)
            result = clean_figure_preserve_axes(
                src,
                output_path=out,
                inset_px=4,
                dry_run=True,
            )
            self.assertTrue(out.is_file())
            self.assertTrue(src.is_file())
            loaded = cv2.imread(str(out))
            self.assertEqual(loaded.shape, image.shape)
            assert_exterior_pixel_identical(image, loaded, result.interior.bbox)
            self.assertIn("clipdrop_dry_run", result.warnings)

    def test_mocked_api_pastes_cleaned_interior(self) -> None:
        image = _synthetic_framed_figure()
        interior = resolve_plot_interior_bbox(image, inset_px=4)

        def fake_remove(crop: np.ndarray) -> np.ndarray:
            cleaned = crop.copy()
            cleaned[:] = (240, 240, 240)
            return cleaned

        result = clean_figure_preserve_axes(
            image,
            output_path=None,
            inset_px=4,
            remove_text_fn=fake_remove,
        )
        x0, y0, x1, y1 = result.interior.bbox
        assert_exterior_pixel_identical(image, result.cleaned_bgr, result.interior.bbox)
        self.assertTrue(
            np.all(result.cleaned_bgr[y0:y1, x0:x1] == 240)
        )
        # Exterior tick strip unchanged.
        self.assertTrue(np.array_equal(image[:, :40], result.cleaned_bgr[:, :40]))

    def test_refuses_overwrite_original(self) -> None:
        image = _synthetic_framed_figure()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "figure_1.png"
            cv2.imwrite(str(src), image)
            with self.assertRaises(ClipdropError):
                clean_figure_preserve_axes(
                    src,
                    output_path=src,
                    dry_run=True,
                )

    def test_default_clean_output_path(self) -> None:
        self.assertEqual(
            default_clean_output_path(Path("figure_1.png")).name,
            "figure_1_clean.png",
        )
        self.assertEqual(
            default_clean_output_path(Path("figure_1.png"), dry_run=True).name,
            "figure_1_clean_dryrun.png",
        )


@unittest.skipUnless(FIGURES_WITH_TEXT.is_dir(), "figures_with_text not present")
class FiguresWithTextInteriorTests(unittest.TestCase):
    def test_figure_1_interior_excludes_axis_margins(self) -> None:
        path = FIGURES_WITH_TEXT / "figure_1.png"
        if not path.is_file():
            self.skipTest("figure_1.png missing")
        image = cv2.imread(str(path))
        frame, confidence, warnings, method = detect_axes_frame_bbox(image)
        interior = resolve_plot_interior_bbox(image, inset_px=4)
        self.assertGreater(confidence, 0.5)
        self.assertTrue(method.startswith("axis_lines"))
        fx0, fy0, fx1, fy1 = frame
        ix0, iy0, ix1, iy1 = interior.bbox
        self.assertGreater(ix0, fx0)
        self.assertGreater(iy0, fy0)
        self.assertLess(ix1, fx1)
        self.assertLess(iy1, fy1)
        # Left / bottom margins (tick labels) remain outside the crop.
        self.assertGreater(ix0, int(image.shape[1] * 0.05))
        self.assertLess(iy1, int(image.shape[0] * 0.96))

        result = clean_figure_preserve_axes(image, inset_px=4, dry_run=True)
        assert_exterior_pixel_identical(image, result.cleaned_bgr, result.interior.bbox)


if __name__ == "__main__":
    unittest.main()
