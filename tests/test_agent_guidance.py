"""Tests for vision-agent metadata extraction and HTTP client."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xrd_digitization.agent_guidance import (  # noqa: E402
    call_vision_agent,
    extract_agent_metadata,
    run_agent_guidance,
    save_agent_metadata,
)


SAMPLE_PAYLOAD = {
    "image_width": 320,
    "image_height": 220,
    "coordinate_space": "original_image_pixels",
    "plot_bbox": [10, 20, 300, 200],
    "x_axis": {"min": 10, "max": 80},
    "y_axis": {"min": 0, "max": 250},
    "text_regions": [
        {
            "bbox": [100, 40, 140, 120],
            "type": "peak_annotation",
            "text": "(104)",
            "orientation": "vertical",
            "confidence": 0.95,
        },
        {"bbox": [-5, 10, 50, 30], "type": "label", "orientation": "horizontal"},
    ],
    "curve_count": 1,
    "curve_layout": "overlay",
    "approximate_peaks": [23.5, 33.1, 35.7],
    "approximate_curve": {
        "two_theta": [10, 40, 80],
        "intensity": [0.1, 1.0, 0.2],
    },
}


class AgentGuidanceTests(unittest.TestCase):
    def test_extract_normalizes_and_clamps(self) -> None:
        meta = extract_agent_metadata(SAMPLE_PAYLOAD, image_shape=(220, 320))
        self.assertEqual(meta.curve_layout, "overlay")
        self.assertEqual(meta.curve_count, 1)
        self.assertEqual(meta.approximate_peaks, [23.5, 33.1, 35.7])
        self.assertEqual(meta.x_axis.min, 10.0)
        self.assertEqual(meta.x_axis.max, 80.0)
        self.assertEqual(meta.image_width, 320)
        self.assertEqual(meta.image_height, 220)
        self.assertEqual(meta.coordinate_space, "original_image_pixels")
        self.assertEqual(meta.text_regions[0].text, "(104)")
        self.assertEqual(meta.text_regions[0].orientation, "vertical")
        # Negative x clamped to 0.
        self.assertGreaterEqual(meta.text_regions[1].bbox[0], 0.0)
        self.assertIsNotNone(meta.approximate_curve)
        assert meta.approximate_curve is not None
        self.assertEqual(len(meta.approximate_curve.two_theta), 3)

    def test_extract_defaults_bad_fields(self) -> None:
        meta = extract_agent_metadata(
            {
                "curve_layout": "mystery",
                "curve_count": "nope",
                "approximate_peaks": ["x", 12],
            },
            image_shape=(100, 100),
        )
        self.assertEqual(meta.curve_layout, "single")
        self.assertEqual(meta.curve_count, 1)
        self.assertEqual(meta.approximate_peaks, [12.0])

    def test_roundtrip_file(self) -> None:
        meta = extract_agent_metadata(SAMPLE_PAYLOAD, image_shape=(220, 320))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fig.agent.json"
            save_agent_metadata(meta, path)
            loaded = extract_agent_metadata(path, image_shape=(220, 320))
            self.assertEqual(loaded.to_dict()["approximate_peaks"], meta.approximate_peaks)

    def test_call_vision_agent_parses_mocked_http(self) -> None:
        image = np.full((40, 60, 3), 255, dtype=np.uint8)
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(SAMPLE_PAYLOAD),
                    }
                }
            ]
        }
        http_post = MagicMock(return_value=response)
        payload = call_vision_agent(
            image,
            api_key="test-key",
            http_post=http_post,
        )
        self.assertEqual(payload["curve_layout"], "overlay")
        http_post.assert_called_once()
        args, kwargs = http_post.call_args
        self.assertIn("/chat/completions", args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_run_agent_guidance_offline_json(self) -> None:
        image = np.full((220, 320, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.json"
            path.write_text(json.dumps(SAMPLE_PAYLOAD), encoding="utf-8")
            meta = run_agent_guidance(image, metadata_path=path)
            self.assertEqual(len(meta.text_regions), 2)
            self.assertEqual(meta.approximate_peaks[0], 23.5)


if __name__ == "__main__":
    unittest.main()
