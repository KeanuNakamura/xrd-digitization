"""Tests for conditional ClipDrop cleaning decisions."""

from __future__ import annotations

import sys
import unittest

import cv2
import numpy as np

ROOT = __file__.rsplit("/tests/", 1)[0]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xrd_digitization.cleaning_decision import decide_cleaning_strategy  # noqa: E402


class CleaningDecisionTests(unittest.TestCase):
    def test_colored_curves_skip_clipdrop_despite_triage_hint(self) -> None:
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        # Blue / orange curves.
        for y in (60, 110, 160):
            cv2.line(img, (20, y), (280, y - 8), (180, 80, 40), 2)
            cv2.line(img, (20, y + 20), (280, y + 12), (40, 40, 200), 2)
        # Sparse black text away from curves.
        cv2.putText(img, "(200)", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        decision = decide_cleaning_strategy(img, triage_needs_clipdrop=True)
        self.assertEqual(decision.cleaning_strategy, "none")
        self.assertFalse(decision.needs_cleaning)
        self.assertEqual(decision.digitization_source, "original")

    def test_gray_curves_prefer_original(self) -> None:
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        for y in (70, 120, 160):
            cv2.line(img, (20, y), (280, y - 5), (40, 40, 40), 2)
        cv2.putText(img, "(200)", (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        decision = decide_cleaning_strategy(img, triage_needs_clipdrop=True)
        self.assertEqual(decision.cleaning_strategy, "none")
        self.assertFalse(decision.needs_cleaning)
        self.assertEqual(decision.digitization_source, "original")
        self.assertEqual(decision.curve_color_mode, "gray_black")


if __name__ == "__main__":
    unittest.main()
