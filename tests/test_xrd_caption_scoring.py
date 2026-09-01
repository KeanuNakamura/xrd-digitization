"""Regression tests for XRD caption scoring used by figure filtering."""

from __future__ import annotations

import unittest

from legacy.pdf_parser import score_xrd_text


CAPTION_XRD_THRESHOLD = 2.5


class XrdCaptionScoringTests(unittest.TestCase):
    def test_diffraction_planes_caption_is_xrd(self) -> None:
        # Scherrer_equation Figure 6: clearly an XRD figure, but historically
        # missed because scoring only rewarded "diffraction pattern"/XRD.
        caption = (
            "Figure 6 . Figure 6. All of the diffraction planes of HA "
            "firing at 900˚C."
        )
        score = score_xrd_text(caption)
        self.assertGreaterEqual(score, CAPTION_XRD_THRESHOLD)
        self.assertGreater(score, score_xrd_text("Figure 1. Modified scherrer equation plot."))

    def test_diffraction_peaks_caption_is_xrd(self) -> None:
        caption = "Figure 4. Diffraction peaks of the calcined powder."
        self.assertGreaterEqual(score_xrd_text(caption), CAPTION_XRD_THRESHOLD)

    def test_explicit_xrd_caption_still_scores_high(self) -> None:
        caption = (
            "Figure 2. Patterns of XRD analysis related to natural HA."
        )
        self.assertGreaterEqual(score_xrd_text(caption), CAPTION_XRD_THRESHOLD)

    def test_scherrer_plot_caption_is_not_caption_xrd(self) -> None:
        # Context may still mark these is_likely_xrd; caption gate should not.
        for caption in (
            "Figure 1. Modified scherrer equation plot.",
            "Figure 3. Linear plots of modified scherrer equation "
            "and obtained intercepts for different firings of ha.",
        ):
            self.assertLess(
                score_xrd_text(caption),
                CAPTION_XRD_THRESHOLD,
                msg=caption,
            )

    def test_bare_diffraction_without_figure_signal_stays_below_threshold(
        self,
    ) -> None:
        # Avoid classifying every mention of electron diffraction as caption XRD.
        self.assertLess(
            score_xrd_text("Electron diffraction image of the sample"),
            CAPTION_XRD_THRESHOLD,
        )


if __name__ == "__main__":
    unittest.main()
