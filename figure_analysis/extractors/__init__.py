"""Extractor prompt modules.

Add new modality extractors here (raman, xps, …) and register them in
``figure_analysis.prompts.build_user_prompt``.
"""

from __future__ import annotations

from figure_analysis.extractors import general, xrd

__all__ = ["general", "xrd"]
