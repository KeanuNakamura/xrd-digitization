"""Runtime configuration for the autonomous digitizer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Pipeline configuration loaded from CLI / environment."""

    openai_api_key: str = ""
    model: str = "gpt-4.1"
    output_dir: Path = Path("outputs/run")
    cache_dir: Path = Path(".cache")
    use_cache: bool = True
    max_retries: int = 2
    samples_per_curve: int = 1000
    debug: bool = False
    verbose: bool = True
    request_timeout_s: float = 120.0
    max_api_retries: int = 3
    temperature: float = 0.0

    # CV heuristics
    background_luma_threshold: int = 245
    axis_dark_threshold: int = 40
    min_curve_pixels: int = 80
    color_cluster_distance: float = 18.0

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        model = overrides.pop("model", None) or os.environ.get(
            "OPENAI_MODEL", "gpt-4.1"
        )
        api_key = overrides.pop("openai_api_key", None) or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        return cls(openai_api_key=api_key, model=model, **overrides)

    def require_api_key(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or place it in the environment."
            )
