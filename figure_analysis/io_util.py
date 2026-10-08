"""I/O helpers for figure analysis outputs."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping

from figure_analysis.analyzer import analysis_to_dataset_dict
from figure_analysis.schemas import FigureAnalysisResult

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def figure_stem(path: Path) -> str:
    return path.stem


def output_dir_for(output_root: Path, stem: str) -> Path:
    return output_root / stem


def analysis_json_path(output_root: Path, stem: str) -> Path:
    return output_dir_for(output_root, stem) / f"{stem}_analysis.json"


def summary_txt_path(output_root: Path, stem: str) -> Path:
    return output_dir_for(output_root, stem) / f"{stem}_summary.txt"


def copied_image_path(output_root: Path, stem: str, source: Path) -> Path:
    return output_dir_for(output_root, stem) / f"{stem}{source.suffix.lower()}"


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, record: Mapping[str, Any] | FigureAnalysisResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(record, FigureAnalysisResult):
        payload = {
            "is_xrd": record.is_xrd,
            "analysis": analysis_to_dataset_dict(record),
        }
    else:
        payload = dict(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def copy_source_image(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == source.resolve():
        return
    shutil.copy2(source, dest)


def load_analysis_if_valid(path: Path) -> FigureAnalysisResult | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        # Sidecar may be {is_xrd, analysis} or flat analysis payload.
        if "analysis" in data and "is_xrd" in data:
            is_xrd = bool(data.get("is_xrd"))
            analysis = data.get("analysis") or {}
            if not is_xrd:
                return FigureAnalysisResult(is_xrd=False)
            if not isinstance(analysis, dict):
                return None
            analysis = dict(analysis)
            analysis["is_xrd"] = True
            return FigureAnalysisResult.model_validate(analysis)

        data.pop("figure", None)
        data.pop("x_min", None)
        data.pop("x_max", None)
        data.pop("peak_assignments", None)
        data.pop("plots", None)
        if "is_xrd" not in data:
            data["is_xrd"] = True
        for curve in data.get("curves") or []:
            if "condition" not in curve and "label" in curve:
                curve["condition"] = curve.pop("label")
            curve.pop("crystallite_sizes", None)
            curve.pop("crystallite_sizes_nm", None)
            curve.pop("crystallite_size_nm", None)
            curve.pop("relative_intensity", None)
            if "peak_positions" not in curve and "peaks" in curve:
                positions = []
                for peak in curve.pop("peaks") or []:
                    pos = peak.get("position_2theta", peak.get("position"))
                    if pos is not None:
                        positions.append(pos)
                curve["peak_positions"] = positions
        if "trends" not in data and "observations" in data:
            data["trends"] = data.pop("observations")
        return FigureAnalysisResult.model_validate(data)
    except Exception as exc:
        logger.warning("Existing analysis invalid at %s: %s", path, exc)
        return None


def format_summary(result: FigureAnalysisResult) -> str:
    """Human-readable summary for optional .txt sidecar."""
    if not result.is_xrd:
        return "is_xrd: false\nanalysis: null\n"

    lines: list[str] = [
        "is_xrd: true",
        f"Sample: {result.sample or 'n/a'}",
        f"Curves: {len(result.curves)}",
        "",
    ]
    for curve in result.curves:
        lines.append(f"Curve: {curve.condition or '(unlabeled)'}")
        lines.append(f"  Peak positions: {curve.peak_positions}")
        lines.append(
            f"  Phases: {', '.join(curve.phases) if curve.phases else 'n/a'}"
        )
        if curve.peak_width:
            lines.append(f"  Peak width: {curve.peak_width}")
        if curve.fwhm is not None:
            lines.append(f"  FWHM (°): {curve.fwhm}")
        if curve.profile_function:
            lines.append(f"  Profile function: {curve.profile_function}")
        for lp in curve.lattice_parameters:
            lines.append(
                f"  Lattice ({lp.phase or 'n/a'}): "
                f"a={lp.a_A} b={lp.b_A} c={lp.c_A} "
                f"α={lp.alpha_deg} β={lp.beta_deg} γ={lp.gamma_deg} "
                f"V={lp.volume_A3} sg={lp.space_group}"
            )
        lines.append("")
    if result.fwhm is not None:
        lines.append(f"Figure FWHM (°): {result.fwhm}")
    if result.profile_function:
        lines.append(f"Figure profile function: {result.profile_function}")
    for lp in result.lattice_parameters:
        lines.append(
            f"Figure lattice ({lp.phase or 'n/a'}): "
            f"a={lp.a_A} b={lp.b_A} c={lp.c_A}"
        )
    if result.trends:
        lines.append("Trends:")
        for trend in result.trends:
            lines.append(f"  - {trend}")
    return "\n".join(lines).rstrip() + "\n"


def compact_jsonl_record(result: FigureAnalysisResult) -> dict[str, Any]:
    """Compact record for the global JSONL index."""
    if not result.is_xrd:
        return {"is_xrd": False, "curve_count": 0, "peak_count": 0, "phases": []}
    peak_count = sum(len(c.peak_positions) for c in result.curves)
    phases: list[str] = []
    for curve in result.curves:
        for phase in curve.phases:
            if phase not in phases:
                phases.append(phase)
    return {
        "is_xrd": True,
        "sample": result.sample,
        "curve_count": len(result.curves),
        "peak_count": peak_count,
        "phases": phases,
    }
