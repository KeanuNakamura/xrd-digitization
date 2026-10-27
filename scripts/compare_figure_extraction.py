#!/usr/bin/env python3
"""
Compare XRD figure extraction before/after (hoje_v2 vs hoje_v3).

Uses ``output/XRD fig extract.xlsx`` as the ground-truth list of documented
errors. Writes a presentation-ready HTML report plus CSV/JSON summaries under
``output/hoje_v3_comparison/``.

Usage:
  python scripts/compare_figure_extraction.py
  python scripts/compare_figure_extraction.py --before output/hoje_v2 --after output/hoje_v3
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE = ROOT / "output" / "hoje_v2"
DEFAULT_AFTER = ROOT / "output" / "hoje_v3"
DEFAULT_XLSX = ROOT / "output" / "XRD fig extract.xlsx"
DEFAULT_OUT = ROOT / "output" / "hoje_v3_comparison"

THUMB_MAX = 420
SKIP_NOTE = re.compile(r"^\s*no xrd figure\s*$", re.IGNORECASE)

FIG_NUM = r"[A-Za-z]?\d+[A-Za-z]?(?:_\d+)?"
# Prefer Extended/Supplementary/Scheme before bare Fig, so "Extended Data Fig"
# is not swallowed by the shorter Fig alternative.
FIG_REF = (
    r"(?:Extended\s+Data\s+Fig(?:ure)?s?\.?|Supplementary\s+Fig(?:ure)?s?\.?|"
    r"Scheme|Figs?\.?|Figures?)"
)


@dataclass
class DocumentedIssue:
    paper_number: int
    doi: str | None
    folder: str
    source: str  # XRD | nonXRD
    issue_type: str
    figure_label: str | None
    expected_stem: str | None
    observed_stem: str | None
    raw_clause: str
    needs_visual: bool = False


@dataclass
class IssueResult:
    issue: DocumentedIssue
    status: str  # fixed | improved | unchanged | worsened | needs_visual | not_applicable
    confidence: str  # automated | visual_review
    rationale: str
    v2_files: list[str] = field(default_factory=list)
    v3_files: list[str] = field(default_factory=list)
    v2_thumb: str | None = None
    v3_thumb: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def normalize_doi(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"10\.\d{4,9}/[^\s\#]+", str(text))
    if not match:
        return None
    doi = match.group(0)
    doi = re.split(r"(?i)digital|object|identifier|#:~", doi)[0]
    return doi.lower().rstrip("/.,;)\"'")


def list_paper_dirs(root: Path) -> list[str]:
    """Match spreadsheet numbering: same order as ``ls`` (not Python sorted)."""
    raw = subprocess.check_output(["ls"], cwd=str(root), text=True)
    names = []
    for line in raw.splitlines():
        name = line.strip()
        if not name or name.startswith("."):
            continue
        path = root / name
        if path.is_dir():
            names.append(name)
    return names


def collect_figure_pngs(paper_dir: Path) -> dict[str, Path]:
    """Map stem -> path, preferring xrd_figures over non_xrd_figures over figures."""
    found: dict[str, Path] = {}
    for sub in ("figures", "non_xrd_figures", "xrd_figures"):
        folder = paper_dir / sub
        if not folder.is_dir():
            continue
        for png in sorted(folder.glob("*.png")):
            found[png.stem] = png
    return found


def stem_kind_and_number(stem: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(?P<kind>(?:ed_|supp_)?fig)_?(?P<num>\d+[a-z]?)(?:_(?P<rest>.+))?$",
        stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group("kind").lower(), match.group("num").lower()


def label_to_stem(label: str | None) -> str | None:
    if not label:
        return None
    text = normalize_whitespace(label)
    ed = re.match(
        rf"(?i)^Extended\s+Data\s+Fig(?:ure)?s?\.?\s*(?P<n>{FIG_NUM})$",
        text,
    )
    if ed:
        return f"ed_fig_{ed.group('n').lower()}"
    supp = re.match(
        rf"(?i)^Supplementary\s+Fig(?:ure)?s?\.?\s*(?P<n>{FIG_NUM})$",
        text,
    )
    if supp:
        return f"supp_fig_{supp.group('n').lower()}"
    scheme = re.match(rf"(?i)^Scheme\s*(?P<n>{FIG_NUM})$", text)
    if scheme:
        return f"scheme_{scheme.group('n').lower()}"
    fig = re.match(rf"(?i)^Figs?\.?\s*(?P<n>{FIG_NUM})$", text)
    if fig:
        return f"fig_{fig.group('n').lower()}"
    return None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _kind_label_stem(kind_raw: str, num: str) -> tuple[str, str]:
    kind = kind_raw.lower()
    num_l = num.lower()
    if "extended" in kind:
        return f"Extended Data Fig. {num}", f"ed_fig_{num_l}"
    if "supplementary" in kind:
        return f"Supplementary Fig. {num}", f"supp_fig_{num_l}"
    if "scheme" in kind:
        return f"Scheme {num}", f"scheme_{num_l}"
    return f"Fig. {num}", f"fig_{num_l}"


def extract_figure_list(blob: str) -> list[tuple[str, str]]:
    """
    Expand phrases like ``Figs. 1, 5, and 6`` or ``Figs. 1–3`` into labels/stems.
    """
    text = normalize_whitespace(blob)
    refs: list[tuple[str, str]] = []

    # Range: Figs. 1–3 / Figs. 1-3
    range_match = re.search(
        rf"(?i)({FIG_REF})\s*(\d+)\s*[–-]\s*(\d+)",
        text,
    )
    if range_match:
        kind, a, b = range_match.group(1), int(range_match.group(2)), int(range_match.group(3))
        for num in range(min(a, b), max(a, b) + 1):
            refs.append(_kind_label_stem(kind, str(num)))
        return refs

    # Leading kind then comma/and list: "Figs. 1, 5, and 6" / "Figs. 4 and 5"
    list_match = re.match(
        rf"(?i)^({FIG_REF})\s*(.+)$",
        text,
    )
    if list_match:
        kind, rest = list_match.group(1), list_match.group(2)
        # Strip trailing parentheticals like (nonXRD)
        rest = re.sub(r"\([^)]*\)", "", rest)
        nums = re.findall(r"\d+[a-z]?(?:_\d+)?", rest, flags=re.IGNORECASE)
        # Ignore years-like noise
        nums = [n for n in nums if len(re.sub(r"\D", "", n)) <= 3]
        if nums:
            for num in nums:
                refs.append(_kind_label_stem(kind, num))
            return refs

    # Single ref
    single = re.search(rf"(?i)({FIG_REF})\s*({FIG_NUM})", text)
    if single:
        refs.append(_kind_label_stem(single.group(1), single.group(2)))
    return refs


def split_note_clauses(note: str) -> list[str]:
    """Split a free-text notes field into evaluable clauses."""
    text = normalize_whitespace(note)
    # Deduplicate accidental repeated paste in some cells.
    if "incorrectFig." in text:
        text = text.split("incorrectFig.")[0]
        if not text.lower().endswith("incorrect"):
            text = text.rstrip() + " incorrect"
        text = normalize_whitespace(text)
    # Also collapse duplicated repeated sentences.
    for marker in (
        "remaining Extended Data figure labels incorrect",
        "Extended Data Fig. 3 extraction failed",
    ):
        if text.count(marker) > 1:
            first = text.find(marker)
            text = text[: first + len(marker)]
            text = normalize_whitespace(text)

    parts = re.split(
        r";|(?<!\d)\.(?=\s+(?:Fig|Figs|Extended|Scheme|image|all|Good|Extracted|remaining))",
        text,
    )
    clauses = []
    for part in parts:
        clause = normalize_whitespace(part).strip(" .;")
        if clause:
            clauses.append(clause)
    return clauses


def classify_clause(clause: str) -> list[tuple[str, str | None, str | None, bool]]:
    """
    Return list of (issue_type, figure_label, observed_label_or_detail, needs_visual).
    """
    c = clause.strip()
    out: list[tuple[str, str | None, str | None, bool]] = []

    # Generic paper-level numbering / ED labels
    if re.search(r"(?i)extracted figure numbers incorrect", c):
        out.append(("wrong_number", None, None, True))
        return out
    if re.search(r"(?i)all extended data figure labels incorrectly", c):
        out.append(("extended_data_labels", "Extended Data", None, True))
        return out
    if re.search(r"(?i)extended data figure (?:numbers|labels) incorrect", c):
        out.append(("extended_data_labels", "Extended Data", None, True))
        return out
    if re.search(r"(?i)remaining extended data figure labels incorrect", c):
        out.append(("extended_data_labels", "Extended Data", None, True))
        return out
    if re.search(r"(?i)good inclusion of extended data", c):
        out.append(("extended_data_labels", "Extended Data", "page_numbers_ok", True))
        return out

    # "Fig. 6 extracted as both Fig. 6 and Fig. 7_2"
    m = re.search(
        rf"(?i)({FIG_REF}\s*{FIG_NUM})\s+extracted as both\s+(.+)",
        c,
    )
    if m:
        left = extract_figure_list(m.group(1))
        right = extract_figure_list(m.group(2))
        label = left[0][0] if left else None
        # Store observed stems joined for duplicate evaluator
        obs = ",".join(stem for _, stem in right) if right else m.group(2)
        out.append(("duplicate", label, obs, False))
        return out

    # "Figs. 7 and 8 extracted as Figs. 7_2 and 8_2"
    # "Fig. 1 extracted as Fig. 1_2"
    # "Extended Data Fig. 1 extracted as Fig. 1"
    # "Scheme 1 extracted as Fig. 1"
    m = re.search(
        rf"(?i)(.+?)\s+extracted as\s+(.+)",
        c,
    )
    if m and "both" not in c.lower():
        left_refs = extract_figure_list(m.group(1))
        right_refs = extract_figure_list(m.group(2))
        if left_refs and right_refs:
            if len(left_refs) == len(right_refs):
                for (lab, _), (obs_lab, _) in zip(left_refs, right_refs):
                    out.append(("wrong_number", lab, obs_lab, False))
                return out
            for lab, _ in left_refs:
                out.append(("wrong_number", lab, right_refs[0][0], False))
            return out
        if left_refs and re.search(r"(?i)figs?\.\s*.+,", m.group(2)):
            out.append(("duplicate", left_refs[0][0], m.group(2), False))
            return out

    # Failed extractions
    m = re.search(rf"(?i)(.+?)\s+extraction failed", c)
    if m:
        blob = m.group(1)
        blob = re.sub(r"(?i)^image labeled\s+", "", blob)
        refs = extract_figure_list(blob)
        if refs:
            for lab, _ in refs:
                out.append(("missing", lab, None, False))
            return out

    # image labeled Fig. X incorrectly extracted
    m = re.search(rf"(?i)image labeled\s+({FIG_REF}\s*{FIG_NUM})", c)
    if m and "incorrectly" in c.lower():
        refs = extract_figure_list(m.group(1))
        for lab, _ in refs:
            out.append(("incorrect", lab, None, True))
        return out

    # Partial / over / under / slightly cropped
    m = re.search(
        rf"(?i)(.+?)\s+"
        rf"(partially extracted|over-extracted|under-extracted|slightly cropped)",
        c,
    )
    if m:
        kind_map = {
            "partially extracted": "partial",
            "over-extracted": "over",
            "under-extracted": "partial",
            "slightly cropped": "partial",
        }
        issue_type = kind_map[m.group(2).lower()]
        refs = extract_figure_list(m.group(1))
        if refs:
            for lab, _ in refs:
                out.append((issue_type, lab, None, True))
            return out

    # "Fig. 2 incorrectly extracted" / "Fig. 34 incorrectly extracted" / "Fig. 1_2 ..."
    m = re.search(rf"(?i)({FIG_REF}\s*{FIG_NUM})\s+incorrectly extracted", c)
    if m:
        refs = extract_figure_list(m.group(1))
        for lab, _ in refs:
            out.append(("incorrect", lab, None, True))
        return out

    out.append(("other", None, None, True))
    return out


def parse_spreadsheet(
    xlsx: Path,
    before_dirs: list[str],
) -> tuple[pd.DataFrame, list[DocumentedIssue]]:
    df = pd.read_excel(xlsx)
    issues: list[DocumentedIssue] = []

    for _, row in df.iterrows():
        number = int(row["Number"])
        doi = normalize_doi(str(row.get("reference") or ""))
        folder = before_dirs[number - 1] if 1 <= number <= len(before_dirs) else ""
        note_pairs = []
        xrd_note = row.get("Notes XRD figure")
        non_note = row.get("Notes nonXRD figure")
        if pd.notna(xrd_note) and str(xrd_note).strip():
            if not SKIP_NOTE.match(str(xrd_note)):
                note_pairs.append(("XRD", str(xrd_note).strip()))
        if pd.notna(non_note) and str(non_note).strip():
            note_pairs.append(("nonXRD", str(non_note).strip()))

        for source, note in note_pairs:
            for clause in split_note_clauses(note):
                for issue_type, label, observed, needs_visual in classify_clause(clause):
                    expected = label_to_stem(label) if label else None
                    observed_stem = None
                    if observed:
                        stems = re.findall(
                            r"(?i)(?:ed_|supp_)?fig_\d+[a-z]?(?:_\d+)?",
                            observed,
                        )
                        if stems:
                            observed_stem = ",".join(s.lower() for s in stems)
                        else:
                            observed_stem = label_to_stem(observed)
                            if not observed_stem:
                                m = re.search(
                                    rf"(?i)Fig\.?\s*({FIG_NUM})",
                                    observed,
                                )
                                if m:
                                    observed_stem = f"fig_{m.group(1).lower()}"
                    issues.append(
                        DocumentedIssue(
                            paper_number=number,
                            doi=doi,
                            folder=folder,
                            source=source,
                            issue_type=issue_type,
                            figure_label=label,
                            expected_stem=expected,
                            observed_stem=observed_stem,
                            raw_clause=clause,
                            needs_visual=needs_visual,
                        )
                    )
    return df, issues


def find_related_files(
    files: dict[str, Path],
    stem: str | None,
    label: str | None = None,
) -> list[Path]:
    if not files:
        return []
    hits: list[Path] = []
    if stem:
        # exact and page variants
        for key, path in files.items():
            if key == stem or key.startswith(stem + "_page_"):
                hits.append(path)
            elif stem.startswith("fig_") and key.startswith(stem + "_"):
                # fig_1_2 collision, but not fig_10 for fig_1
                rest = key[len(stem) + 1 :]
                if rest.isdigit() or rest.startswith("page_"):
                    hits.append(path)
        # Also accept ed_/supp_ equivalents for canonical stems only
        # (do not treat fig_1 as a hit when searching for collision fig_1_2).
        if re.match(r"^(?:ed_|supp_)?fig_\d+[a-z]?$", stem, flags=re.IGNORECASE):
            kind_num = stem_kind_and_number(stem)
            if kind_num:
                _kind, num = kind_num
                for alt_kind in ("fig", "ed_fig", "supp_fig"):
                    alt = f"{alt_kind}_{num}"
                    for key, path in files.items():
                        if key == alt or key.startswith(alt + "_page_"):
                            if path not in hits:
                                hits.append(path)
    if not hits and label:
        soft = label_to_stem(label)
        if soft and soft != stem:
            return find_related_files(files, soft, None)
    return hits


def image_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        w, h = img.size
    return {"width": w, "height": h, "area": w * h, "path": str(path)}


def thumb_data_uri(path: Path, max_side: int = THUMB_MAX) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def pick_primary(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    # Prefer exact stem without _2 / page suffixes, else largest area.
    def score(p: Path) -> tuple[int, int]:
        stem = p.stem
        exact = 1 if re.match(r"^(?:ed_|supp_)?fig_\d+[a-z]?$", stem, re.I) else 0
        try:
            area = image_metrics(p)["area"]
        except OSError:
            area = 0
        return exact, area

    return sorted(paths, key=score, reverse=True)[0]


def evaluate_issue(
    issue: DocumentedIssue,
    v2_files: dict[str, Path],
    v3_files: dict[str, Path],
) -> IssueResult:
    v2_hits = find_related_files(v2_files, issue.expected_stem or issue.observed_stem, issue.figure_label)
    v3_hits = find_related_files(v3_files, issue.expected_stem or issue.observed_stem, issue.figure_label)
    # For wrong_number, also look at observed wrong stem in v2
    if issue.issue_type == "wrong_number" and issue.observed_stem:
        v2_wrong = find_related_files(v2_files, issue.observed_stem)
        v3_wrong = find_related_files(v3_files, issue.observed_stem)
        if not v2_hits:
            v2_hits = v2_wrong
    else:
        v2_wrong, v3_wrong = [], []

    v2_names = [p.name for p in v2_hits]
    v3_names = [p.name for p in v3_hits]
    metrics: dict[str, Any] = {}

    def attach_thumbs(status: str, rationale: str, confidence: str) -> IssueResult:
        v2_primary = pick_primary(v2_hits) or pick_primary(v2_wrong)
        v3_primary = pick_primary(v3_hits) or pick_primary(
            find_related_files(v3_files, issue.expected_stem, issue.figure_label)
        )
        # For missing→fixed, v3 may be the only image
        if issue.issue_type == "missing" and not v3_primary:
            v3_primary = pick_primary(v3_hits)
        try:
            v2_thumb = thumb_data_uri(v2_primary) if v2_primary else None
        except OSError:
            v2_thumb = None
        try:
            v3_thumb = thumb_data_uri(v3_primary) if v3_primary else None
        except OSError:
            v3_thumb = None
        if v2_primary:
            metrics["v2"] = image_metrics(v2_primary)
        if v3_primary:
            metrics["v3"] = image_metrics(v3_primary)
        return IssueResult(
            issue=issue,
            status=status,
            confidence=confidence,
            rationale=rationale,
            v2_files=v2_names or [p.name for p in v2_wrong],
            v3_files=v3_names or [p.name for p in find_related_files(v3_files, issue.expected_stem)],
            v2_thumb=v2_thumb,
            v3_thumb=v3_thumb,
            metrics=metrics,
        )

    itype = issue.issue_type

    if itype == "wrong_number":
        expected = issue.expected_stem
        observed = issue.observed_stem
        v3_expected = find_related_files(v3_files, expected) if expected else []
        v3_obs = find_related_files(v3_files, observed) if observed else []
        v2_expected = find_related_files(v2_files, expected) if expected else []
        v2_obs = find_related_files(v2_files, observed) if observed else []

        # Extended Data mislabeled as main fig
        if expected and expected.startswith("ed_fig"):
            has_ed = any(n.startswith("ed_fig") for n in v3_files)
            still_main_collision = bool(v3_obs) and not has_ed
            if has_ed and not still_main_collision:
                return attach_thumbs(
                    "fixed",
                    f"v3 uses Extended Data ids ({', '.join(sorted(k for k in v3_files if k.startswith('ed_fig'))[:4])}…)",
                    "automated",
                )
            if has_ed:
                return attach_thumbs(
                    "improved",
                    "v3 introduces ed_fig_* files but some main-fig collisions remain",
                    "automated",
                )
            return attach_thumbs("unchanged", "No ed_fig_* files in v3", "automated")

        # Canonical case: Fig.1 as fig_1_2
        v3_has_exact = False
        if expected:
            v3_has_exact = any(
                re.match(rf"^{re.escape(expected)}$", p.stem, re.I) for p in v3_expected
            )

        # Collision hits must be true suffixes (fig_1_2), not the canonical file.
        v3_obs_true = [
            p
            for p in v3_obs
            if observed
            and (
                p.stem == observed
                or p.stem.startswith(observed + "_page_")
                or re.match(
                    rf"^{re.escape(expected or '')}_\d+(?:_page_\d+)?$",
                    p.stem,
                    re.I,
                )
            )
        ]
        v2_obs_true = [
            p
            for p in v2_obs
            if observed
            and (
                p.stem == observed
                or p.stem.startswith(observed + "_page_")
                or (
                    expected
                    and re.match(
                        rf"^{re.escape(expected)}_\d+(?:_page_\d+)?$",
                        p.stem,
                        re.I,
                    )
                )
            )
        ]

        if v3_has_exact and not v3_obs_true:
            if not v2_has_exact_expected(v2_files, expected) and (
                bool(v2_obs_true) or not v2_expected
            ):
                return attach_thumbs(
                    "fixed",
                    f"Corrected naming: v2 had {[p.name for p in v2_obs_true] or v2_names}; "
                    f"v3 has {[p.name for p in v3_expected]}",
                    "automated",
                )
            if v2_has_exact_expected(v2_files, expected) and not v2_obs_true:
                return attach_thumbs(
                    "unchanged",
                    "Correct filename already present in v2",
                    "automated",
                )
            if v2_has_exact_expected(v2_files, expected) and v2_obs_true and not v3_obs_true:
                return attach_thumbs(
                    "fixed",
                    f"Removed collision {[p.name for p in v2_obs_true]}; kept {expected}",
                    "automated",
                )
            return attach_thumbs(
                "fixed",
                f"v3 provides canonical {expected}",
                "automated",
            )
        if v3_has_exact and v3_obs_true:
            return attach_thumbs(
                "improved",
                f"Canonical {expected} exists but collision {[p.name for p in v3_obs_true]} remains",
                "automated",
            )
        if observed and v3_obs_true and not v3_has_exact:
            return attach_thumbs(
                "unchanged",
                f"Still only collision id {[p.name for p in v3_obs_true]}",
                "automated",
            )
        # Scheme→Fig mislabel etc.
        if expected and expected.startswith("scheme_"):
            return attach_thumbs(
                "needs_visual",
                "Scheme vs figure labeling requires visual confirmation",
                "visual_review",
            )
        return attach_thumbs(
            "unchanged",
            "Could not confirm corrected numbering from filenames",
            "automated",
        )

    if itype == "duplicate":
        expected = issue.expected_stem
        # Collision suffixes for this figure number OR explicitly observed stems
        num = None
        if expected:
            kn = stem_kind_and_number(expected)
            num = kn[1] if kn else None

        observed_stems = []
        if issue.observed_stem:
            observed_stems = [
                s.strip() for s in issue.observed_stem.split(",") if s.strip()
            ]

        def collisions(files: dict[str, Path]) -> list[str]:
            names: list[str] = []
            if num:
                names.extend(
                    name
                    for name in files
                    if re.match(
                        rf"^(?:ed_|supp_)?fig_{num}_\d+(?:_page_\d+)?$",
                        name,
                        re.I,
                    )
                )
            # Also track wrong sibling ids mentioned in the note (e.g. fig_7_2
            # for a Fig. 6 duplicate).
            for stem in observed_stems:
                if stem == expected:
                    continue
                if stem in files and stem not in names:
                    names.append(stem)
                for key in files:
                    if key.startswith(stem + "_page_") and key not in names:
                        names.append(key)
            return names

        v2_c = collisions(v2_files)
        v3_c = collisions(v3_files)
        v3_base = bool(expected and expected in v3_files)
        metrics["v2_collisions"] = v2_c
        metrics["v3_collisions"] = v3_c
        # Refresh hit lists for thumbnails around expected + collisions
        v2_hits = find_related_files(v2_files, expected) or [
            v2_files[n] for n in v2_c if n in v2_files
        ]
        v3_hits = find_related_files(v3_files, expected) or [
            v3_files[n] for n in v3_c if n in v3_files
        ]
        v2_names = [p.name for p in v2_hits]
        v3_names = [p.name for p in v3_hits]

        if v2_c and not v3_c and (v3_base or expected is None):
            return attach_thumbs(
                "fixed",
                f"Removed duplicate collision(s) {v2_c}"
                + (f"; kept {expected}" if expected else ""),
                "automated",
            )
        if v2_c and not v3_c:
            return attach_thumbs(
                "improved",
                f"Collision files removed ({v2_c}) but base {expected} missing",
                "automated",
            )
        if v2_c and v3_c:
            if len(v3_c) < len(v2_c):
                return attach_thumbs(
                    "improved",
                    f"Fewer collisions: {v2_c} → {v3_c}",
                    "automated",
                )
            return attach_thumbs(
                "unchanged",
                f"Collisions remain: {v3_c}",
                "automated",
            )
        if not v2_c and not v3_c:
            return attach_thumbs(
                "unchanged",
                "No collision filenames found in either version (may be content-level duplicate)",
                "visual_review",
            )
        return attach_thumbs("unchanged", "Duplicate status unclear from filenames", "automated")

    if itype == "missing":
        expected = issue.expected_stem
        v2_has = bool(find_related_files(v2_files, expected, issue.figure_label))
        v3_has = bool(find_related_files(v3_files, expected, issue.figure_label))
        if not v2_has and v3_has:
            return attach_thumbs(
                "fixed",
                f"Recovered {expected or issue.figure_label} in v3",
                "automated",
            )
        if v2_has and v3_has:
            return attach_thumbs(
                "unchanged",
                "File already present in v2 (failure may be crop quality)",
                "visual_review",
            )
        if not v3_has:
            return attach_thumbs(
                "unchanged",
                f"Still missing {expected or issue.figure_label}",
                "automated",
            )
        return attach_thumbs("unchanged", "Missing-figure check inconclusive", "automated")

    if itype in {"partial", "over", "incorrect"}:
        expected = issue.expected_stem
        # Prefer exact stem for incorrect junk ids like fig_34 / fig_1_2.
        if itype == "incorrect" and expected:
            v2_exact = [
                p for p in find_related_files(v2_files, expected) if p.stem == expected
            ]
            v3_exact = [
                p for p in find_related_files(v3_files, expected) if p.stem == expected
            ]
            if v2_exact and not v3_exact:
                v2_hits = v2_exact
                v3_hits = []
                v2_names = [p.name for p in v2_exact]
                v3_names = []
                return attach_thumbs(
                    "fixed",
                    f"Removed incorrect extraction {expected}",
                    "automated",
                )

        v2_p = pick_primary(find_related_files(v2_files, expected, issue.figure_label))
        v3_p = pick_primary(find_related_files(v3_files, expected, issue.figure_label))
        if not v2_p and not v3_p:
            return attach_thumbs(
                "unchanged",
                "No comparable crop found in v2 or v3",
                "automated",
            )
        if not v2_p and v3_p:
            return attach_thumbs(
                "improved",
                "Crop present in v3 but absent/unmatched in v2 — confirm visually",
                "visual_review",
            )
        if v2_p and not v3_p:
            if itype == "incorrect":
                return attach_thumbs(
                    "fixed",
                    "Incorrect crop no longer present in v3",
                    "automated",
                )
            return attach_thumbs(
                "worsened",
                "Crop present in v2 but missing in v3",
                "automated",
            )
        assert v2_p and v3_p
        m2, m3 = image_metrics(v2_p), image_metrics(v3_p)
        metrics["v2"] = m2
        metrics["v3"] = m3
        ratio = m3["area"] / max(m2["area"], 1)
        metrics["area_ratio"] = round(ratio, 3)
        # Aspect change can indicate different crop region
        ar2 = m2["width"] / max(m2["height"], 1)
        ar3 = m3["width"] / max(m3["height"], 1)
        metrics["aspect_v2"] = round(ar2, 3)
        metrics["aspect_v3"] = round(ar3, 3)
        # Never auto-fixed for crop quality.
        if itype == "over":
            # Smaller can be better for over-extraction; larger may be worse
            if ratio < 0.85:
                status = "improved"
                why = f"Crop area decreased ({m2['width']}×{m2['height']} → {m3['width']}×{m3['height']}); confirm less unrelated content"
            elif ratio > 1.15:
                status = "needs_visual"
                why = f"Crop grew ({ratio:.2f}×); for over-extraction this may be worse — visual review required"
            else:
                status = "needs_visual"
                why = "Crop size similar; visual review required for over-extraction"
        else:
            # partial / incorrect: larger often candidate improvement if not absurd
            if m2["height"] < 80 and m3["height"] >= 200:
                status = "improved"
                why = f"Collapsed crop repaired ({m2['width']}×{m2['height']} → {m3['width']}×{m3['height']})"
            elif ratio >= 1.4 and 0.25 <= (ar3 / max(ar2, 1e-6)) <= 4.0:
                status = "improved"
                why = f"Substantially larger crop ({ratio:.2f}× area); confirm it captures more of the intended figure without excess"
            elif ratio <= 0.7:
                status = "needs_visual"
                why = f"Crop shrank ({ratio:.2f}×); may be better or worse depending on content"
            else:
                status = "needs_visual"
                why = "Crop metrics changed only modestly; visual confirmation required"
        return attach_thumbs(status, why, "visual_review")

    if itype == "extended_data_labels":
        v2_ed = [k for k in v2_files if k.startswith("ed_fig")]
        v3_ed = [k for k in v3_files if k.startswith("ed_fig")]
        v2_coll = [k for k in v2_files if re.search(r"fig_\d+_\d+", k)]
        v3_coll = [k for k in v3_files if re.search(r"fig_\d+_\d+", k)]
        metrics["v2_ed"] = v2_ed
        metrics["v3_ed"] = v3_ed
        if v3_ed and not v2_ed:
            return attach_thumbs(
                "improved",
                f"v3 introduces Extended Data ids ({len(v3_ed)} files); confirm completeness visually",
                "visual_review",
            )
        if v3_ed and len(v3_coll) < len(v2_coll):
            return attach_thumbs(
                "improved",
                f"ed_fig_* present and fewer collision names ({len(v2_coll)}→{len(v3_coll)})",
                "visual_review",
            )
        if not v3_ed:
            return attach_thumbs(
                "unchanged",
                "No ed_fig_* filenames in v3",
                "automated",
            )
        return attach_thumbs(
            "needs_visual",
            "Extended Data files present in both or mixed — review labels",
            "visual_review",
        )

    # other
    return attach_thumbs(
        "needs_visual",
        f"Unclassified note: {issue.raw_clause}",
        "visual_review",
    )


def v2_has_exact_expected(files: dict[str, Path], expected: str | None) -> bool:
    if not expected:
        return False
    return any(re.match(rf"^{re.escape(expected)}$", k, re.I) for k in files)


def status_rank(status: str) -> int:
    order = {
        "fixed": 0,
        "improved": 1,
        "needs_visual": 2,
        "unchanged": 3,
        "worsened": 4,
        "not_applicable": 5,
    }
    return order.get(status, 9)


def build_html(
    results: list[IssueResult],
    papers: list[dict[str, Any]],
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    status_colors = {
        "fixed": "#1b7f4a",
        "improved": "#2f6fed",
        "needs_visual": "#b45309",
        "unchanged": "#6b7280",
        "worsened": "#b42318",
        "not_applicable": "#9ca3af",
    }

    type_labels = {
        "wrong_number": "Wrong numbering",
        "duplicate": "Duplicates",
        "missing": "Missing figures",
        "partial": "Partial / under-crop",
        "over": "Over-extraction",
        "incorrect": "Incorrect extraction",
        "extended_data_labels": "Extended Data labels",
        "other": "Other",
    }

    # Highlight gallery: prioritize focus papers, then fixed/improved with images.
    focus_papers = {8, 15, 21, 43, 57, 65, 70}
    gallery: list[IssueResult] = []
    seen: set[tuple[int, str, str]] = set()

    def add_result(result: IssueResult) -> None:
        key = (
            result.issue.paper_number,
            result.issue.issue_type,
            result.issue.figure_label or "",
        )
        if key in seen:
            return
        if not (result.v2_thumb or result.v3_thumb):
            # Still show missing→fixed / naming fixes without both thumbs when useful
            if result.status not in {"fixed", "improved", "worsened"}:
                return
        seen.add(key)
        gallery.append(result)

    for result in results:
        if result.issue.paper_number in focus_papers:
            add_result(result)
    for status in ("fixed", "improved", "worsened", "needs_visual", "unchanged"):
        for result in results:
            if result.status != status:
                continue
            add_result(result)
            if len(gallery) >= 56:
                break
        if len(gallery) >= 56:
            break

    by_type = summary["by_issue_type"]
    type_rows = "".join(
        f"<tr><td>{type_labels.get(k, k)}</td><td>{v['fixed']}</td>"
        f"<td>{v['improved']}</td><td>{v['needs_visual']}</td>"
        f"<td>{v['unchanged']}</td><td>{v['worsened']}</td><td>{v['total']}</td></tr>"
        for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["total"])
    )

    paper_rows = []
    for paper in papers:
        statuses = paper["status_counts"]
        badge = (
            "fixed"
            if statuses.get("fixed", 0) and not statuses.get("worsened", 0)
            else "improved"
            if statuses.get("improved", 0) or statuses.get("fixed", 0)
            else "worsened"
            if statuses.get("worsened", 0)
            else "unchanged"
        )
        paper_rows.append(
            "<tr>"
            f"<td>{paper['number']}</td>"
            f"<td class='doi'>{paper.get('doi') or ''}</td>"
            f"<td class='folder'>{paper['folder'][:48]}{'…' if len(paper['folder'])>48 else ''}</td>"
            f"<td><span class='pill {badge}'>{badge}</span></td>"
            f"<td>{statuses.get('fixed', 0)}</td>"
            f"<td>{statuses.get('improved', 0)}</td>"
            f"<td>{statuses.get('needs_visual', 0)}</td>"
            f"<td>{statuses.get('unchanged', 0)}</td>"
            f"<td>{statuses.get('worsened', 0)}</td>"
            f"<td class='notes'>{html_escape(paper['notes_preview'])}</td>"
            "</tr>"
        )

    cards = []
    for result in gallery:
        issue = result.issue
        v2_img = (
            f"<img src='{result.v2_thumb}' alt='v2'>"
            if result.v2_thumb
            else "<div class='missing'>No v2 crop</div>"
        )
        v3_img = (
            f"<img src='{result.v3_thumb}' alt='v3'>"
            if result.v3_thumb
            else "<div class='missing'>No v3 crop</div>"
        )
        meta = result.metrics
        size_line = ""
        if "v2" in meta or "v3" in meta:
            v2s = meta.get("v2")
            v3s = meta.get("v3")
            v2_txt = f"{v2s['width']}×{v2s['height']}" if v2s else "—"
            v3_txt = f"{v3s['width']}×{v3s['height']}" if v3s else "—"
            ratio_txt = (
                f" (area ×{meta['area_ratio']})" if "area_ratio" in meta else ""
            )
            size_line = (
                f"<div class='sizes'>v2: {v2_txt} → v3: {v3_txt}{ratio_txt}</div>"
            )
        cards.append(
            f"""
            <article class="card status-{result.status}">
              <header>
                <div class="title">Paper #{issue.paper_number} · {html_escape(issue.figure_label or 'paper-level')}</div>
                <span class="pill {result.status}">{result.status.replace('_', ' ')}</span>
              </header>
              <div class="meta">
                <div><strong>DOI:</strong> {html_escape(issue.doi or '—')}</div>
                <div><strong>Type:</strong> {type_labels.get(issue.issue_type, issue.issue_type)}</div>
                <div><strong>Source note:</strong> {html_escape(issue.raw_clause)}</div>
                <div><strong>Check:</strong> {result.confidence}
                  · {html_escape(result.rationale)}</div>
                <div><strong>Files:</strong> v2 [{html_escape(', '.join(result.v2_files) or '—')}]
                  → v3 [{html_escape(', '.join(result.v3_files) or '—')}]</div>
                {size_line}
              </div>
              <div class="pair">
                <figure><figcaption>Before (v2)</figcaption>{v2_img}</figure>
                <figure><figcaption>After (v3)</figcaption>{v3_img}</figure>
              </div>
            </article>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>XRD Figure Extraction: v2 → v3 Comparison</title>
<style>
  :root {{
    --bg: #f6f4ef;
    --ink: #1c1917;
    --muted: #57534e;
    --card: #ffffff;
    --line: #e7e5e4;
    --accent: #0f766e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Source Serif 4", "Iowan Old Style", Palatino, "Book Antiqua", Georgia, serif;
    color: var(--ink);
    background:
      radial-gradient(1200px 500px at 10% -10%, #dbeafe 0%, transparent 55%),
      radial-gradient(900px 400px at 100% 0%, #fde68a55 0%, transparent 50%),
      var(--bg);
    line-height: 1.45;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 64px; }}
  header.hero {{
    padding: 28px 0 18px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 28px;
  }}
  header.hero h1 {{
    font-size: 2rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    margin: 0 0 8px;
  }}
  header.hero p {{ margin: 0; color: var(--muted); max-width: 70ch; }}
  .kicker {{
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    color: var(--accent);
    margin-bottom: 8px;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin: 22px 0 8px;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  }}
  .stat {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 14px 12px;
  }}
  .stat .n {{ font-size: 1.7rem; font-weight: 700; line-height: 1; }}
  .stat .l {{ font-size: 0.78rem; color: var(--muted); margin-top: 6px; }}
  h2 {{
    font-size: 1.35rem;
    margin: 34px 0 12px;
    letter-spacing: -0.01em;
  }}
  .note {{
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    font-size: 0.9rem;
    color: var(--muted);
    margin: 0 0 16px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    font-size: 0.86rem;
  }}
  th, td {{
    padding: 9px 10px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: #fafaf9; font-weight: 600; }}
  tr:last-child td {{ border-bottom: 0; }}
  td.doi, td.folder, td.notes {{ color: var(--muted); font-size: 0.8rem; }}
  .pill {{
    display: inline-block;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.02em;
    color: white;
    text-transform: uppercase;
  }}
  .pill.fixed {{ background: {status_colors['fixed']}; }}
  .pill.improved {{ background: {status_colors['improved']}; }}
  .pill.needs_visual {{ background: {status_colors['needs_visual']}; }}
  .pill.unchanged {{ background: {status_colors['unchanged']}; }}
  .pill.worsened {{ background: {status_colors['worsened']}; }}
  .gallery {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px;
    border-left: 5px solid var(--line);
  }}
  .card.status-fixed {{ border-left-color: {status_colors['fixed']}; }}
  .card.status-improved {{ border-left-color: {status_colors['improved']}; }}
  .card.status-needs_visual {{ border-left-color: {status_colors['needs_visual']}; }}
  .card.status-unchanged {{ border-left-color: {status_colors['unchanged']}; }}
  .card.status-worsened {{ border-left-color: {status_colors['worsened']}; }}
  .card header {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: baseline;
    margin-bottom: 8px;
  }}
  .card .title {{ font-weight: 650; }}
  .card .meta {{
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    font-size: 0.82rem;
    color: var(--muted);
    display: grid;
    gap: 3px;
    margin-bottom: 10px;
  }}
  .pair {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}
  figure {{ margin: 0; }}
  figcaption {{
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  .pair img {{
    width: 100%;
    height: auto;
    max-height: 360px;
    object-fit: contain;
    background: #f5f5f4;
    border: 1px solid var(--line);
    border-radius: 8px;
  }}
  .missing {{
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 160px;
    background: #f5f5f4;
    border: 1px dashed #d6d3d1;
    border-radius: 8px;
    color: var(--muted);
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    font-size: 0.85rem;
  }}
  .sizes {{ margin-top: 2px; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 8px;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    margin: 10px 0 18px;
  }}
  footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    font-size: 0.82rem;
  }}
  @media (max-width: 900px) {{
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
    .pair {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <div class="kicker">XRD figure extraction evaluation</div>
      <h1>Before / after comparison (hoje_v2 → hoje_v3)</h1>
      <p>
        Issues are taken from <code>XRD fig extract.xlsx</code>. Automated checks
        cover numbering, duplicates, and presence. Crop-quality claims are marked
        for visual review and are never counted as fully fixed from size alone.
      </p>
      <div class="stats">
        <div class="stat"><div class="n">{summary['papers_with_issues']}</div><div class="l">Papers with documented errors</div></div>
        <div class="stat"><div class="n" style="color:{status_colors['fixed']}">{summary['status_counts'].get('fixed',0)}</div><div class="l">Issues fixed</div></div>
        <div class="stat"><div class="n" style="color:{status_colors['improved']}">{summary['status_counts'].get('improved',0)}</div><div class="l">Partially improved</div></div>
        <div class="stat"><div class="n" style="color:{status_colors['unchanged']}">{summary['status_counts'].get('unchanged',0)}</div><div class="l">Unresolved</div></div>
        <div class="stat"><div class="n" style="color:{status_colors['worsened']}">{summary['status_counts'].get('worsened',0)}</div><div class="l">Worsened</div></div>
      </div>
    </header>

    <h2>A. Summary by error type</h2>
    <p class="note">
      Total documented issues evaluated: <strong>{summary['issues_total']}</strong>.
      Visual-review items: <strong>{summary['status_counts'].get('needs_visual',0)}</strong>
      (crop quality / ambiguous labels — not auto-scored as fixed).
    </p>
    <table>
      <thead>
        <tr>
          <th>Error type</th><th>Fixed</th><th>Improved</th>
          <th>Needs visual</th><th>Unresolved</th><th>Worsened</th><th>Total</th>
        </tr>
      </thead>
      <tbody>{type_rows}</tbody>
    </table>

    <h2>B. Side-by-side figure comparisons</h2>
    <div class="legend">
      <span class="pill fixed">fixed</span>
      <span class="pill improved">improved</span>
      <span class="pill needs_visual">needs visual</span>
      <span class="pill unchanged">unchanged</span>
      <span class="pill worsened">worsened</span>
    </div>
    <p class="note">
      Showing {len(gallery)} informative examples (prioritizing verified fixes, then
      improvements, then remaining failures). Images are embedded thumbnails.
    </p>
    <div class="gallery">
      {''.join(cards)}
    </div>

    <h2>C. Per-paper results</h2>
    <p class="note">One row per paper that has spreadsheet-documented extraction errors (excluding pure “No XRD figure”).</p>
    <table>
      <thead>
        <tr>
          <th>#</th><th>DOI</th><th>Folder</th><th>Overall</th>
          <th>Fixed</th><th>Improved</th><th>Visual</th><th>Unresolved</th><th>Worse</th><th>Notes (preview)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(paper_rows)}
      </tbody>
    </table>

    <footer>
      Generated by <code>scripts/compare_figure_extraction.py</code>.
      Before: <code>{summary['before']}</code> · After: <code>{summary['after']}</code>.
      Automated naming/presence checks are definitive for numbering and duplicates;
      crop-quality judgments require looking at the side-by-side images above.
    </footer>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    before = args.before
    after = args.after
    if not before.is_dir():
        raise SystemExit(f"Before directory not found: {before}")
    if not after.is_dir():
        raise SystemExit(
            f"After directory not found: {after}\n"
            "Generate hoje_v3 first, then re-run this script."
        )
    if not args.xlsx.is_file():
        raise SystemExit(f"Spreadsheet not found: {args.xlsx}")

    before_dirs = list_paper_dirs(before)
    after_dirs = list_paper_dirs(after)
    if before_dirs != after_dirs:
        # Still proceed by name; warn in summary.
        missing = sorted(set(before_dirs) - set(after_dirs))
        extra = sorted(set(after_dirs) - set(before_dirs))
        print(f"Warning: folder set differs (missing in after: {len(missing)}, extra: {len(extra)})")

    df, issues = parse_spreadsheet(args.xlsx, before_dirs)
    print(f"Parsed {len(issues)} documented issues across spreadsheet notes")

    results: list[IssueResult] = []
    for issue in issues:
        folder = issue.folder
        v2_dir = before / folder
        v3_dir = after / folder
        if not v3_dir.is_dir():
            # try match by identical name only
            result = IssueResult(
                issue=issue,
                status="unchanged",
                confidence="automated",
                rationale=f"Folder missing in after: {folder}",
            )
            results.append(result)
            continue
        v2_files = collect_figure_pngs(v2_dir) if v2_dir.is_dir() else {}
        v3_files = collect_figure_pngs(v3_dir)
        results.append(evaluate_issue(issue, v2_files, v3_files))

    # Paper rollup
    by_paper: dict[int, list[IssueResult]] = defaultdict(list)
    for result in results:
        by_paper[result.issue.paper_number].append(result)

    papers = []
    for number, group in sorted(by_paper.items()):
        issue0 = group[0].issue
        counts = Counter(r.status for r in group)
        notes = []
        row = df.loc[df["Number"] == number].iloc[0]
        for col in ("Notes XRD figure", "Notes nonXRD figure"):
            val = row.get(col)
            if pd.notna(val) and str(val).strip() and not SKIP_NOTE.match(str(val)):
                notes.append(str(val).strip())
        papers.append(
            {
                "number": number,
                "doi": issue0.doi,
                "folder": issue0.folder,
                "status_counts": dict(counts),
                "notes_preview": " | ".join(notes)[:220],
                "issue_count": len(group),
            }
        )

    status_counts = Counter(r.status for r in results)
    by_issue_type: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_issue_type.setdefault(
            result.issue.issue_type,
            {"fixed": 0, "improved": 0, "needs_visual": 0, "unchanged": 0, "worsened": 0, "total": 0},
        )
        bucket["total"] += 1
        if result.status in bucket:
            bucket[result.status] += 1

    summary = {
        "before": str(before),
        "after": str(after),
        "xlsx": str(args.xlsx),
        "papers_total_in_sheet": int(len(df)),
        "papers_with_issues": len(papers),
        "issues_total": len(results),
        "status_counts": dict(status_counts),
        "by_issue_type": by_issue_type,
        "automated_fixed_or_improved": status_counts.get("fixed", 0)
        + status_counts.get("improved", 0),
        "notes": [
            "Crop-quality statuses use needs_visual/improved heuristics; fixed is reserved for naming/presence.",
            "Paper index follows `ls` order in the before directory (matches spreadsheet Number).",
        ],
    }

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "summary": summary,
        "papers": papers,
        "issues": [
            {
                **asdict(r.issue),
                "status": r.status,
                "confidence": r.confidence,
                "rationale": r.rationale,
                "v2_files": r.v2_files,
                "v3_files": r.v3_files,
                "metrics": r.metrics,
                # omit huge thumbs from JSON
            }
            for r in results
        ],
    }
    (out / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # CSV
    with (out / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_number",
                "doi",
                "folder",
                "source",
                "issue_type",
                "figure_label",
                "expected_stem",
                "observed_stem",
                "raw_clause",
                "status",
                "confidence",
                "rationale",
                "v2_files",
                "v3_files",
                "v2_size",
                "v3_size",
                "area_ratio",
            ],
        )
        writer.writeheader()
        for r in results:
            v2m = r.metrics.get("v2") or {}
            v3m = r.metrics.get("v3") or {}
            writer.writerow(
                {
                    "paper_number": r.issue.paper_number,
                    "doi": r.issue.doi or "",
                    "folder": r.issue.folder,
                    "source": r.issue.source,
                    "issue_type": r.issue.issue_type,
                    "figure_label": r.issue.figure_label or "",
                    "expected_stem": r.issue.expected_stem or "",
                    "observed_stem": r.issue.observed_stem or "",
                    "raw_clause": r.issue.raw_clause,
                    "status": r.status,
                    "confidence": r.confidence,
                    "rationale": r.rationale,
                    "v2_files": "|".join(r.v2_files),
                    "v3_files": "|".join(r.v3_files),
                    "v2_size": f"{v2m.get('width','')}x{v2m.get('height','')}" if v2m else "",
                    "v3_size": f"{v3m.get('width','')}x{v3m.get('height','')}" if v3m else "",
                    "area_ratio": r.metrics.get("area_ratio", ""),
                }
            )

    build_html(results, papers, summary, out / "comparison_report.html")

    print(f"Wrote {out / 'comparison_report.html'}")
    print(f"Wrote {out / 'comparison_summary.csv'}")
    print(f"Wrote {out / 'comparison_summary.json'}")
    print("Status counts:", dict(status_counts))


if __name__ == "__main__":
    main()
