from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pymupdf
import requests
from lxml import etree

# Allow imports from this directory (parse_figures) and repo root (digitization).
_LEGACY_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LEGACY_DIR.parent
for _path in (_LEGACY_DIR, _REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from parse_figures import (
    _infer_crop_rect_for_caption_box,
    crop_figure_from_grobid_coords,
    extract_document_figures,
    find_pdf_figure_caption_anchors,
    format_grobid_coords,
    select_figure_crop_coords,
)


LOGGER = logging.getLogger(__name__)

TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"

NS = {
    "tei": TEI_NAMESPACE,
}

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

CONTEXT_WINDOW = 4
CONTEXT_SECTION_XRD_THRESHOLD = 1.0
CONTEXT_WINDOW_XRD_THRESHOLD = 1.5

GROBID_FIGURE_ID_PATTERN = re.compile(r"^fig_(\d+)$", re.IGNORECASE)

XRD_KEYWORDS = {
    "xrd",
    "x-ray diffraction",
    "x ray diffraction",
    "powder diffraction",
    "diffraction",
    "diffraction pattern",
    "diffraction patterns",
    "diffraction plane",
    "diffraction planes",
    "diffraction peak",
    "diffraction peaks",
    "diffractogram",
    "diffractograms",
    "2θ",
    "2 theta",
    "two theta",
    "rietveld",
    "rietveld refinement",
    "bragg",
    "bragg peak",
    "xrd pattern",
    "pxrd",
    "synchrotron diffraction",
}

RESULT_SECTION_KEYWORDS = {
    "result",
    "results",
    "discussion",
    "results and discussion",
    "characterization",
    "structural characterization",
    "crystal structure",
    "phase analysis",
    "x-ray diffraction",
    "xrd analysis",
    "structural analysis",
    "phase identification",
    "rietveld refinement",
    "experimental results",
}

METHOD_SECTION_KEYWORDS = {
    "experimental",
    "materials and methods",
    "methods",
    "characterization",
    "xrd measurement",
    "x-ray diffraction measurement",
    "instrumentation",
}

XRD_ANALYSIS_TERMS = {
    "peak",
    "peaks",
    "reflection",
    "reflections",
    "plane",
    "planes",
    "phase",
    "phases",
    "crystalline",
    "crystallinity",
    "amorphous",
    "lattice",
    "cell parameter",
    "lattice parameter",
    "space group",
    "indexed",
    "indexing",
    "shift",
    "broadening",
    "intensity",
    "crystallite size",
    "scherrer",
    "rietveld",
    "bragg",
    "impurity",
    "secondary phase",
    "preferred orientation",
    "full width at half maximum",
    "fwhm",
}

FIGURE_REFERENCE_PATTERN = re.compile(
    r"""
    \b
    (?:
        extended\s+data\s+fig(?:ure)?s?\.?
        |
        supplementary\s+(?:information\s+)?fig(?:ure)?s?\.?
        |
        supporting\s+(?:information\s+)?fig(?:ure)?s?\.?
        |
        fig(?:ure)?s?\.?
    )
    \s*
    (?:[\(\[\s|])*
    (?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Caption lead-in: figure identity at (or near) the start of a caption.
CAPTION_FIGURE_IDENTITY_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[\|\-–—.:]|\b\d{1,3}\b|[A-Za-z]\b){0,6}\s*
    (?P<kind>
        extended\s+data\s+fig(?:ure)?s?\.?
        | data\s+fig(?:ure)?s?\.?
        | supplementary\s+(?:information\s+)?fig(?:ure)?s?\.?
        | supporting\s+(?:information\s+)?fig(?:ure)?s?\.?
        | fig(?:ure)?s?\.?
    )
    \s*
    (?:[\(\[|\s])*
    (?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

EXTENDED_DATA_FIGURE_PATTERN = re.compile(
    r"\bextended\s+data\s+fig(?:ure)?s?\.?\s*(?:[\(\[|\s])*(?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)",
    flags=re.IGNORECASE,
)
SUPPLEMENTARY_FIGURE_PATTERN = re.compile(
    r"\b(?:supplementary|supporting)\s+(?:information\s+)?fig(?:ure)?s?\.?\s*(?:[\(\[|\s])*(?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)",
    flags=re.IGNORECASE,
)

FIGURE_KIND_MAIN = "fig"
FIGURE_KIND_EXTENDED = "ed_fig"
FIGURE_KIND_SUPPLEMENTARY = "supp_fig"
MAX_PLAUSIBLE_FIGURE_NUMBER = 99

ABBREVIATION_DOT_PATTERN = re.compile(
    r"\b(?:Fig(?:ure)?|vs|etc)\.",
    flags=re.IGNORECASE,
)

SENTENCE_ABBREVIATION_PLACEHOLDER = "\uE000"


def split_into_sentences(text: str) -> list[str]:
    protected_text = ABBREVIATION_DOT_PATTERN.sub(
        lambda match: match.group(0).replace(".", SENTENCE_ABBREVIATION_PLACEHOLDER),
        normalize_whitespace(text),
    )
    sentences = re.split(r"(?<=[.!?])\s+", protected_text)

    return [
        sentence.replace(SENTENCE_ABBREVIATION_PLACEHOLDER, ".")
        for sentence in sentences
        if sentence.strip()
    ]

CLAUSE_SPLIT_PATTERN = re.compile(
    r";\s*|(?:\.\s+)?On the other hand,\s*",
    flags=re.IGNORECASE,
)

DOI_PATTERN = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    flags=re.IGNORECASE,
)

THETA_PATTERN = re.compile(
    r"""
    (?:
        2\s*theta
        |
        2\s*θ
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

PEAK_POSITION_PATTERN = re.compile(
    r"""
    (?P<value>\d{1,3}(?:\.\d+)?)
    \s*
    (?:
        °
        |
        degrees?
        |
        deg
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

MILLER_INDEX_PATTERN = re.compile(
    r"""
    \(
    \s*
    [\-\u2212]?\d+
    \s*
    [,\s]*
    [\-\u2212]?\d+
    \s*
    [,\s]*
    [\-\u2212]?\d+
    \s*
    \)
    """,
    flags=re.VERBOSE,
)


@dataclass
class Author:
    given_name: str | None = None
    middle_name: str | None = None
    surname: str | None = None
    email: str | None = None
    affiliation: str | None = None
    orcid: str | None = None


@dataclass
class Paragraph:
    paragraph_id: str
    text: str
    section_path: list[str]
    section_type: str
    figure_references: list[str] = field(default_factory=list)
    xrd_score: float = 0.0
    peak_positions: list[float] = field(default_factory=list)
    miller_indices: list[str] = field(default_factory=list)


@dataclass
class Section:
    section_id: str
    title: str
    path: list[str]
    section_type: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


@dataclass
class Figure:
    figure_id: str | None
    label: str | None
    normalized_label: str | None
    caption: str
    graphic_targets: list[str]
    coords: str | None
    graphic_coords: str | None
    figure_type: str | None
    xrd_score: float
    is_likely_xrd: bool
    is_caption_xrd: bool
    image_paths: list[str] = field(default_factory=list)
    referenced_by_paragraph_ids: list[str] = field(default_factory=list)
    context_paragraph_ids: list[str] = field(default_factory=list)


@dataclass
class Table:
    table_id: str | None
    label: str | None
    caption: str
    text: str


@dataclass
class DocumentMetadata:
    title: str | None
    doi: str | None
    abstract: str | None
    journal: str | None
    publication_date: str | None
    authors: list[Author]
    keywords: list[str]


@dataclass
class ParsedDocument:
    source_pdf: str
    source_sha256: str
    metadata: DocumentMetadata
    sections: list[Section]
    paragraphs: list[Paragraph]
    figures: list[Figure]
    tables: list[Table]
    references: list[dict[str, Any]]
    counts: dict[str, int]


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def element_text(element: etree._Element | None) -> str:
    if element is None:
        return ""

    return normalize_whitespace(" ".join(element.itertext()))


def xpath_first(
    root: etree._Element,
    expression: str,
) -> etree._Element | str | None:
    matches = root.xpath(expression, namespaces=NS)

    if not matches:
        return None

    return matches[0]


def xpath_text(
    root: etree._Element,
    expression: str,
) -> str | None:
    value = xpath_first(root, expression)

    if value is None:
        return None

    if isinstance(value, etree._Element):
        text = element_text(value)
    else:
        text = normalize_whitespace(str(value))

    return text or None


def xpath_all_text(
    root: etree._Element,
    expression: str,
) -> list[str]:
    values = root.xpath(expression, namespaces=NS)
    results: list[str] = []

    for value in values:
        if isinstance(value, etree._Element):
            text = element_text(value)
        else:
            text = normalize_whitespace(str(value))

        if text:
            results.append(text)

    return results


def normalize_for_matching(text: str) -> str:
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9θ\s\-]", " ", text)
    return normalize_whitespace(text)


def contains_phrase(text: str, phrases: Iterable[str]) -> bool:
    normalized = normalize_for_matching(text)

    return any(
        normalize_for_matching(phrase) in normalized
        for phrase in phrases
    )


def classify_section(title: str) -> str:
    normalized = normalize_for_matching(title)

    if contains_phrase(normalized, RESULT_SECTION_KEYWORDS):
        return "results"

    if contains_phrase(normalized, METHOD_SECTION_KEYWORDS):
        return "methods"

    if "introduction" in normalized or "background" in normalized:
        return "introduction"

    if "conclusion" in normalized or "summary" in normalized:
        return "conclusion"

    if "abstract" in normalized:
        return "abstract"

    if "reference" in normalized or "bibliography" in normalized:
        return "references"

    return "other"


@dataclass(frozen=True)
class FigureIdentity:
    """Logical figure identity: kind + numeric/panel label."""

    kind: str
    number: str

    @property
    def figure_id(self) -> str:
        return f"{self.kind}_{self.number}"


def _plausible_figure_number(number: str | None) -> str | None:
    if not number:
        return None
    cleaned = normalize_whitespace(number).lower().strip(" .")
    match = re.fullmatch(r"([a-z]?)(\d{1,3})([a-z]?)", cleaned)
    if not match:
        return None
    prefix, digits, suffix = match.groups()
    value = int(digits)
    if value < 1 or value > MAX_PLAUSIBLE_FIGURE_NUMBER:
        return None
    # Reject concatenated OCR junk like "1345789" (already blocked by {1,3})
    # and odd mid-caption artifacts like "53" when clearly two digits glued
    # without a panel letter — still allow 10–99 for Extended Data.
    return f"{prefix}{digits}{suffix}"


def _kind_from_match_text(kind_text: str) -> str:
    lowered = kind_text.lower()
    if "extended" in lowered or lowered.startswith("data fig"):
        return FIGURE_KIND_EXTENDED
    if "supplementary" in lowered or "supporting" in lowered:
        return FIGURE_KIND_SUPPLEMENTARY
    return FIGURE_KIND_MAIN


def normalize_figure_label(label: str | None) -> str | None:
    if not label:
        return None

    normalized = normalize_whitespace(label).lower()
    # Prefer structured prefixes before stripping words.
    identity = parse_figure_identity(normalized)
    if identity is not None:
        return identity.number

    normalized = normalized.replace("figure", "fig")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")

    match = re.search(r"(\d{1,3})", normalized)
    if not match:
        return normalized or None

    # Reject concatenated digit soups (e.g. label "1345789").
    if re.fullmatch(r"\d{4,}", normalized):
        return None

    return _plausible_figure_number(match.group(1))


def parse_figure_identity(text: str | None) -> FigureIdentity | None:
    """
    Parse the primary figure identity from a caption or label string.

    Prefers Extended Data / Supplementary identities when a caption is an
    Extended-Data bundle (common in Nature TEI merges). Only trusts
    caption-leading figure mentions — buried "see Fig. 1" prose is ignored.
    """
    if not text or not str(text).strip():
        return None

    caption = normalize_whitespace(str(text))
    # Only the caption lead-in may decide Extended/Supplementary kind.
    # Mid-caption citations like "see Extended Data Fig. 2" inside a main
    # figure caption must not reclassify the figure.
    lead = caption[:140]
    ed_hits = list(EXTENDED_DATA_FIGURE_PATTERN.finditer(lead))
    # Truncated Nature captions sometimes drop the "Extended " prefix on the
    # first entry: "Data Fig. 1 |Extended Data Fig. 7 | ...".
    data_fig_hits = list(
        re.finditer(
            r"(?<![A-Za-z])data\s+fig(?:ure)?s?\.?\s*(?:[\(\[|\s])*(?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)",
            lead,
            flags=re.IGNORECASE,
        )
    )
    # Avoid matching the "Data Fig" inside "Extended Data Fig" twice.
    data_fig_hits = [
        hit
        for hit in data_fig_hits
        if not re.search(
            r"extended\s+$",
            lead[: hit.start()],
            flags=re.IGNORECASE,
        )
    ]
    supp_hits = list(SUPPLEMENTARY_FIGURE_PATTERN.finditer(lead))

    ed_like = sorted(data_fig_hits + ed_hits, key=lambda match: match.start())

    # Nature merged Extended Data heads: "Fig. 1 |Extended Data Fig. 3 |..."
    # or "Data Fig. 1 |Extended Data Fig. 7 |...".
    pipe_ed_bundle = bool(
        re.match(
            r"^\s*(?:fig(?:ure)?s?\.?\s*[A-Za-z]?\d+[A-Za-z]?\s*\|)?\s*"
            r"(?:extended\s+)?data\s+fig",
            lead,
            flags=re.IGNORECASE,
        )
        or (
            len(ed_like) >= 2
            and ed_like[0].start() <= 40
        )
    )
    if pipe_ed_bundle and ed_like:
        number = _plausible_figure_number(ed_like[0].group("number"))
        if number:
            return FigureIdentity(FIGURE_KIND_EXTENDED, number)

    if len(supp_hits) >= 2 and supp_hits[0].start() <= 40:
        number = _plausible_figure_number(supp_hits[0].group("number"))
        if number:
            return FigureIdentity(FIGURE_KIND_SUPPLEMENTARY, number)

    match = CAPTION_FIGURE_IDENTITY_PATTERN.match(caption)
    if not match:
        # Allow truncated "Data Fig. N" from "Extended Data Fig. N".
        soft = re.match(
            r"^\s*data\s+fig(?:ure)?s?\.?\s*(?:[\(\[|\s])*(?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)",
            caption,
            flags=re.IGNORECASE,
        )
        if soft:
            number = _plausible_figure_number(soft.group("number"))
            if number:
                return FigureIdentity(FIGURE_KIND_EXTENDED, number)
        # OCR junk before the real lead-in, e.g. "5 HalfFig. 3 | Fig. 3 | ...".
        head = caption[:80]
        rescue = re.search(
            r"(?:extended\s+data\s+)?fig(?:ure)?s?\.?\s*(?:[\(\[|\s])*(?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)",
            head,
            flags=re.IGNORECASE,
        )
        if rescue and rescue.start() <= 24:
            number = _plausible_figure_number(rescue.group("number"))
            if number:
                kind = (
                    FIGURE_KIND_EXTENDED
                    if re.search(r"extended\s+data", head[: rescue.start() + 20], flags=re.IGNORECASE)
                    else FIGURE_KIND_MAIN
                )
                return FigureIdentity(kind, number)
        return None

    number = _plausible_figure_number(match.group("number"))
    if not number:
        return None

    kind = _kind_from_match_text(match.group("kind"))
    return FigureIdentity(kind, number)


def figure_label_from_caption(caption: str) -> str | None:
    identity = parse_figure_identity(caption)
    return identity.number if identity else None


def figure_id_from_caption(caption: str) -> str | None:
    identity = parse_figure_identity(caption)
    return identity.figure_id if identity else None


# Pipe-separated / merged caption heads: "Fig. 1 | Fig. 2 | Extended Data Fig. 3 |"
_MERGED_CAPTION_IDENTITY_PATTERN = re.compile(
    r"""
    (?:^|\|)
    \s*
    (?P<kind>
        extended\s+data\s+fig(?:ure)?s?\.?
        | data\s+fig(?:ure)?s?\.?
        | supplementary\s+(?:information\s+)?fig(?:ure)?s?\.?
        | supporting\s+(?:information\s+)?fig(?:ure)?s?\.?
        | fig(?:ure)?s?\.?
    )
    \s*
    (?:[\(\[|\s])*
    (?P<number>[A-Za-z]?\d{1,3}[A-Za-z]?)
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def parse_all_figure_identities(text: str | None) -> list[FigureIdentity]:
    """
    Extract every figure identity from a (possibly merged) caption head.

    Nature TEI often concatenates ``Fig. 1 | Fig. 2 | Extended Data Fig. 3 |``.
    Returns identities in order of appearance, de-duplicated by figure_id.
    """
    if not text or not str(text).strip():
        return []

    caption = normalize_whitespace(str(text))
    # Only trust the caption head — body prose after ~220 chars is noisy.
    head = caption[:220]
    identities: list[FigureIdentity] = []
    seen: set[str] = set()

    for match in _MERGED_CAPTION_IDENTITY_PATTERN.finditer(head):
        number = _plausible_figure_number(match.group("number"))
        if not number:
            continue
        kind = _kind_from_match_text(match.group("kind"))
        # Truncated "Data Fig" without Extended → treat as Extended Data.
        if kind == FIGURE_KIND_MAIN and re.match(
            r"^\s*data\s+fig",
            match.group("kind"),
            flags=re.IGNORECASE,
        ):
            kind = FIGURE_KIND_EXTENDED
        identity = FigureIdentity(kind, number)
        if identity.figure_id in seen:
            continue
        seen.add(identity.figure_id)
        identities.append(identity)

    if not identities:
        primary = parse_figure_identity(caption)
        if primary is not None:
            identities.append(primary)

    return identities


def expand_merged_caption_figures(figures: list[Figure]) -> list[str]:
    """
    P2: register stub figures for sibling identities in merged TEI captions.

    When GROBID merges ``Fig. 1 | Fig. 2 |``, only the primary identity is kept
    on the original row; siblings are appended as caption-only stubs so P1
    orphan recovery can crop them from the PDF. Does not invent gaps — only
    identities explicitly present in the caption text.
    """
    notes: list[str] = []
    if not figures:
        return notes

    existing_ids = {
        figure.figure_id for figure in figures if figure.figure_id
    }
    additions: list[Figure] = []

    for figure in figures:
        identities = parse_all_figure_identities(figure.caption)
        if len(identities) < 2:
            continue

        # Keep the original row as-is (already refined). Emit stubs only for
        # sibling identities that have no figure row yet.
        for identity in identities:
            if identity.figure_id in existing_ids:
                continue
            if any(item.figure_id == identity.figure_id for item in additions):
                continue

            kind_name = {
                FIGURE_KIND_EXTENDED: "Extended Data Fig.",
                FIGURE_KIND_SUPPLEMENTARY: "Supplementary Fig.",
            }.get(identity.kind, "Fig.")
            stub = Figure(
                figure_id=identity.figure_id,
                label=f"{kind_name} {identity.number}",
                normalized_label=identity.number,
                caption=figure.caption,
                graphic_targets=[],
                coords=None,
                graphic_coords=None,
                figure_type=figure.figure_type,
                xrd_score=figure.xrd_score,
                is_likely_xrd=figure.is_likely_xrd,
                is_caption_xrd=figure.is_caption_xrd,
                image_paths=[],
                referenced_by_paragraph_ids=list(
                    figure.referenced_by_paragraph_ids
                ),
                context_paragraph_ids=list(figure.context_paragraph_ids),
            )
            additions.append(stub)
            existing_ids.add(identity.figure_id)
            notes.append(
                f"split merged caption → stub {identity.figure_id}"
            )

    figures.extend(additions)
    return notes


def _figure_has_successful_crop(figure: Figure) -> bool:
    return bool(figure.image_paths)


def recover_missing_figures_from_pdf(
    pdf_path: str | Path,
    figures: list[Figure],
    output_dir: str | Path,
    *,
    dpi: int = 300,
    xrd_only: bool = False,
) -> list[str]:
    """
    P1: recover figures missing from GROBID TEI by scanning PDF caption leads.

    Only runs for figure identities that lack a successful crop. Uses PDF
    caption positions (not the evaluation spreadsheet) and the same crop
    inference as caption-only TEI figures. Mutates ``figures`` in place.
    """
    notes: list[str] = []
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_ids = {
        figure.figure_id
        for figure in figures
        if figure.figure_id and _figure_has_successful_crop(figure)
    }

    # Also treat collision suffixes as covering the base id when cropped.
    covered_bases = set(successful_ids)
    for figure_id in list(successful_ids):
        match = re.match(
            r"^((?:ed_|supp_)?fig_\d+[a-z]?)_\d+$",
            figure_id,
            flags=re.IGNORECASE,
        )
        if match:
            covered_bases.add(match.group(1).lower())

    try:
        anchors = find_pdf_figure_caption_anchors(pdf_path)
    except (OSError, ValueError) as exc:
        notes.append(f"PDF caption scan failed: {exc}")
        return notes

    by_id = {
        figure.figure_id: figure
        for figure in figures
        if figure.figure_id
    }

    # Retry TEI caption-only figures that still lack crops (Mode B) using
    # current inference — covers cases where P0 should have worked but
    # extract_document_figures skipped for other reasons.
    for figure in figures:
        if not figure.figure_id or _figure_has_successful_crop(figure):
            continue
        if figure.figure_id in covered_bases:
            continue
        if not figure.coords or not str(figure.coords).strip():
            continue
        # Do not revive demoted body-text phantoms (hash ids / non-captions).
        caption = figure.caption or ""
        if not re.match(
            r"^\s*(?:extended\s+data\s+|supplementary\s+|supporting\s+|data\s+)?"
            r"fig(?:ure)?s?\.?\b",
            caption,
            flags=re.IGNORECASE,
        ):
            continue
        if xrd_only and not figure.is_caption_xrd:
            continue
        crop_coords = select_figure_crop_coords(
            figure.coords,
            figure.graphic_coords,
            pdf_path=pdf_path,
        )
        if not crop_coords:
            continue
        try:
            image_paths = crop_figure_from_grobid_coords(
                pdf_path=pdf_path,
                coords_string=crop_coords,
                output_dir=output_dir,
                figure_id=figure.figure_id,
                dpi=dpi,
                caption_coords=figure.coords,
            )
        except (OSError, ValueError) as exc:
            notes.append(
                f"retry {figure.figure_id}: crop failed ({exc})"
            )
            continue
        if not image_paths:
            continue
        figure.graphic_coords = figure.graphic_coords or crop_coords
        figure.image_paths = [str(path) for path in image_paths]
        covered_bases.add(figure.figure_id)
        notes.append(f"recovered caption-only {figure.figure_id}")

    # Known missing rows (TEI caption-only / P2 stubs) are always eligible.
    known_missing = {
        figure.figure_id
        for figure in figures
        if figure.figure_id and not _figure_has_successful_crop(figure)
    }

    document = pymupdf.open(pdf_path)
    try:
        for anchor in anchors:
            figure_id = anchor.figure_id
            if figure_id in covered_bases:
                continue
            existing = by_id.get(figure_id)
            if existing is not None and _figure_has_successful_crop(existing):
                continue

            # Pure orphans (no TEI/stub row): only recover strong caption leads
            # so we do not invent figures from body callouts.
            is_pure_orphan = existing is None and figure_id not in known_missing
            if is_pure_orphan:
                strong = bool(
                    re.search(r"[\|:\-–—]", anchor.text[:40])
                    or len(anchor.text) >= 36
                    or re.fullmatch(
                        r"fig(?:ure)?s?\.?\s*\d{1,2}\.?",
                        anchor.text.strip(),
                        flags=re.IGNORECASE,
                    )
                )
                if not strong:
                    continue
                # Supplementary orphans are noisy in body text; require a stub.
                if anchor.kind == FIGURE_KIND_SUPPLEMENTARY:
                    continue

            if xrd_only and existing is not None and not existing.is_caption_xrd:
                continue

            if not 1 <= anchor.page <= document.page_count:
                continue
            page = document.load_page(anchor.page - 1)
            crop_rect = _infer_crop_rect_for_caption_box(
                page,
                anchor.x0,
                anchor.y0,
                anchor.x1,
                anchor.y1,
            )
            if crop_rect is None:
                notes.append(
                    f"orphan {figure_id}: caption found, crop inference failed"
                )
                continue

            page_area = max(page.rect.width * page.rect.height, 1.0)
            crop_area = crop_rect.width * crop_rect.height
            # Reject near-full-page orphan crops (usually grabbed the wrong band).
            if is_pure_orphan and crop_area > 0.82 * page_area:
                notes.append(
                    f"orphan {figure_id}: rejected oversized crop"
                )
                continue

            crop_coords = format_grobid_coords(
                [
                    (
                        anchor.page,
                        crop_rect.x0,
                        crop_rect.y0,
                        crop_rect.width,
                        crop_rect.height,
                    )
                ]
            )

            try:
                image_paths = crop_figure_from_grobid_coords(
                    pdf_path=pdf_path,
                    coords_string=crop_coords,
                    output_dir=output_dir,
                    figure_id=figure_id,
                    dpi=dpi,
                    caption_coords=anchor.coords,
                )
            except (OSError, ValueError) as exc:
                notes.append(f"orphan {figure_id}: crop failed ({exc})")
                continue

            if not image_paths:
                continue

            kind_name = {
                FIGURE_KIND_EXTENDED: "Extended Data Fig.",
                FIGURE_KIND_SUPPLEMENTARY: "Supplementary Fig.",
            }.get(anchor.kind, "Fig.")

            if existing is None:
                existing = Figure(
                    figure_id=figure_id,
                    label=f"{kind_name} {anchor.number}",
                    normalized_label=anchor.number,
                    caption=anchor.text,
                    graphic_targets=[],
                    coords=anchor.coords,
                    graphic_coords=crop_coords,
                    figure_type=None,
                    xrd_score=0.0,
                    is_likely_xrd=False,
                    is_caption_xrd=False,
                    image_paths=[],
                )
                figures.append(existing)
                by_id[figure_id] = existing
                notes.append(f"recovered orphan {figure_id} from PDF caption")
            else:
                existing.coords = existing.coords or anchor.coords
                existing.graphic_coords = crop_coords
                if not existing.caption or len(existing.caption) < 8:
                    existing.caption = anchor.text
                notes.append(f"recovered missing crop for {figure_id}")

            existing.image_paths = [str(path) for path in image_paths]
            covered_bases.add(figure_id)
    finally:
        document.close()

    return notes


def is_valid_figure_label(
    normalized_label: str | None,
    caption: str,
) -> bool:
    if not normalized_label or not re.fullmatch(
        r"[a-z]?\d{1,2}[a-z]?",
        normalized_label,
    ):
        return False

    if not _plausible_figure_number(normalized_label):
        return False

    identity = parse_figure_identity(caption)
    if identity is None:
        # Label alone is not enough — avoids section headings / body text
        # that GROBID mis-tags as figures (e.g. "3.1 Structural... Fig. 1").
        return False

    return identity.number == normalized_label


def resolve_one_based_figure_id(
    grobid_figure_id: str | None,
    normalized_label: str | None,
    caption: str,
) -> str:
    identity = parse_figure_identity(caption)
    if identity is not None:
        if (
            normalized_label
            and is_valid_figure_label(normalized_label, caption)
            and normalized_label == identity.number
        ):
            return identity.figure_id
        return identity.figure_id

    if is_valid_figure_label(normalized_label, caption):
        return f"fig_{normalized_label}"

    if grobid_figure_id:
        match = GROBID_FIGURE_ID_PATTERN.match(grobid_figure_id)

        if match:
            return f"fig_{int(match.group(1)) + 1}"

        return grobid_figure_id

    return make_stable_id("figure", caption)


def figure_metadata_from_element(
    figure_element: etree._Element,
) -> tuple[str | None, str | None, str]:
    label = xpath_text(
        figure_element,
        "./tei:head/tei:label[1]",
    )

    if not label:
        label = xpath_text(
            figure_element,
            "./tei:label[1]",
        )

    fig_desc = normalize_whitespace(
        " ".join(dict.fromkeys(xpath_all_text(figure_element, "./tei:figDesc")))
    )
    head_text = normalize_whitespace(
        " ".join(
            dict.fromkeys(
                xpath_all_text(figure_element, "./tei:head | ./tei:label")
            )
        )
    )
    # Nature TEI often parks Extended Data label lists in <head> while
    # <figDesc> holds only the descriptive paragraph for the first panel.
    head_identity = parse_figure_identity(head_text)
    desc_identity = parse_figure_identity(fig_desc)

    if (
        head_identity is not None
        and head_identity.kind != FIGURE_KIND_MAIN
        and (desc_identity is None or desc_identity.kind == FIGURE_KIND_MAIN)
    ):
        caption = normalize_whitespace(f"{head_text} {fig_desc}".strip())
    elif fig_desc:
        caption = fig_desc
        # If figDesc has no leading identity, prepend head when it provides one.
        if desc_identity is None and head_identity is not None:
            caption = normalize_whitespace(f"{head_text} {fig_desc}".strip())
    else:
        caption = normalize_whitespace(
            " ".join(
                dict.fromkeys(
                    xpath_all_text(
                        figure_element,
                        "./tei:figDesc | ./tei:head | ./tei:p",
                    )
                )
            )
        )

    identity = parse_figure_identity(caption) or parse_figure_identity(label)
    normalized = identity.number if identity else normalize_figure_label(label)
    return label, normalized, caption


def _coords_area(coords_string: str | None) -> float:
    if not coords_string or not str(coords_string).strip():
        return 0.0
    total = 0.0
    for raw_box in str(coords_string).split(";"):
        fields = raw_box.strip().split(",")
        if len(fields) != 5:
            continue
        try:
            width = float(fields[3])
            height = float(fields[4])
        except ValueError:
            continue
        if width > 0 and height > 0:
            total += width * height
    return total


def _graphic_sized_coords_area(coords_string: str | None) -> float:
    """Area of GROBID boxes large enough to be plots (not caption lines)."""
    if not coords_string or not str(coords_string).strip():
        return 0.0
    total = 0.0
    for raw_box in str(coords_string).split(";"):
        fields = raw_box.strip().split(",")
        if len(fields) != 5:
            continue
        try:
            width = float(fields[3])
            height = float(fields[4])
        except ValueError:
            continue
        if width >= 50.0 and height >= 30.0:
            total += width * height
    return total


def tei_figure_graphic_score(figure_element: etree._Element) -> float:
    """Higher = more likely a real graphic figure vs caption/body fragment."""
    graphic_coords = ";".join(
        str(value).strip()
        for value in figure_element.xpath(
            ".//tei:graphic/@coords",
            namespaces=NS,
        )
        if str(value).strip()
    )
    element_coords = figure_element.get("coords")
    graphic_area = _coords_area(graphic_coords)
    sized_area = _graphic_sized_coords_area(element_coords)
    has_graphic_node = 1.0 if graphic_coords else 0.0
    return graphic_area * 10.0 + sized_area + has_graphic_node * 1000.0


def reindex_tei_figure_ids(tei_xml: bytes) -> bytes:
    """
    Rewrite GROBID's 0-based figure xml:id values to paper labels.

    Prefers graphic-backed figures when labels collide, keeps Extended Data /
    Supplementary figures in separate id namespaces, and avoids giving
    caption-only phantoms the canonical ``fig_N`` id.
    """
    xml_parser = etree.XMLParser(
        recover=True,
        remove_blank_text=True,
        huge_tree=True,
        resolve_entities=False,
        no_network=True,
    )
    root = etree.fromstring(tei_xml, parser=xml_parser)

    figure_entries: list[
        tuple[etree._Element, str, str | None, str, float, FigureIdentity | None]
    ] = []

    for figure_element in root.xpath(
        ".//tei:text/tei:body//tei:figure[not(@type='table')]",
        namespaces=NS,
    ):
        old_id = figure_element.get(XML_ID)

        if not old_id:
            continue

        label, normalized_label, caption = figure_metadata_from_element(
            figure_element
        )
        identity = parse_figure_identity(caption)
        if identity is None and is_valid_figure_label(normalized_label, caption):
            identity = FigureIdentity(FIGURE_KIND_MAIN, normalized_label)
        score = tei_figure_graphic_score(figure_element)
        figure_entries.append(
            (
                figure_element,
                old_id,
                normalized_label,
                caption,
                score,
                identity,
            )
        )

    id_map: dict[str, str] = {}
    used_ids: set[str] = set()

    def assign_id(old_id: str, candidate: str, *, allow_suffix: bool) -> str:
        base_id = candidate
        counter = 2
        new_id = candidate

        while new_id in used_ids:
            if not allow_suffix:
                # Caption-only / low-evidence duplicates should not mint
                # fig_N_2 names that look like real alternate extractions.
                new_id = make_stable_id("figure", f"{old_id}:{candidate}")
                break
            new_id = f"{base_id}_{counter}"
            counter += 1

        id_map[old_id] = new_id
        used_ids.add(new_id)
        return new_id

    # Pass 1: caption/label identities, highest graphic score first so the
    # real crop owns fig_N / ed_fig_N instead of a phantom sibling.
    ranked = sorted(
        figure_entries,
        key=lambda item: (-item[4], item[1]),
    )
    for figure_element, old_id, normalized_label, caption, score, identity in ranked:
        if old_id in id_map:
            continue
        if identity is None:
            continue
        allow_suffix = score > 0.0
        new_id = assign_id(old_id, identity.figure_id, allow_suffix=allow_suffix)
        figure_element.set(XML_ID, new_id)

    # Pass 2: remaining figures — GROBID index fallback (1-based).
    for figure_element, old_id, normalized_label, caption, score, identity in figure_entries:
        if old_id in id_map:
            continue

        match = GROBID_FIGURE_ID_PATTERN.match(old_id)
        if match:
            candidate = f"fig_{int(match.group(1)) + 1}"
            # Avoid colliding with an already-claimed main figure number when
            # this node has no graphic evidence.
            allow_suffix = score > 0.0
            if candidate in used_ids and score <= 0.0:
                candidate = make_stable_id("figure", f"{old_id}:{caption[:80]}")
                allow_suffix = False
            new_id = assign_id(old_id, candidate, allow_suffix=allow_suffix)
            figure_element.set(XML_ID, new_id)
            continue

        new_id = assign_id(old_id, old_id, allow_suffix=True)
        figure_element.set(XML_ID, new_id)

    for reference in root.xpath(".//tei:ref[@target]", namespaces=NS):
        target = reference.get("target")

        if not target or not target.startswith("#"):
            continue

        mapped_id = id_map.get(target[1:])

        if mapped_id:
            reference.set("target", f"#{mapped_id}")

    return etree.tostring(
        root,
        encoding="UTF-8",
        xml_declaration=True,
    )


def refine_figure_identities(figures: list[Figure]) -> list[str]:
    """
    Repair figure_id / label collisions on an already-parsed figure list.

    Returns human-readable validation notes. Mutates ``figures`` in place so
    downstream cropping writes the corrected filenames. Safe to run on TEI that
    was reindexed by an older pipeline.
    """
    notes: list[str] = []
    if not figures:
        return notes

    scored: list[tuple[int, float, FigureIdentity | None]] = []
    for index, figure in enumerate(figures):
        graphic_area = _coords_area(figure.graphic_coords)
        sized_area = _graphic_sized_coords_area(figure.coords)
        score = graphic_area * 10.0 + sized_area + (
            1000.0 if figure.graphic_coords else 0.0
        )
        identity = parse_figure_identity(figure.caption)
        if identity is None and is_valid_figure_label(
            figure.normalized_label,
            figure.caption,
        ):
            identity = FigureIdentity(FIGURE_KIND_MAIN, figure.normalized_label)
        scored.append((index, score, identity))

    used_ids: set[str] = set()
    assignments: dict[int, str] = {}

    # Highest evidence first.
    for index, score, identity in sorted(scored, key=lambda item: (-item[1], item[0])):
        if identity is None:
            continue
        candidate = identity.figure_id
        if candidate in used_ids:
            if score <= 0.0:
                # Drop phantom ownership of fig_N_2.
                new_id = make_stable_id(
                    "figure",
                    f"{figures[index].figure_id}:{figures[index].caption[:80]}",
                )
                notes.append(
                    f"demoted caption-only duplicate of {candidate} "
                    f"({figures[index].figure_id} → {new_id})"
                )
            else:
                suffix = 2
                new_id = f"{candidate}_{suffix}"
                while new_id in used_ids:
                    suffix += 1
                    new_id = f"{candidate}_{suffix}"
                notes.append(
                    f"kept secondary graphic for {candidate} as {new_id}"
                )
        else:
            new_id = candidate
            old_id = figures[index].figure_id
            if old_id and old_id != new_id:
                notes.append(f"renamed {old_id} → {new_id}")
        assignments[index] = new_id
        used_ids.add(new_id)

    for index, score, identity in scored:
        if index in assignments:
            continue
        figure = figures[index]
        old_id = figure.figure_id or make_stable_id("figure", figure.caption)
        # Strip legacy collision suffixes when the base id is free and this
        # row has the graphic (e.g. fig_1_2 with image, fig_1 empty).
        base_match = re.match(
            r"^((?:ed_|supp_)?fig_\d+[a-z]?)_(\d+)$",
            old_id,
            flags=re.IGNORECASE,
        )
        if base_match and score > 0.0:
            base_id = base_match.group(1)
            if base_id not in used_ids:
                assignments[index] = base_id
                used_ids.add(base_id)
                notes.append(f"collapsed collision suffix {old_id} → {base_id}")
                continue
        # Body-text / caption-only fragments should not keep fig_N or fig_N_2 ids.
        if identity is None and score <= 0.0:
            new_id = make_stable_id(
                "figure",
                f"{old_id}:{figure.caption[:80]}",
            )
            assignments[index] = new_id
            used_ids.add(new_id)
            if old_id != new_id:
                notes.append(f"demoted non-figure fragment {old_id} → {new_id}")
            continue
        if old_id in used_ids:
            new_id = make_stable_id("figure", f"{old_id}:{figure.caption[:40]}")
            assignments[index] = new_id
            used_ids.add(new_id)
        else:
            assignments[index] = old_id
            used_ids.add(old_id)

    for index, figure in enumerate(figures):
        new_id = assignments[index]
        identity = parse_figure_identity(figure.caption)
        figure.figure_id = new_id
        if identity is not None:
            figure.normalized_label = identity.number
            if not figure.label:
                kind_name = {
                    FIGURE_KIND_EXTENDED: "Extended Data Fig.",
                    FIGURE_KIND_SUPPLEMENTARY: "Supplementary Fig.",
                }.get(identity.kind, "Fig.")
                figure.label = f"{kind_name} {identity.number}"

    notes.extend(validate_extracted_figures(figures))
    return notes


def validate_extracted_figures(figures: list[Figure]) -> list[str]:
    """
    Lightweight checks for duplicate ids, collision suffixes, and mismatches.
    """
    warnings: list[str] = []
    ids = [figure.figure_id for figure in figures if figure.figure_id]
    seen: set[str] = set()
    for figure_id in ids:
        if figure_id in seen:
            warnings.append(f"duplicate figure_id: {figure_id}")
        seen.add(figure_id)

    by_number: dict[tuple[str, str], list[Figure]] = {}
    for figure in figures:
        identity = parse_figure_identity(figure.caption)
        if identity is None and figure.figure_id:
            match = re.match(
                r"^(fig|ed_fig|supp_fig)_(\d+[a-z]?)",
                figure.figure_id,
                flags=re.IGNORECASE,
            )
            if match:
                identity = FigureIdentity(match.group(1).lower(), match.group(2))
        if identity is None:
            continue
        by_number.setdefault((identity.kind, identity.number), []).append(figure)

    for (kind, number), group in by_number.items():
        with_images = [
            figure
            for figure in group
            if figure.image_paths or figure.graphic_coords
        ]
        if len(with_images) > 1:
            warnings.append(
                f"multiple extractions for {kind}_{number}: "
                + ", ".join(str(figure.figure_id) for figure in with_images)
            )
        collision = [
            figure
            for figure in group
            if figure.figure_id
            and re.match(
                r"^(?:ed_|supp_)?fig_\d+[a-z]?_\d+$",
                figure.figure_id,
                flags=re.IGNORECASE,
            )
            and not figure.graphic_coords
            and not figure.image_paths
        ]
        for figure in collision:
            warnings.append(
                f"caption-only collision id {figure.figure_id} "
                f"for {kind}_{number}"
            )

        for figure in group:
            if not figure.figure_id:
                continue
            caption_id = figure_id_from_caption(figure.caption)
            if not caption_id:
                continue
            # Ignore page suffixes and collision suffixes for comparison.
            stem = re.sub(r"_page_\d+$", "", figure.figure_id)
            stem = re.sub(r"_\d+$", "", stem) if stem.count("_") >= 2 else stem
            base = caption_id
            if not stem.startswith(base) and stem != base:
                # fig_1 vs ed_fig_1 is a real mismatch worth surfacing.
                if stem.split("_")[-1] == base.split("_")[-1] and stem != base:
                    warnings.append(
                        f"kind mismatch for {figure.figure_id}: "
                        f"caption implies {caption_id}"
                    )

    return warnings


def extract_figure_references(text: str) -> list[str]:
    references: list[str] = []

    for match in FIGURE_REFERENCE_PATTERN.finditer(text):
        label = normalize_figure_label(match.group("number"))

        if label and label not in references:
            references.append(label)

    return references


def sentence_is_specific_to_figure(
    sentence: str,
    figure_label: str,
) -> bool:
    references = extract_figure_references(sentence)

    if figure_label not in references:
        return False

    if len(references) == 1:
        return True

    for clause in CLAUSE_SPLIT_PATTERN.split(sentence):
        clause = clause.strip()

        if not clause:
            continue

        clause_references = extract_figure_references(clause)

        if figure_label not in clause_references:
            continue

        if len(clause_references) == 1:
            return True

        if re.search(
            rf"\bin\s+fig(?:ure)?s?\.?\s*{re.escape(figure_label)}\s*[,]",
            clause,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def extract_figure_specific_sentences(
    text: str,
    figure_label: str,
) -> list[str]:
    sentences: list[str] = []

    for sentence in split_into_sentences(text):
        if sentence_is_specific_to_figure(sentence, figure_label):
            sentences.append(sentence)

    return sentences


def extract_figure_specific_text(
    text: str,
    figure_label: str,
) -> str | None:
    sentences = extract_figure_specific_sentences(text, figure_label)

    if not sentences:
        return None

    return " ".join(sentences)


def extract_figure_mention_sentences(
    text: str,
    figure_label: str,
) -> list[str]:
    sentences: list[str] = []

    for sentence in split_into_sentences(text):
        if figure_label in extract_figure_references(sentence):
            sentences.append(sentence)

    return sentences


def extract_figure_mention_text(
    text: str,
    figure_label: str,
) -> str | None:
    sentences = extract_figure_mention_sentences(text, figure_label)

    if not sentences:
        return None

    return " ".join(sentences)


def is_relevant_context_paragraph(
    paragraph: Paragraph,
    figure: Figure,
    figure_label: str,
    anchor_paragraphs: list[Paragraph],
) -> bool:
    if paragraph.paragraph_id in figure.referenced_by_paragraph_ids:
        return False

    if (
        paragraph.figure_references
        and figure_label not in paragraph.figure_references
    ):
        return False

    anchor_sections = {
        tuple(anchor.section_path)
        for anchor in anchor_paragraphs
    }
    same_section = tuple(paragraph.section_path) in anchor_sections
    in_context_window = (
        paragraph.paragraph_id in figure.context_paragraph_ids
    )

    if same_section and paragraph.xrd_score >= CONTEXT_SECTION_XRD_THRESHOLD:
        return True

    if not in_context_window:
        return False

    if same_section:
        return paragraph.xrd_score >= CONTEXT_WINDOW_XRD_THRESHOLD

    return (
        paragraph.section_type in {"results", "methods"}
        and paragraph.xrd_score >= CONTEXT_WINDOW_XRD_THRESHOLD
    )


def extract_peak_positions(text: str) -> list[float]:
    positions: list[float] = []

    for match in PEAK_POSITION_PATTERN.finditer(text):
        try:
            value = float(match.group("value"))
        except ValueError:
            continue

        if 0.0 <= value <= 180.0 and value not in positions:
            positions.append(value)

    return positions


def extract_miller_indices(text: str) -> list[str]:
    values: list[str] = []

    for match in MILLER_INDEX_PATTERN.finditer(text):
        value = normalize_whitespace(match.group(0))

        if value not in values:
            values.append(value)

    return values


def score_xrd_text(text: str) -> float:
    normalized = normalize_for_matching(text)

    if not normalized:
        return 0.0

    score = 0.0

    strong_terms = {
        "xrd": 4.0,
        "x-ray diffraction": 4.0,
        "x ray diffraction": 4.0,
        "powder diffraction": 4.0,
        "diffractogram": 4.0,
        "rietveld": 4.0,
        "2 theta": 3.0,
        "2θ": 3.0,
        "bragg": 2.0,
        # Captions often say "diffraction planes/peaks" without "XRD" or
        # "diffraction pattern"; bare "diffraction" is medium-strong so it
        # still needs a little extra signal (figure label, planes, etc.).
        "diffraction": 2.0,
        "diffraction pattern": 3.0,
        "diffraction patterns": 3.0,
        "diffraction peak": 3.0,
        "diffraction peaks": 3.0,
        "xrd pattern": 4.0,
        "pxrd": 4.0,
    }

    for term, weight in strong_terms.items():
        if normalize_for_matching(term) in normalized:
            score += weight

    for term in XRD_ANALYSIS_TERMS:
        if normalize_for_matching(term) in normalized:
            score += 0.4

    if THETA_PATTERN.search(text):
        score += 2.0

    peak_positions = extract_peak_positions(text)
    score += min(len(peak_positions), 5) * 0.5

    miller_indices = extract_miller_indices(text)
    score += min(len(miller_indices), 5) * 0.5

    if "figure" in normalized or "fig " in f"{normalized} ":
        score += 0.5

    return round(score, 3)


def make_stable_id(prefix: str, *parts: str) -> str:
    content = "\n".join(parts)
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class GrobidClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        timeout_seconds: int = 180,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()

    def is_alive(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}/api/isalive",
                timeout=10,
            )
            return response.ok
        except requests.RequestException:
            return False

    def process_fulltext(
        self,
        pdf_path: str | Path,
        include_coordinates: bool = True,
        consolidate_header: int = 1,
        consolidate_citations: int = 0,
    ) -> bytes:
        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        endpoint = f"{self.base_url}/api/processFulltextDocument"

        data: list[tuple[str, str]] = [
            ("consolidateHeader", str(consolidate_header)),
            ("consolidateCitations", str(consolidate_citations)),
            ("includeRawCitations", "1"),
            ("includeRawAffiliations", "1"),
            ("segmentSentences", "1"),
        ]

        if include_coordinates:
            for coordinate_type in (
                "figure",
                "head",
                "p",
                "ref",
                "s",
                "table",
                "formula",
            ):
                data.append(("teiCoordinates", coordinate_type))

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with pdf_path.open("rb") as pdf_file:
                    response = self.session.post(
                        endpoint,
                        files={
                            "input": (
                                pdf_path.name,
                                pdf_file,
                                "application/pdf",
                            )
                        },
                        data=data,
                        timeout=self.timeout_seconds,
                    )

                if response.status_code == 503:
                    raise RuntimeError(
                        "GROBID is temporarily unavailable or overloaded."
                    )

                response.raise_for_status()

                if not response.content.strip():
                    raise RuntimeError("GROBID returned an empty response.")

                return response.content

            except (
                requests.RequestException,
                RuntimeError,
            ) as exc:
                last_error = exc

                if attempt == self.max_retries:
                    break

                delay = 2 ** (attempt - 1)

                LOGGER.warning(
                    "GROBID request failed on attempt %d/%d: %s. "
                    "Retrying in %d seconds.",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Unable to process PDF with GROBID: {last_error}"
        )


class TEIXRDParser:
    def parse(
        self,
        tei_xml: bytes | str,
        source_pdf: str | Path,
    ) -> ParsedDocument:
        xml_parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            huge_tree=True,
            resolve_entities=False,
            no_network=True,
        )

        if isinstance(tei_xml, str):
            tei_xml = tei_xml.encode("utf-8")

        root = etree.fromstring(tei_xml, parser=xml_parser)
        source_path = Path(source_pdf)

        metadata = self._extract_metadata(root)
        paragraphs: list[Paragraph] = []
        sections = self._extract_sections(root, paragraphs)
        figures = self._extract_figures(root)
        refine_notes = refine_figure_identities(figures)
        for note in refine_notes:
            LOGGER.info("Figure identity refine: %s", note)
        tables = self._extract_tables(root)
        references = self._extract_references(root)

        self._link_figures_to_paragraphs(
            figures=figures,
            paragraphs=paragraphs,
        )

        results_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if paragraph.section_type == "results"
        ]

        xrd_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if paragraph.xrd_score >= 2.0
        ]

        xrd_figures = [
            figure
            for figure in figures
            if figure.is_likely_xrd
        ]

        return ParsedDocument(
            source_pdf=str(source_path),
            source_sha256=file_sha256(source_path),
            metadata=metadata,
            sections=sections,
            paragraphs=paragraphs,
            figures=figures,
            tables=tables,
            references=references,
            counts={
                "sections": self._count_sections(sections),
                "paragraphs": len(paragraphs),
                "results_paragraphs": len(results_paragraphs),
                "xrd_paragraphs": len(xrd_paragraphs),
                "figures": len(figures),
                "xrd_figures": len(xrd_figures),
                "tables": len(tables),
                "references": len(references),
            },
        )

    def _extract_metadata(
        self,
        root: etree._Element,
    ) -> DocumentMetadata:
        title = xpath_text(
            root,
            ".//tei:teiHeader/tei:fileDesc/"
            "tei:titleStmt/tei:title[@type='main'][1]",
        )

        if not title:
            title = xpath_text(
                root,
                ".//tei:teiHeader/tei:fileDesc/"
                "tei:titleStmt/tei:title[1]",
            )

        abstract_element = xpath_first(
            root,
            ".//tei:teiHeader/tei:profileDesc/tei:abstract[1]",
        )

        abstract = (
            element_text(abstract_element)
            if isinstance(abstract_element, etree._Element)
            else None
        )

        doi = xpath_text(
            root,
            ".//tei:teiHeader//tei:idno["
            "translate(@type, 'DOI', 'doi')='doi'][1]",
        )

        if not doi:
            header_text = element_text(
                xpath_first(root, ".//tei:teiHeader")
            )
            doi_match = DOI_PATTERN.search(header_text)

            if doi_match:
                doi = doi_match.group(0).rstrip(".,;)")

        journal = xpath_text(
            root,
            ".//tei:teiHeader//tei:monogr/tei:title[1]",
        )

        publication_date = xpath_text(
            root,
            ".//tei:teiHeader//tei:publicationStmt/"
            "tei:date/@when",
        )

        if not publication_date:
            publication_date = xpath_text(
                root,
                ".//tei:teiHeader//tei:publicationStmt/tei:date[1]",
            )

        keyword_values = xpath_all_text(
            root,
            ".//tei:teiHeader/tei:profileDesc/"
            "tei:textClass/tei:keywords//tei:term",
        )

        authors = self._extract_authors(root)

        return DocumentMetadata(
            title=title,
            doi=doi,
            abstract=abstract,
            journal=journal,
            publication_date=publication_date,
            authors=authors,
            keywords=keyword_values,
        )

    def _extract_authors(
        self,
        root: etree._Element,
    ) -> list[Author]:
        authors: list[Author] = []

        author_elements = root.xpath(
            ".//tei:teiHeader/tei:fileDesc/"
            "tei:sourceDesc//tei:author",
            namespaces=NS,
        )

        for author_element in author_elements:
            given_names = xpath_all_text(
                author_element,
                ".//tei:persName/tei:forename",
            )

            given_name = given_names[0] if given_names else None
            middle_name = (
                " ".join(given_names[1:])
                if len(given_names) > 1
                else None
            )

            surname = xpath_text(
                author_element,
                ".//tei:persName/tei:surname[1]",
            )

            email = xpath_text(
                author_element,
                ".//tei:email[1]",
            )

            orcid = xpath_text(
                author_element,
                ".//tei:idno["
                "contains(translate(@type, 'ORCID', 'orcid'), 'orcid')"
                "][1]",
            )

            affiliations = xpath_all_text(
                author_element,
                ".//tei:affiliation",
            )

            affiliation = (
                "; ".join(dict.fromkeys(affiliations))
                if affiliations
                else None
            )

            if given_name or surname or email:
                authors.append(
                    Author(
                        given_name=given_name,
                        middle_name=middle_name,
                        surname=surname,
                        email=email,
                        affiliation=affiliation,
                        orcid=orcid,
                    )
                )

        return authors

    def _extract_sections(
        self,
        root: etree._Element,
        all_paragraphs: list[Paragraph],
    ) -> list[Section]:
        body = xpath_first(root, ".//tei:text/tei:body[1]")

        if not isinstance(body, etree._Element):
            return []

        sections: list[Section] = []

        for div_index, div in enumerate(
            body.xpath("./tei:div", namespaces=NS)
        ):
            section = self._parse_div(
                div=div,
                parent_path=[],
                sibling_index=div_index,
                all_paragraphs=all_paragraphs,
            )
            sections.append(section)

        direct_paragraphs = body.xpath("./tei:p", namespaces=NS)

        if direct_paragraphs:
            section = Section(
                section_id="section_body_unclassified",
                title="Unclassified body text",
                path=["Unclassified body text"],
                section_type="other",
            )

            for index, paragraph_element in enumerate(direct_paragraphs):
                paragraph = self._parse_paragraph(
                    paragraph_element=paragraph_element,
                    section_path=section.path,
                    section_type=section.section_type,
                    paragraph_index=index,
                )

                if paragraph:
                    section.paragraphs.append(paragraph)
                    all_paragraphs.append(paragraph)

            sections.insert(0, section)

        return sections

    def _parse_div(
        self,
        div: etree._Element,
        parent_path: list[str],
        sibling_index: int,
        all_paragraphs: list[Paragraph],
    ) -> Section:
        head = xpath_text(div, "./tei:head[1]")
        title = head or f"Untitled section {sibling_index + 1}"

        path = [*parent_path, title]
        section_type = classify_section(title)

        if section_type == "other" and parent_path:
            parent_type = classify_section(parent_path[-1])

            if parent_type in {"results", "methods"}:
                section_type = parent_type

        xml_id = div.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        section_id = (
            xml_id
            or make_stable_id(
                "section",
                "/".join(path),
                str(sibling_index),
            )
        )

        section = Section(
            section_id=section_id,
            title=title,
            path=path,
            section_type=section_type,
        )

        direct_paragraphs = div.xpath("./tei:p", namespaces=NS)

        for paragraph_index, paragraph_element in enumerate(
            direct_paragraphs
        ):
            paragraph = self._parse_paragraph(
                paragraph_element=paragraph_element,
                section_path=path,
                section_type=section_type,
                paragraph_index=paragraph_index,
            )

            if paragraph:
                section.paragraphs.append(paragraph)
                all_paragraphs.append(paragraph)

        child_divs = div.xpath("./tei:div", namespaces=NS)

        for child_index, child_div in enumerate(child_divs):
            child_section = self._parse_div(
                div=child_div,
                parent_path=path,
                sibling_index=child_index,
                all_paragraphs=all_paragraphs,
            )
            section.children.append(child_section)

        return section

    def _parse_paragraph(
        self,
        paragraph_element: etree._Element,
        section_path: list[str],
        section_type: str,
        paragraph_index: int,
    ) -> Paragraph | None:
        text = element_text(paragraph_element)

        if not text:
            return None

        xml_id = paragraph_element.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        paragraph_id = (
            xml_id
            or make_stable_id(
                "paragraph",
                "/".join(section_path),
                str(paragraph_index),
                text,
            )
        )

        return Paragraph(
            paragraph_id=paragraph_id,
            text=text,
            section_path=section_path.copy(),
            section_type=section_type,
            figure_references=extract_figure_references(text),
            xrd_score=score_xrd_text(text),
            peak_positions=extract_peak_positions(text),
            miller_indices=extract_miller_indices(text),
        )

    def _extract_figures(
        self,
        root: etree._Element,
    ) -> list[Figure]:
        figures: list[Figure] = []

        figure_elements = root.xpath(
            ".//tei:text/tei:body//tei:figure[not(@type='table')]",
            namespaces=NS,
        )

        for figure_element in figure_elements:
            grobid_figure_id = figure_element.get(XML_ID)
            label, normalized_label, caption = figure_metadata_from_element(
                figure_element
            )
            figure_id = grobid_figure_id or resolve_one_based_figure_id(
                grobid_figure_id=grobid_figure_id,
                normalized_label=normalized_label,
                caption=caption,
            )

            graphic_targets = [
                str(value)
                for value in figure_element.xpath(
                    ".//tei:graphic/@url | .//tei:graphic/@target",
                    namespaces=NS,
                )
                if str(value).strip()
            ]

            coords = figure_element.get("coords")
            graphic_coord_values = [
                str(value).strip()
                for value in figure_element.xpath(
                    ".//tei:graphic/@coords",
                    namespaces=NS,
                )
                if str(value).strip()
            ]
            graphic_coords = (
                ";".join(graphic_coord_values)
                if graphic_coord_values
                else None
            )

            figure_type = figure_element.get("type")
            xrd_score = score_xrd_text(caption)
            is_caption_xrd = xrd_score >= 2.5

            identity = parse_figure_identity(caption)
            resolved_normalized = (
                identity.number
                if identity is not None
                else normalized_label
            )
            figures.append(
                Figure(
                    figure_id=figure_id,
                    label=label,
                    normalized_label=resolved_normalized,
                    caption=caption,
                    graphic_targets=graphic_targets,
                    coords=coords,
                    graphic_coords=graphic_coords,
                    figure_type=figure_type,
                    xrd_score=xrd_score,
                    is_likely_xrd=is_caption_xrd,
                    is_caption_xrd=is_caption_xrd,
                )
            )

        return figures

    def _extract_tables(
        self,
        root: etree._Element,
    ) -> list[Table]:
        tables: list[Table] = []

        table_figures = root.xpath(
            ".//tei:text/tei:body//tei:figure[@type='table']",
            namespaces=NS,
        )

        for table_element in table_figures:
            table_id = table_element.get(
                "{http://www.w3.org/XML/1998/namespace}id"
            )

            label = xpath_text(
                table_element,
                "./tei:head/tei:label[1] | ./tei:label[1]",
            )

            caption = xpath_text(
                table_element,
                "./tei:figDesc[1] | ./tei:head[1]",
            ) or ""

            table_node = xpath_first(
                table_element,
                ".//tei:table[1]",
            )

            text = (
                element_text(table_node)
                if isinstance(table_node, etree._Element)
                else element_text(table_element)
            )

            tables.append(
                Table(
                    table_id=table_id,
                    label=label,
                    caption=caption,
                    text=text,
                )
            )

        return tables

    def _extract_references(
        self,
        root: etree._Element,
    ) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []

        entries = root.xpath(
            ".//tei:text/tei:back//tei:listBibl/tei:biblStruct",
            namespaces=NS,
        )

        for entry in entries:
            reference_id = entry.get(
                "{http://www.w3.org/XML/1998/namespace}id"
            )

            title = xpath_text(
                entry,
                ".//tei:analytic/tei:title[1] | "
                ".//tei:monogr/tei:title[1]",
            )

            doi = xpath_text(
                entry,
                ".//tei:idno["
                "translate(@type, 'DOI', 'doi')='doi'][1]",
            )

            year = xpath_text(
                entry,
                ".//tei:imprint/tei:date/@when",
            )

            if not year:
                year = xpath_text(
                    entry,
                    ".//tei:imprint/tei:date[1]",
                )

            authors: list[str] = []

            for author in entry.xpath(
                ".//tei:analytic/tei:author",
                namespaces=NS,
            ):
                given = xpath_text(
                    author,
                    ".//tei:forename[1]",
                )
                surname = xpath_text(
                    author,
                    ".//tei:surname[1]",
                )

                name = normalize_whitespace(
                    " ".join(
                        part
                        for part in (given, surname)
                        if part
                    )
                )

                if name:
                    authors.append(name)

            raw_text = element_text(entry)

            references.append(
                {
                    "reference_id": reference_id,
                    "title": title,
                    "doi": doi,
                    "year": year,
                    "authors": authors,
                    "raw_text": raw_text,
                }
            )

        return references

    def _link_figures_to_paragraphs(
        self,
        figures: list[Figure],
        paragraphs: list[Paragraph],
        context_window: int = CONTEXT_WINDOW,
    ) -> None:
        label_to_figure: dict[str, Figure] = {}

        for figure in figures:
            if figure.normalized_label:
                label_to_figure[figure.normalized_label] = figure

        for paragraph_index, paragraph in enumerate(paragraphs):
            linked_figures: list[Figure] = []

            for reference in paragraph.figure_references:
                figure = label_to_figure.get(reference)

                if figure is None:
                    continue

                if (
                    paragraph.paragraph_id
                    not in figure.referenced_by_paragraph_ids
                ):
                    figure.referenced_by_paragraph_ids.append(
                        paragraph.paragraph_id
                    )

                linked_figures.append(figure)

            for figure in linked_figures:
                start = max(0, paragraph_index - context_window)
                end = min(
                    len(paragraphs),
                    paragraph_index + context_window + 1,
                )

                for context_paragraph in paragraphs[start:end]:
                    if (
                        context_paragraph.paragraph_id
                        not in figure.context_paragraph_ids
                    ):
                        figure.context_paragraph_ids.append(
                            context_paragraph.paragraph_id
                        )

                    if context_paragraph.xrd_score >= 2.0:
                        figure.xrd_score += 0.5

                figure.xrd_score = round(figure.xrd_score, 3)

                if figure.xrd_score >= 2.5:
                    figure.is_likely_xrd = True

        # A figure may be obviously XRD from surrounding discussion even
        # when its caption is generic.
        for figure in figures:
            context_ids = set(figure.context_paragraph_ids)

            context_score = sum(
                paragraph.xrd_score
                for paragraph in paragraphs
                if paragraph.paragraph_id in context_ids
            )

            if context_score >= 4.0:
                figure.is_likely_xrd = True

    def _count_sections(
        self,
        sections: list[Section],
    ) -> int:
        total = 0

        for section in sections:
            total += 1
            total += self._count_sections(section.children)

        return total


def collect_figure_analysis_text(
    document: ParsedDocument,
    figure: Figure,
) -> str:
    figure_label = figure.normalized_label

    if not figure_label:
        return ""

    paragraph_index = {
        paragraph.paragraph_id: index
        for index, paragraph in enumerate(document.paragraphs)
    }

    anchor_paragraphs = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.paragraph_id in figure.referenced_by_paragraph_ids
    ]

    selected_entries: list[tuple[int, str]] = []
    seen_ids: set[str] = set()

    for paragraph in document.paragraphs:
        if figure_label not in paragraph.figure_references:
            continue

        relevant_text = extract_figure_mention_text(
            paragraph.text,
            figure_label,
        )

        if not relevant_text:
            continue

        selected_entries.append(
            (paragraph_index[paragraph.paragraph_id], relevant_text)
        )
        seen_ids.add(paragraph.paragraph_id)

    for paragraph in document.paragraphs:
        if paragraph.paragraph_id in seen_ids:
            continue

        if not is_relevant_context_paragraph(
            paragraph=paragraph,
            figure=figure,
            figure_label=figure_label,
            anchor_paragraphs=anchor_paragraphs,
        ):
            continue

        selected_entries.append(
            (paragraph_index[paragraph.paragraph_id], paragraph.text)
        )
        seen_ids.add(paragraph.paragraph_id)

    if not selected_entries:
        return ""

    selected_entries.sort(key=lambda entry: entry[0])
    return "\n\n".join(text for _, text in selected_entries)


def build_xrd_records(
    document: ParsedDocument,
) -> list[dict[str, Any]]:
    """
    Build training-ready records where each likely XRD figure is paired
    with explicitly mentioned text plus nearby relevant discussion.
    """
    records: list[dict[str, Any]] = []

    for figure in document.figures:
        if not figure.is_likely_xrd:
            continue

        figure_label = figure.normalized_label

        if not figure_label:
            continue

        combined_analysis_text = collect_figure_analysis_text(
            document,
            figure,
        )

        if not combined_analysis_text:
            continue

        analysis_paragraphs = []
        peak_positions: set[float] = set()
        miller_indices: set[str] = set()

        for paragraph in document.paragraphs:
            if figure_label not in paragraph.figure_references:
                continue

            relevant_text = extract_figure_mention_text(
                paragraph.text,
                figure_label,
            )

            if relevant_text:
                analysis_paragraphs.append(
                    {
                        **asdict(paragraph),
                        "relevant_text": relevant_text,
                        "selection_reason": "explicit_figure_reference",
                    }
                )
                peak_positions.update(paragraph.peak_positions)
                miller_indices.update(paragraph.miller_indices)

        records.append(
            {
                "document": {
                    "source_pdf": document.source_pdf,
                    "source_sha256": document.source_sha256,
                    "title": document.metadata.title,
                    "doi": document.metadata.doi,
                    "journal": document.metadata.journal,
                    "publication_date": (
                        document.metadata.publication_date
                    ),
                },
                "figure": asdict(figure),
                "analysis_paragraphs": analysis_paragraphs,
                "combined_analysis_text": combined_analysis_text,
                "extracted_features": {
                    "peak_positions_degrees": sorted(peak_positions),
                    "miller_indices": sorted(miller_indices),
                },
            }
        )

    return records


def format_author_name(author: Author) -> str:
    """Human-readable author name from GROBID Author fields."""
    parts = [
        part
        for part in (author.given_name, author.middle_name, author.surname)
        if part
    ]
    name = normalize_whitespace(" ".join(parts))
    if name:
        return name
    if author.email:
        return author.email
    return "Unknown author"


def figure_entries_from_analysis_payload(payload: Any) -> list[dict[str, Any]]:
    """
    Extract figure entries from figure_analysis.json payload.

    Supports legacy list format and the wrapped document format::

        {"title": ..., "doi": ..., "authors": [...], "figures": [...]}
    """
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        figures = payload.get("figures")
        if isinstance(figures, list):
            return [entry for entry in figures if isinstance(entry, dict)]
    return []


def primary_figure_number_from_stem(stem: str) -> int | None:
    """
    Extract the primary figure number from an extraction stem/id.

    Handles ``fig_1``, ``figure_3``, ``fig_1_2``, ``fig_7_2``, ``ed_fig_3``,
    and ``supp_fig_2`` by taking the first numeric component after an optional
    kind prefix — never the trailing extraction/collision suffix alone.
    """
    raw = str(stem or "").strip()
    if not raw:
        return None
    # Allow accidental filenames with extension.
    if "." in raw and not raw.startswith("."):
        raw = Path(raw).stem
    match = re.match(
        r"^(?:(?:ed|supp)[_-]?)?fig(?:ure)?[_-]?(\d+)(?:[_-].*)?$",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d+)(?:[_-].*)?$", raw)
    if match:
        return int(match.group(1))
    return None


def _resolve_figure_image_path_string(figure: Figure) -> str:
    """Return the best on-disk (or recorded) image path for a GROBID figure."""
    for raw in figure.image_paths or []:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text)
        if path.is_file():
            return str(path.resolve())
    for raw in figure.image_paths or []:
        text = str(raw or "").strip()
        if text:
            return str(Path(text))
    return ""


def _figure_number_for_entry(figure: Figure) -> int | None:
    label = figure.normalized_label
    if label and str(label).isdigit():
        return int(label)
    if figure.figure_id:
        return primary_figure_number_from_stem(figure.figure_id)
    return None


def _entry_text_quality(entry: dict[str, Any]) -> tuple[int, int, int]:
    """Higher is better: caption length, text length, has figure_id."""
    return (
        len(str(entry.get("caption") or "").strip()),
        len(str(entry.get("text") or "").strip()),
        1 if entry.get("figure_id") else 0,
    )


def _prefer_figure_entry(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Prefer the entry with richer caption/context when merging duplicates."""
    if _entry_text_quality(right) > _entry_text_quality(left):
        merged = dict(right)
        # Keep a figure number if only one side has it.
        if merged.get("figure") is None and left.get("figure") is not None:
            merged["figure"] = left["figure"]
        if not merged.get("figure_id") and left.get("figure_id"):
            merged["figure_id"] = left["figure_id"]
        return merged
    merged = dict(left)
    if merged.get("figure") is None and right.get("figure") is not None:
        merged["figure"] = right["figure"]
    if not merged.get("figure_id") and right.get("figure_id"):
        merged["figure_id"] = right["figure_id"]
    # Prefer non-empty path.
    if not merged.get("figure_path") and right.get("figure_path"):
        merged["figure_path"] = right["figure_path"]
    return merged


def _figure_analysis_entry_from_figure(
    document: ParsedDocument,
    figure: Figure,
) -> dict[str, Any] | None:
    caption = normalize_whitespace(figure.caption or "")
    analysis = normalize_whitespace(
        collect_figure_analysis_text(document, figure)
    )
    if caption and analysis:
        text = f"{caption} {analysis}"
    else:
        text = caption or analysis

    figure_path = _resolve_figure_image_path_string(figure)
    if not text and not figure_path:
        return None

    return {
        "figure": _figure_number_for_entry(figure),
        "figure_id": figure.figure_id,
        "figure_path": figure_path,
        "caption": caption,
        "text": text,
    }


def build_figure_analysis_dataset(
    document: ParsedDocument,
    *,
    xrd_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Build figure_analysis entries from GROBID figures.

    Deduplicates by resolved image path (same crop → one entry), not by numeric
    label alone. Image-backed entries always win over text-only duplicates with
    the same label. Distinct crops that share a label are all preserved.
    """
    candidates: list[dict[str, Any]] = []

    for figure in document.figures:
        if xrd_only and not figure.is_caption_xrd:
            continue
        entry = _figure_analysis_entry_from_figure(document, figure)
        if entry is not None:
            candidates.append(entry)

    by_image_path: dict[str, dict[str, Any]] = {}
    without_image: list[dict[str, Any]] = []

    for entry in candidates:
        raw_path = str(entry.get("figure_path") or "").strip()
        if not raw_path:
            without_image.append(entry)
            continue
        path_key = str(Path(raw_path).resolve())
        existing = by_image_path.get(path_key)
        if existing is None:
            by_image_path[path_key] = entry
        else:
            by_image_path[path_key] = _prefer_figure_entry(existing, entry)

    image_backed_numbers = {
        entry.get("figure")
        for entry in by_image_path.values()
        if entry.get("figure") is not None
    }

    # Text/metadata-only rows: keep only when no image-backed row covers that
    # figure number (avoids keeping a bad empty fig_1 over a valid fig_1_2).
    best_without_image: dict[Any, dict[str, Any]] = {}
    unresolved_without_image: list[dict[str, Any]] = []
    for entry in without_image:
        number = entry.get("figure")
        if number is not None and number in image_backed_numbers:
            continue
        if number is None:
            unresolved_without_image.append(entry)
            continue
        existing = best_without_image.get(number)
        if existing is None:
            best_without_image[number] = entry
        else:
            best_without_image[number] = _prefer_figure_entry(existing, entry)

    dataset: list[dict[str, Any]] = list(by_image_path.values())
    dataset.extend(best_without_image.values())
    dataset.extend(unresolved_without_image)

    covered_labels = {
        str(entry["figure"])
        for entry in dataset
        if entry.get("figure") is not None
    }

    if not xrd_only:
        referenced_labels: set[str] = set()
        for paragraph in document.paragraphs:
            referenced_labels.update(paragraph.figure_references)

        for figure_label in sorted(
            referenced_labels - covered_labels,
            key=lambda label: int(label) if label.isdigit() else label,
        ):
            if not figure_label.isdigit():
                continue

            analysis_parts: list[str] = []
            for paragraph in document.paragraphs:
                if figure_label not in paragraph.figure_references:
                    continue
                relevant_text = extract_figure_mention_text(
                    paragraph.text,
                    figure_label,
                )
                if relevant_text:
                    analysis_parts.append(relevant_text)

            analysis = normalize_whitespace(
                "\n\n".join(dict.fromkeys(analysis_parts))
            )
            if not analysis:
                continue

            dataset.append(
                {
                    "figure": int(figure_label),
                    "figure_id": f"fig_{figure_label}",
                    "figure_path": "",
                    "caption": "",
                    "text": analysis,
                    "association_status": "text_only",
                    "association_note": (
                        "Referenced in body text but no extracted image was "
                        "associated during dataset construction."
                    ),
                }
            )

    dataset.sort(
        key=lambda entry: (
            entry.get("figure") is None,
            entry.get("figure") if entry.get("figure") is not None else 0,
            str(entry.get("figure_id") or ""),
            str(entry.get("figure_path") or ""),
        )
    )
    return dataset


def build_figure_analysis_document(
    document: ParsedDocument,
    *,
    xrd_only: bool = False,
) -> dict[str, Any]:
    """Paper metadata + figure entries for ``*.figure_analysis.json``."""
    metadata = document.metadata
    figures = build_figure_analysis_dataset(document, xrd_only=xrd_only)
    # figures_total / xrd_figures are finalized during vision enrichment;
    # seed totals from extracted dataset entries (image-backed preferred).
    image_backed = sum(1 for entry in figures if entry.get("figure_path"))
    return {
        "title": metadata.title,
        "doi": metadata.doi,
        "authors": [format_author_name(author) for author in metadata.authors],
        "figures_total": image_backed or len(figures),
        "xrd_figures": None,
        "figures": figures,
    }


def xrd_figure_extraction_counts(document: ParsedDocument) -> tuple[int, int]:
    xrd_figures = [
        figure for figure in document.figures if figure.is_likely_xrd
    ]
    extracted = sum(
        1 for figure in xrd_figures if figure.image_paths
    )
    return extracted, len(xrd_figures)


def truncate_display_title(title: str | None, *, max_length: int = 80) -> str:
    cleaned = normalize_whitespace(title)
    if not cleaned:
        return "Unknown title"
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 3].rstrip() + "..."


def format_quiet_summary_line(
    index: int,
    extracted_xrd_figures: int | None,
    total_xrd_figures: int | None,
    *,
    success: bool,
    title: str | None = None,
) -> str:
    if success and extracted_xrd_figures is not None and total_xrd_figures is not None:
        figures_summary = f"{extracted_xrd_figures}/{total_xrd_figures}"
    else:
        figures_summary = "-/-"

    status = "Successful" if success else "Failed"
    display_title = truncate_display_title(title)
    return (
        f"PDF {index}: {figures_summary} XRD figures | "
        f"{display_title} | {status}"
    )


def parse_pdf(
    pdf_path: str | Path,
    output_directory: str | Path,
    grobid_url: str = "http://localhost:8070",
    extract_figures: bool = True,
    figure_dpi: int = 300,
    xrd_figures_only: bool = False,
) -> ParsedDocument:
    pdf_path = Path(pdf_path)
    output_directory = Path(output_directory)
    extra_directory = output_directory / "extra"
    output_directory.mkdir(parents=True, exist_ok=True)
    extra_directory.mkdir(parents=True, exist_ok=True)

    pdf_copy_path = extra_directory / pdf_path.name
    if pdf_path.resolve() != pdf_copy_path.resolve():
        shutil.copy2(pdf_path, pdf_copy_path)
        LOGGER.info("Copied source PDF to %s", pdf_copy_path)

    client = GrobidClient(base_url=grobid_url)

    if not client.is_alive():
        raise RuntimeError(
            f"GROBID is not responding at {grobid_url}. "
            "Start the GROBID server before running the parser."
        )

    LOGGER.info("Sending %s to GROBID", pdf_path)

    tei_xml = client.process_fulltext(pdf_path)
    tei_xml = reindex_tei_figure_ids(tei_xml)

    tei_output_path = extra_directory / f"{pdf_path.stem}.tei.xml"
    tei_output_path.write_bytes(tei_xml)

    parser = TEIXRDParser()
    document = parser.parse(
        tei_xml=tei_xml,
        source_pdf=pdf_path,
    )

    if extract_figures:
        figures_directory = output_directory / "figures"
        # P2: split Nature-style merged caption heads into recovery stubs.
        merge_notes = expand_merged_caption_figures(document.figures)
        for note in merge_notes:
            LOGGER.info("Merged-caption: %s", note)

        extracted_count = extract_document_figures(
            pdf_path=pdf_path,
            figures=document.figures,
            output_dir=figures_directory,
            dpi=figure_dpi,
            xrd_only=xrd_figures_only,
        )
        # P1: recover figures GROBID never segmented (PDF caption scan).
        recover_notes = recover_missing_figures_from_pdf(
            pdf_path=pdf_path,
            figures=document.figures,
            output_dir=figures_directory,
            dpi=figure_dpi,
            xrd_only=xrd_figures_only,
        )
        for note in recover_notes:
            LOGGER.info("Orphan recovery: %s", note)

        extracted_with_images = sum(
            1 for figure in document.figures if figure.image_paths
        )
        LOGGER.info(
            "Extracted %d/%d figures to %s (after recovery: %d with images)",
            extracted_count,
            len(document.figures),
            figures_directory,
            extracted_with_images,
        )

        for warning in validate_extracted_figures(document.figures):
            LOGGER.warning("Figure validation: %s", warning)

    parsed_output_path = (
        extra_directory / f"{pdf_path.stem}.parsed.json"
    )

    with parsed_output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            asdict(document),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    records = build_xrd_records(document)

    records_output_path = (
        extra_directory / f"{pdf_path.stem}.xrd_records.json"
    )

    with records_output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            records,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    figure_analysis = build_figure_analysis_document(
        document,
        xrd_only=xrd_figures_only,
    )

    figure_analysis_output_path = (
        output_directory / f"{pdf_path.stem}.figure_analysis.json"
    )

    with figure_analysis_output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            figure_analysis,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info("Source PDF: %s", pdf_copy_path)
    LOGGER.info("TEI XML: %s", tei_output_path)
    LOGGER.info("Parsed document: %s", parsed_output_path)
    LOGGER.info("XRD records: %s", records_output_path)
    LOGGER.info("Figure analysis: %s", figure_analysis_output_path)

    return document


def collect_pdf_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Path not found: {input_path}")

    pdf_paths = sorted(
        {
            path.resolve()
            for path in input_path.rglob("*.pdf")
            if path.is_file()
        }
    )

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_path}")

    return pdf_paths


def resolve_output_directory(
    input_path: Path,
    output_directory: Path | None,
    pdf_path: Path,
) -> Path:
    if input_path.is_file():
        return output_directory or Path("parsed_output")

    batch_root = output_directory or Path("grobid_output") / input_path.name
    return batch_root / pdf_path.stem


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Parse scientific PDFs with GROBID and extract "
            "XRD-related figures and analysis text."
        )
    )

    argument_parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Path to a PDF file or a directory containing PDFs "
            "(searched recursively)."
        ),
    )

    argument_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory for a single PDF, or the batch root directory "
            "when parsing a folder. Defaults to parsed_output for one file "
            "and grobid_output/<folder_name> for a directory."
        ),
    )

    argument_parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="Base URL of the GROBID server.",
    )

    argument_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip PyMuPDF figure extraction from GROBID coordinates.",
    )

    argument_parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI used when rendering cropped figure images.",
    )

    argument_parser.add_argument(
        "--xrd-figures-only",
        action="store_true",
        help=(
            "Limit figure images and figure_analysis.json to figures "
            "classified as likely XRD."
        ),
    )

    argument_parser.add_argument(
        "--digitize-figures",
        action="store_true",
        help=(
            "After parsing, run PlotDigitizer on extracted figure PNGs into "
            "figures_digitized/."
        ),
    )

    argument_parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Print one summary line per PDF instead of verbose logs and JSON "
            "output."
        ),
    )

    argument_parser.add_argument(
        "--log-level",
        default=None,
        choices={
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        },
        help="Logging verbosity. Defaults to ERROR with --quiet, INFO otherwise.",
    )

    args = argument_parser.parse_args()

    log_level = args.log_level or ("ERROR" if args.quiet else "INFO")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        pdf_paths = collect_pdf_paths(args.input_path)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc

    batch_mode = args.input_path.is_dir()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        output_directory = resolve_output_directory(
            input_path=args.input_path,
            output_directory=args.output,
            pdf_path=pdf_path,
        )

        if batch_mode and not args.quiet:
            LOGGER.info(
                "Processing PDF %d/%d: %s",
                index,
                len(pdf_paths),
                pdf_path,
            )

        try:
            document = parse_pdf(
                pdf_path=pdf_path,
                output_directory=output_directory,
                grobid_url=args.grobid_url,
                extract_figures=not args.skip_figures,
                figure_dpi=args.figure_dpi,
                xrd_figures_only=args.xrd_figures_only,
            )
        except Exception as exc:
            if args.quiet:
                print(
                    format_quiet_summary_line(
                        index,
                        None,
                        None,
                        success=False,
                        title=pdf_path.stem.replace("_", " "),
                    )
                )
            else:
                LOGGER.exception("PDF parsing failed for %s: %s", pdf_path, exc)
            failures.append(
                {
                    "source_pdf": str(pdf_path),
                    "error": str(exc),
                }
            )
            continue

        extracted_xrd_figures, total_xrd_figures = xrd_figure_extraction_counts(
            document
        )

        if args.digitize_figures:
            from plotdigitizer_pipeline import digitize_parsed_paper

            try:
                digitize_parsed_paper(
                    output_directory,
                    xrd_only=args.xrd_figures_only,
                )
            except Exception as exc:
                if args.quiet:
                    LOGGER.error(
                        "Figure digitization failed for %s: %s",
                        output_directory,
                        exc,
                    )
                else:
                    LOGGER.exception(
                        "Figure digitization failed for %s: %s",
                        output_directory,
                        exc,
                    )

        if args.quiet:
            print(
                format_quiet_summary_line(
                    index,
                    extracted_xrd_figures,
                    total_xrd_figures,
                    success=True,
                    title=document.metadata.title,
                )
            )

        results.append(
            {
                "source_pdf": document.source_pdf,
                "output_directory": str(output_directory),
                "title": document.metadata.title,
                "doi": document.metadata.doi,
                "counts": document.counts,
            }
        )

    if batch_mode:
        if not args.quiet:
            summary = {
                "input_directory": str(args.input_path.resolve()),
                "processed": len(results),
                "failed": len(failures),
                "results": results,
                "failures": failures,
            }

            if args.output is not None:
                summary["output_root"] = str(args.output)
            else:
                summary["output_root"] = str(
                    Path("grobid_output") / args.input_path.name
                )

            print(json.dumps(summary, indent=2))

        if failures and not results:
            raise SystemExit(1)

        if failures:
            raise SystemExit(2)

        return

    if not results:
        raise SystemExit(1)

    if not args.quiet:
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
