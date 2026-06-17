from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
from lxml import etree

from parse_figures import extract_document_figures


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
    "diffraction pattern",
    "diffractogram",
    "diffractograms",
    "2θ",
    "2 theta",
    "two theta",
    "rietveld",
    "rietveld refinement",
    "bragg",
    "bragg peak",
    "diffraction peak",
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
        fig(?:ure)?s?\.?
        |
        supplementary\s+fig(?:ure)?s?\.?
    )
    \s*
    (?:[\(\[\s])*
    (?P<number>[A-Za-z]?\d+[A-Za-z]?)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

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


def normalize_figure_label(label: str | None) -> str | None:
    if not label:
        return None

    normalized = normalize_whitespace(label).lower()
    normalized = normalized.replace("figure", "fig")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")

    match = re.search(r"(\d+)", normalized)

    if match:
        return match.group(1)

    return normalized or None


def figure_label_from_caption(caption: str) -> str | None:
    match = re.search(
        r"\bfig(?:ure)?s?\.?\s*(?P<number>[A-Za-z]?\d+[A-Za-z]?)",
        caption,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return normalize_figure_label(match.group("number"))


def is_valid_figure_label(
    normalized_label: str | None,
    caption: str,
) -> bool:
    if not normalized_label or not re.fullmatch(
        r"\d+[a-z]?",
        normalized_label,
    ):
        return False

    caption_label = figure_label_from_caption(caption)

    if caption_label and caption_label != normalized_label:
        return False

    return True


def resolve_one_based_figure_id(
    grobid_figure_id: str | None,
    normalized_label: str | None,
    caption: str,
) -> str:
    if is_valid_figure_label(normalized_label, caption):
        return f"fig_{normalized_label}"

    caption_label = figure_label_from_caption(caption)

    if caption_label:
        return f"fig_{caption_label}"

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

    caption_parts = xpath_all_text(
        figure_element,
        "./tei:figDesc | ./tei:head | ./tei:p",
    )

    caption = normalize_whitespace(
        " ".join(dict.fromkeys(caption_parts))
    )

    return label, normalize_figure_label(label), caption


def reindex_tei_figure_ids(tei_xml: bytes) -> bytes:
    """
    Rewrite GROBID's 0-based figure xml:id values to 1-based paper labels.
    """
    xml_parser = etree.XMLParser(
        recover=True,
        remove_blank_text=True,
        huge_tree=True,
        resolve_entities=False,
        no_network=True,
    )
    root = etree.fromstring(tei_xml, parser=xml_parser)

    figure_entries: list[tuple[etree._Element, str, str | None, str]] = []

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
        figure_entries.append(
            (figure_element, old_id, normalized_label, caption)
        )

    id_map: dict[str, str] = {}
    used_ids: set[str] = set()

    def assign_id(old_id: str, candidate: str) -> str:
        base_id = candidate
        counter = 2
        new_id = candidate

        while new_id in used_ids:
            new_id = f"{base_id}_{counter}"
            counter += 1

        id_map[old_id] = new_id
        used_ids.add(new_id)
        return new_id

    for figure_element, old_id, normalized_label, caption in figure_entries:
        if not is_valid_figure_label(normalized_label, caption):
            continue

        new_id = assign_id(old_id, f"fig_{normalized_label}")
        figure_element.set(XML_ID, new_id)

    for figure_element, old_id, normalized_label, caption in figure_entries:
        if old_id in id_map:
            continue

        caption_label = figure_label_from_caption(caption)

        if caption_label:
            new_id = assign_id(old_id, f"fig_{caption_label}")
            figure_element.set(XML_ID, new_id)

    for figure_element, old_id, normalized_label, caption in figure_entries:
        if old_id in id_map:
            continue

        match = GROBID_FIGURE_ID_PATTERN.match(old_id)

        if match:
            new_id = assign_id(old_id, f"fig_{int(match.group(1)) + 1}")
            figure_element.set(XML_ID, new_id)
            continue

        new_id = assign_id(old_id, old_id)
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
        "diffraction pattern": 3.0,
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

            figures.append(
                Figure(
                    figure_id=figure_id,
                    label=label,
                    normalized_label=normalize_figure_label(label),
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


def build_figure_analysis_dataset(
    document: ParsedDocument,
    *,
    xrd_only: bool = False,
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    for figure in document.figures:
        if xrd_only and not figure.is_caption_xrd:
            continue

        figure_label = figure.normalized_label

        if not figure_label or not figure_label.isdigit():
            continue

        if figure_label in seen_labels:
            continue

        seen_labels.add(figure_label)

        caption = normalize_whitespace(figure.caption or "")
        analysis = normalize_whitespace(
            collect_figure_analysis_text(document, figure)
        )

        if caption and analysis:
            text = f"{caption} {analysis}"
        else:
            text = caption or analysis

        if not text:
            continue

        image_paths = figure.image_paths or []
        figure_path = image_paths[0] if image_paths else ""

        dataset.append(
            {
                "figure": int(figure_label),
                "figure_path": figure_path,
                "caption": caption,
                "text": text,
            }
        )

    if not xrd_only:
        referenced_labels: set[str] = set()

        for paragraph in document.paragraphs:
            referenced_labels.update(paragraph.figure_references)

        for figure_label in sorted(
            referenced_labels - seen_labels,
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
                    "figure_path": "",
                    "caption": "",
                    "text": analysis,
                }
            )

    dataset.sort(key=lambda entry: entry["figure"])

    return dataset


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
        extracted_count = extract_document_figures(
            pdf_path=pdf_path,
            figures=document.figures,
            output_dir=figures_directory,
            dpi=figure_dpi,
            xrd_only=xrd_figures_only,
        )
        LOGGER.info(
            "Extracted %d/%d figures to %s",
            extracted_count,
            len(document.figures),
            figures_directory,
        )

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

    figure_analysis = build_figure_analysis_dataset(
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
