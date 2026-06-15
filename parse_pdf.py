from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lxml import etree


TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"

NS = {
    "tei": TEI_NAMESPACE,
    "xml": XML_NAMESPACE,
}

XML_ID = f"{{{XML_NAMESPACE}}}id"


# Matches short strings that strongly resemble standalone captions.
# This is only a fallback because XML structure is the primary filter.
CAPTION_START_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        fig(?:ure)?\.?
        |
        table
        |
        scheme
        |
        chart
        |
        plate
    )
    \s*
    (?:s?\d+|[ivxlcdm]+)
    \s*
    [.:)\-]?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


RESULTS_HEADING_PATTERN = re.compile(
    r"""
    \b(
        results?
        |
        discussion
        |
        findings
        |
        observations
    )\b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


NON_RESULTS_HEADING_PATTERN = re.compile(
    r"""
    ^\s*
    (
        supplementary
        |
        supporting\s+information
        |
        references?
        |
        bibliography
        |
        acknowledg(?:e)?ments?
    )
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def normalize_whitespace(text: str | None) -> str:
    """Collapse repeated whitespace and trim the result."""
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def clean_text(element: etree._Element | None) -> str:
    """
    Return all text contained in an XML element, including nested inline tags.
    """
    if element is None:
        return ""

    return normalize_whitespace(" ".join(element.itertext()))


def first_text(
    root: etree._Element,
    xpath: str,
) -> str | None:
    """Return cleaned text from the first XPath match."""
    matches = root.xpath(xpath, namespaces=NS)

    if not matches:
        return None

    value = matches[0]

    if isinstance(value, etree._Element):
        text = clean_text(value)
    else:
        text = normalize_whitespace(str(value))

    return text or None


def all_text(
    root: etree._Element,
    xpath: str,
) -> list[str]:
    """Return cleaned nonempty text for all XPath matches."""
    values: list[str] = []

    for match in root.xpath(xpath, namespaces=NS):
        if isinstance(match, etree._Element):
            text = clean_text(match)
        else:
            text = normalize_whitespace(str(match))

        if text:
            values.append(text)

    return values


def unique_preserving_order(values: list[str]) -> list[str]:
    """Remove duplicate strings while preserving their original order."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized_key = normalize_whitespace(value).casefold()

        if not normalized_key or normalized_key in seen:
            continue

        seen.add(normalized_key)
        result.append(value)

    return result


def is_probable_caption(text: str) -> bool:
    """
    Identify strings that strongly resemble standalone figure/table captions.

    Long prose such as "Figure 3 demonstrates that..." is retained because
    standalone captions are usually relatively short and label-like.
    """
    normalized = normalize_whitespace(text)

    if not normalized:
        return False

    if len(normalized) > 500:
        return False

    return bool(CAPTION_START_PATTERN.match(normalized))


def is_results_heading(heading: str | None) -> bool:
    """Return True when a heading appears to identify a Results section."""
    if not heading:
        return False

    normalized = normalize_whitespace(heading)

    if NON_RESULTS_HEADING_PATTERN.search(normalized):
        return False

    return bool(RESULTS_HEADING_PATTERN.search(normalized))


def element_id(element: etree._Element) -> str | None:
    """Return the xml:id attribute of a TEI element."""
    return element.get(XML_ID)


def combine_caption_parts(parts: list[str | None]) -> str | None:
    """
    Combine label, head, and description while avoiding obvious duplication.
    """
    cleaned = [
        normalize_whitespace(part)
        for part in parts
        if part and normalize_whitespace(part)
    ]

    cleaned = unique_preserving_order(cleaned)

    if not cleaned:
        return None

    result: list[str] = []

    for part in cleaned:
        part_key = part.casefold()

        # Skip a part if it is already fully contained in a longer part.
        if any(part_key in existing.casefold() for existing in result):
            continue

        # Replace a short existing part if the new part contains all of it.
        result = [
            existing
            for existing in result
            if existing.casefold() not in part_key
        ]

        result.append(part)

    return normalize_whitespace(" ".join(result)) or None


def extract_paragraphs_from_div(
    div: etree._Element,
) -> list[str]:
    """
    Extract prose paragraphs directly owned by one section.

    Using './tei:p' rather than './/tei:p' prevents paragraphs from nested
    subsections from being duplicated in their parent sections.
    """
    paragraphs: list[str] = []

    for paragraph in div.xpath("./tei:p", namespaces=NS):
        # Defensive check: exclude anything under figures, tables, notes,
        # bibliographies, or formulas even if a malformed TEI document places
        # it unexpectedly.
        excluded_ancestor = paragraph.xpath(
            """
            ancestor::tei:figure
            or ancestor::tei:table
            or ancestor::tei:listBibl
            or ancestor::tei:note
            or ancestor::tei:formula
            """,
            namespaces=NS,
        )

        if excluded_ancestor:
            continue

        text = clean_text(paragraph)

        if not text:
            continue

        # Structural filtering should normally be enough. This catches cases
        # where GROBID incorrectly emits a caption as an ordinary <p>.
        if is_probable_caption(text):
            continue

        paragraphs.append(text)

    return unique_preserving_order(paragraphs)


def extract_sections(root: etree._Element) -> list[dict[str, Any]]:
    """
    Extract the document's nested section structure.

    Each section includes:
      - section_id
      - heading
      - heading_path
      - depth
      - paragraphs
      - is_results_section
    """
    body_nodes = root.xpath(
        ".//tei:text/tei:body",
        namespaces=NS,
    )

    if not body_nodes:
        return []

    body = body_nodes[0]
    sections: list[dict[str, Any]] = []

    def visit_div(
        div: etree._Element,
        parent_path: list[str],
        parent_is_results: bool,
        depth: int,
    ) -> None:
        heading = first_text(div, "./tei:head[1]")
        paragraphs = extract_paragraphs_from_div(div)

        heading_path = list(parent_path)
        if heading:
            heading_path.append(heading)

        starts_results = is_results_heading(heading)
        belongs_to_results = parent_is_results or starts_results

        # Do not emit empty anonymous wrapper divs, but still recurse through
        # them because they may contain meaningful nested sections.
        if heading or paragraphs:
            sections.append(
                {
                    "section_id": element_id(div),
                    "heading": heading,
                    "heading_path": heading_path,
                    "depth": depth,
                    "paragraphs": paragraphs,
                    "is_results_section": belongs_to_results,
                }
            )

        for child_div in div.xpath("./tei:div", namespaces=NS):
            visit_div(
                div=child_div,
                parent_path=heading_path,
                parent_is_results=belongs_to_results,
                depth=depth + 1,
            )

    for top_level_div in body.xpath("./tei:div", namespaces=NS):
        visit_div(
            div=top_level_div,
            parent_path=[],
            parent_is_results=False,
            depth=0,
        )

    return sections


def extract_graphic_references(
    figure: etree._Element,
) -> list[str]:
    """Extract image paths or URLs referenced by <graphic> elements."""
    references: list[str] = []

    for graphic in figure.xpath(".//tei:graphic", namespaces=NS):
        url = (
            graphic.get("url")
            or graphic.get("target")
            or graphic.get("href")
            or graphic.get(f"{{{XML_NAMESPACE}}}base")
        )

        if url:
            references.append(normalize_whitespace(url))

    return unique_preserving_order(references)


def extract_figures(root: etree._Element) -> list[dict[str, Any]]:
    """
    Extract true figures.

    GROBID commonly represents tables as <figure type="table">, so those are
    excluded here and handled by extract_tables().
    """
    figures: list[dict[str, Any]] = []

    figure_nodes = root.xpath(
        """
        .//tei:text//tei:figure[
            not(
                translate(
                    normalize-space(@type),
                    'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    'abcdefghijklmnopqrstuvwxyz'
                ) = 'table'
            )
        ]
        """,
        namespaces=NS,
    )

    for index, figure in enumerate(figure_nodes, start=1):
        label = first_text(figure, "./tei:label[1]")
        head = first_text(figure, "./tei:head[1]")
        description = first_text(figure, "./tei:figDesc[1]")

        caption = combine_caption_parts(
            [label, head, description]
        )

        figures.append(
            {
                "figure_id": element_id(figure) or f"figure_{index}",
                "label": label,
                "head": head,
                "description": description,
                "caption": caption,
                "graphic_references": extract_graphic_references(figure),
            }
        )

    return figures


def extract_tables(root: etree._Element) -> list[dict[str, Any]]:
    """
    Extract tables represented either as <figure type="table"> or standalone
    <table> elements.
    """
    tables: list[dict[str, Any]] = []
    seen_elements: set[int] = set()

    table_figures = root.xpath(
        """
        .//tei:text//tei:figure[
            translate(
                normalize-space(@type),
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                'abcdefghijklmnopqrstuvwxyz'
            ) = 'table'
        ]
        """,
        namespaces=NS,
    )

    for index, table_figure in enumerate(table_figures, start=1):
        seen_elements.add(id(table_figure))

        label = first_text(table_figure, "./tei:label[1]")
        head = first_text(table_figure, "./tei:head[1]")
        description = first_text(table_figure, "./tei:figDesc[1]")

        tables.append(
            {
                "table_id": (
                    element_id(table_figure)
                    or f"table_figure_{index}"
                ),
                "label": label,
                "head": head,
                "description": description,
                "caption": combine_caption_parts(
                    [label, head, description]
                ),
                "text": first_text(
                    table_figure,
                    ".//tei:table[1]",
                ),
            }
        )

    standalone_tables = root.xpath(
        """
        .//tei:text//tei:table[
            not(ancestor::tei:figure)
        ]
        """,
        namespaces=NS,
    )

    for index, table in enumerate(standalone_tables, start=1):
        if id(table) in seen_elements:
            continue

        tables.append(
            {
                "table_id": element_id(table) or f"table_{index}",
                "label": first_text(table, "./tei:label[1]"),
                "head": first_text(table, "./tei:head[1]"),
                "description": None,
                "caption": combine_caption_parts(
                    [
                        first_text(table, "./tei:label[1]"),
                        first_text(table, "./tei:head[1]"),
                    ]
                ),
                "text": clean_text(table) or None,
            }
        )

    return tables


def extract_authors(root: etree._Element) -> list[dict[str, str | None]]:
    """Extract author names from the TEI header."""
    authors: list[dict[str, str | None]] = []

    author_nodes = root.xpath(
        ".//tei:teiHeader//tei:sourceDesc//tei:author",
        namespaces=NS,
    )

    if not author_nodes:
        author_nodes = root.xpath(
            ".//tei:teiHeader//tei:titleStmt/tei:author",
            namespaces=NS,
        )

    for author in author_nodes:
        first_name = first_text(
            author,
            ".//tei:forename[@type='first'][1]",
        )
        middle_name = first_text(
            author,
            ".//tei:forename[@type='middle'][1]",
        )
        surname = first_text(
            author,
            ".//tei:surname[1]",
        )

        full_name = normalize_whitespace(
            " ".join(
                part
                for part in [first_name, middle_name, surname]
                if part
            )
        )

        if not full_name:
            full_name = clean_text(author)

        if full_name:
            authors.append(
                {
                    "full_name": full_name,
                    "first_name": first_name,
                    "middle_name": middle_name,
                    "surname": surname,
                }
            )

    # Deduplicate authors by normalized full name.
    deduplicated: list[dict[str, str | None]] = []
    seen_names: set[str] = set()

    for author in authors:
        key = (author["full_name"] or "").casefold()

        if key and key not in seen_names:
            seen_names.add(key)
            deduplicated.append(author)

    return deduplicated


def parse_xml(xml_path: str | Path) -> etree._Element:
    """
    Parse TEI XML safely.

    A strict parse is attempted first. If GROBID produced slightly malformed
    XML, a recovery parse is attempted as a fallback.
    """
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"TEI file does not exist: {xml_path}")

    if not xml_path.is_file():
        raise ValueError(f"TEI path is not a file: {xml_path}")

    strict_parser = etree.XMLParser(
        recover=False,
        remove_blank_text=True,
        resolve_entities=False,
        no_network=True,
        huge_tree=True,
    )

    try:
        tree = etree.parse(str(xml_path), strict_parser)
    except etree.XMLSyntaxError:
        recovery_parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            no_network=True,
            huge_tree=True,
        )
        tree = etree.parse(str(xml_path), recovery_parser)

    root = tree.getroot()

    if root is None:
        raise ValueError(f"No XML root element found in {xml_path}")

    return root


def parse_tei_file(xml_path: str | Path) -> dict[str, Any]:
    """Parse one GROBID TEI XML document into a structured dictionary."""
    xml_path = Path(xml_path)
    root = parse_xml(xml_path)

    title = (
        first_text(
            root,
            """
            .//tei:teiHeader//tei:titleStmt/
            tei:title[@type='main'][1]
            """,
        )
        or first_text(
            root,
            ".//tei:teiHeader//tei:titleStmt/tei:title[1]",
        )
    )

    doi = first_text(
        root,
        """
        .//tei:teiHeader//tei:idno[
            translate(
                @type,
                'abcdefghijklmnopqrstuvwxyz',
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            ) = 'DOI'
        ][1]
        """,
    )

    abstract = first_text(
        root,
        ".//tei:profileDesc//tei:abstract[1]",
    )

    publication_date = (
        first_text(
            root,
            ".//tei:teiHeader//tei:publicationStmt/tei:date/@when",
        )
        or first_text(
            root,
            ".//tei:teiHeader//tei:publicationStmt/tei:date[1]",
        )
        or first_text(
            root,
            ".//tei:sourceDesc//tei:date/@when",
        )
    )

    sections = extract_sections(root)

    results_sections = [
        section
        for section in sections
        if section["is_results_section"]
    ]

    # Flatten Results prose for downstream NLP while preserving structured
    # results_sections above.
    results_paragraphs = unique_preserving_order(
        [
            paragraph
            for section in results_sections
            for paragraph in section["paragraphs"]
        ]
    )

    figures = extract_figures(root)
    tables = extract_tables(root)

    return {
        "source_file": xml_path.name,
        "source_path": str(xml_path),
        "title": title,
        "doi": doi,
        "publication_date": publication_date,
        "authors": extract_authors(root),
        "abstract": abstract,
        "sections": sections,
        "results_sections": results_sections,
        "results_paragraphs": results_paragraphs,
        "figures": figures,
        "tables": tables,
        "counts": {
            "sections": len(sections),
            "results_sections": len(results_sections),
            "results_paragraphs": len(results_paragraphs),
            "figures": len(figures),
            "tables": len(tables),
        },
    }


def save_json(
    record: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save a parsed record as UTF-8 JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    input_file = Path(
        "grobid_output/carbonStacking.tei.xml"
    )
    output_file = Path(
        "parsed_outputs/example_parsed.json"
    )

    try:
        record = parse_tei_file(input_file)
        save_json(record, output_file)

        print(f"Saved parsed data to {output_file}")
        print(
            f"Extracted {record['counts']['sections']} sections, "
            f"{record['counts']['results_paragraphs']} Results paragraphs, "
            f"{record['counts']['figures']} figures, and "
            f"{record['counts']['tables']} tables."
        )

    except (
        FileNotFoundError,
        ValueError,
        etree.XMLSyntaxError,
        OSError,
    ) as error:
        print(f"Failed to parse {input_file}: {error}")
        raise
