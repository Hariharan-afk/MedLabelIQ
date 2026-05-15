from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lxml import etree

from medlabeliq.config.settings import settings
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set

HL7_NS = "urn:hl7-org:v3"
NS = {"hl7": HL7_NS}


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def first_text(root: etree._Element, xpath: str) -> str | None:
    result = root.xpath(xpath, namespaces=NS)
    if not result:
        return None

    first = result[0]
    if isinstance(first, etree._Element):
        return clean_text(" ".join(first.itertext()))

    return clean_text(str(first))


def all_values(root: etree._Element, xpath: str) -> list[str]:
    result = root.xpath(xpath, namespaces=NS)
    values: list[str] = []

    for item in result:
        if isinstance(item, etree._Element):
            value = clean_text(" ".join(item.itertext()))
        else:
            value = clean_text(str(item))

        if value:
            values.append(value)

    return values


def parse_xml(path: Path) -> etree._Element:
    parser = etree.XMLParser(remove_blank_text=True, recover=False)
    tree = etree.parse(str(path), parser)
    return tree.getroot()


def extract_top_level_metadata(root: etree._Element) -> dict[str, Any]:
    return {
        "root_tag": etree.QName(root).localname,
        "document_id_root": first_text(root, "./hl7:id/@root"),
        "set_id": first_text(root, "./hl7:setId/@root"),
        "version_number": first_text(root, "./hl7:versionNumber/@value"),
        "effective_time": first_text(root, "./hl7:effectiveTime/@value"),
        "document_code": first_text(root, "./hl7:code/@code"),
        "document_code_display_name": first_text(root, "./hl7:code/@displayName"),
        "title": first_text(root, "./hl7:title"),
    }


def extract_product_summary(root: etree._Element) -> dict[str, Any]:
    product_names = all_values(
        root,
        ".//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name",
    )

    ingredient_names = all_values(
        root,
        ".//hl7:ingredient/hl7:ingredientSubstance/hl7:name",
    )

    active_moiety_names = all_values(
        root,
        ".//hl7:activeMoiety/hl7:activeMoiety/hl7:name",
    )

    dosage_forms = all_values(
        root,
        ".//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:formCode/@displayName",
    )

    routes = all_values(
        root,
        ".//hl7:routeCode/@displayName",
    )

    ndc_product_codes = all_values(
        root,
        ".//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:code/@code",
    )

    return {
        "product_names": product_names,
        "ingredient_names": ingredient_names,
        "active_moiety_names": active_moiety_names,
        "dosage_forms": dosage_forms,
        "routes": routes,
        "ndc_product_codes": ndc_product_codes,
    }


def extract_sections(root: etree._Element) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    for idx, section in enumerate(root.xpath(".//hl7:section", namespaces=NS), start=1):
        title = first_text(section, "./hl7:title")
        code = first_text(section, "./hl7:code/@code")
        display_name = first_text(section, "./hl7:code/@displayName")

        parent_section = section.getparent()
        depth = 0
        while parent_section is not None:
            if etree.QName(parent_section).localname == "section":
                depth += 1
            parent_section = parent_section.getparent()

        text_node_count = len(section.xpath("./hl7:text", namespaces=NS))
        subsection_count = len(section.xpath("./hl7:component/hl7:section", namespaces=NS))

        raw_text = clean_text(" ".join(section.xpath("./hl7:text//text()", namespaces=NS)))
        char_count = len(raw_text) if raw_text else 0

        sections.append(
            {
                "section_index": idx,
                "depth": depth,
                "code": code,
                "display_name": display_name,
                "title": title,
                "text_node_count": text_node_count,
                "direct_subsection_count": subsection_count,
                "char_count_direct_text": char_count,
            }
        )

    return sections


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seeds = load_locked_smoke_set(settings.smoke_set_path)

    document_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []

    section_title_counts: Counter[str] = Counter()
    section_code_counts: Counter[str] = Counter()
    sections_by_concept: dict[str, int] = defaultdict(int)

    for seed in seeds:
        assert seed.set_id is not None
        assert seed.locked_spl_version is not None

        xml_path = (
            settings.raw_spl_dir
            / seed.set_id
            / f"v{seed.locked_spl_version}"
            / "label.xml"
        )

        root = parse_xml(xml_path)

        metadata = extract_top_level_metadata(root)
        product_summary = extract_product_summary(root)
        sections = extract_sections(root)

        document_rows.append(
            {
                "concept_name": seed.concept_name,
                "xml_path": str(xml_path),
                **metadata,
                "product_names": " | ".join(product_summary["product_names"]),
                "ingredient_names": " | ".join(product_summary["ingredient_names"]),
                "active_moiety_names": " | ".join(product_summary["active_moiety_names"]),
                "dosage_forms": " | ".join(product_summary["dosage_forms"]),
                "routes": " | ".join(product_summary["routes"]),
                "ndc_product_codes": " | ".join(product_summary["ndc_product_codes"]),
                "section_count": len(sections),
            }
        )

        for section in sections:
            section_rows.append(
                {
                    "concept_name": seed.concept_name,
                    "set_id": seed.set_id,
                    "locked_spl_version": seed.locked_spl_version,
                    **section,
                }
            )

            sections_by_concept[seed.concept_name] += 1

            if section["title"]:
                section_title_counts[section["title"]] += 1

            if section["code"]:
                section_code_counts[section["code"]] += 1

    document_report_path = settings.interim_dir / "spl_document_inspection.csv"
    section_report_path = settings.interim_dir / "spl_section_inspection.csv"

    write_rows(
        document_report_path,
        document_rows,
        fieldnames=[
            "concept_name",
            "xml_path",
            "root_tag",
            "document_id_root",
            "set_id",
            "version_number",
            "effective_time",
            "document_code",
            "document_code_display_name",
            "title",
            "product_names",
            "ingredient_names",
            "active_moiety_names",
            "dosage_forms",
            "routes",
            "ndc_product_codes",
            "section_count",
        ],
    )

    write_rows(
        section_report_path,
        section_rows,
        fieldnames=[
            "concept_name",
            "set_id",
            "locked_spl_version",
            "section_index",
            "depth",
            "code",
            "display_name",
            "title",
            "text_node_count",
            "direct_subsection_count",
            "char_count_direct_text",
        ],
    )

    print("\nSPL XML INSPECTION SUMMARY")
    print("=" * 40)
    print(f"Labels inspected: {len(document_rows)}")
    print(f"Total sections found: {len(section_rows)}")

    print("\nSections per concept:")
    for concept_name in sorted(sections_by_concept):
        print(f"  - {concept_name}: {sections_by_concept[concept_name]}")

    print("\nMost common section titles:")
    for title, count in section_title_counts.most_common(20):
        print(f"  - {title}: {count}")

    print("\nMost common section codes:")
    for code, count in section_code_counts.most_common(20):
        print(f"  - {code}: {count}")

    print("\nReports written:")
    print(f"  - {document_report_path}")
    print(f"  - {section_report_path}")


if __name__ == "__main__":
    main()