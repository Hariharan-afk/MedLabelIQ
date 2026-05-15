from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from lxml import etree

from medlabeliq.parsing.section_mapper import (
    RETRIEVAL_FAMILY_SECTIONS,
    map_canonical_section,
    normalize_section_title,
)

HL7_NS = "urn:hl7-org:v3"
NS = {"hl7": HL7_NS}


@dataclass(frozen=True)
class IngredientRecord:
    class_code: str | None
    name: str | None
    unii_code: str | None
    strength_numerator_value: str | None
    strength_numerator_unit: str | None
    strength_denominator_value: str | None
    strength_denominator_unit: str | None


@dataclass(frozen=True)
class ProductRecord:
    product_name: str | None
    product_code: str | None
    product_code_system: str | None
    dosage_form_code: str | None
    dosage_form_display_name: str | None
    ingredients: list[IngredientRecord]


@dataclass(frozen=True)
class SectionRecord:
    section_uid: str
    parent_section_uid: str | None
    order_index: int
    depth: int
    loinc_code: str | None
    code_display_name: str | None
    raw_title: str | None
    normalized_title: str | None

    # Direct identity of this section, if directly recognized by LOINC or title.
    canonical_section_name: str | None
    mapping_method: str

    # Nearest directly mapped section at this node or above it.
    nearest_canonical_section_name: str | None
    nearest_canonical_source_section_uid: str | None

    # Broader clinical bucket useful for downstream retrieval/filtering.
    retrieval_family: str | None
    retrieval_family_source_section_uid: str | None

    is_unclassified: bool
    direct_text: str | None
    direct_text_sha256: str | None
    direct_char_count: int
    child_count: int
    heading_path: list[str]


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def xpath_first_text(element: etree._Element, xpath: str) -> str | None:
    result = element.xpath(xpath, namespaces=NS)
    if not result:
        return None

    first = result[0]
    if isinstance(first, etree._Element):
        return clean_text(" ".join(first.itertext()))

    return clean_text(str(first))


def xpath_all_text(element: etree._Element, xpath: str) -> list[str]:
    result = element.xpath(xpath, namespaces=NS)
    values: list[str] = []

    for item in result:
        if isinstance(item, etree._Element):
            value = clean_text(" ".join(item.itertext()))
        else:
            value = clean_text(str(item))

        if value:
            values.append(value)

    return values


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_xml(xml_path: Path) -> etree._Element:
    parser = etree.XMLParser(remove_blank_text=True, recover=False)
    tree = etree.parse(str(xml_path), parser)
    return tree.getroot()


def extract_document_metadata(root: etree._Element) -> dict[str, Any]:
    return {
        "document_id_root": xpath_first_text(root, "./hl7:id/@root"),
        "set_id": xpath_first_text(root, "./hl7:setId/@root"),
        "version_number": xpath_first_text(root, "./hl7:versionNumber/@value"),
        "effective_time": xpath_first_text(root, "./hl7:effectiveTime/@value"),
        "document_code": xpath_first_text(root, "./hl7:code/@code"),
        "document_code_display_name": xpath_first_text(
            root, "./hl7:code/@displayName"
        ),
        "title": xpath_first_text(root, "./hl7:title"),
    }


def extract_products(root: etree._Element) -> list[ProductRecord]:
    products: list[ProductRecord] = []

    product_nodes = root.xpath(
        ".//hl7:manufacturedProduct/hl7:manufacturedProduct",
        namespaces=NS,
    )

    for product in product_nodes:
        ingredient_records: list[IngredientRecord] = []

        ingredient_nodes = product.xpath("./hl7:ingredient", namespaces=NS)
        for ingredient in ingredient_nodes:
            ingredient_records.append(
                IngredientRecord(
                    class_code=xpath_first_text(ingredient, "./@classCode"),
                    name=xpath_first_text(
                        ingredient,
                        "./hl7:ingredientSubstance/hl7:name",
                    ),
                    unii_code=xpath_first_text(
                        ingredient,
                        "./hl7:ingredientSubstance/hl7:code/@code",
                    ),
                    strength_numerator_value=xpath_first_text(
                        ingredient,
                        "./hl7:quantity/hl7:numerator/@value",
                    ),
                    strength_numerator_unit=xpath_first_text(
                        ingredient,
                        "./hl7:quantity/hl7:numerator/@unit",
                    ),
                    strength_denominator_value=xpath_first_text(
                        ingredient,
                        "./hl7:quantity/hl7:denominator/@value",
                    ),
                    strength_denominator_unit=xpath_first_text(
                        ingredient,
                        "./hl7:quantity/hl7:denominator/@unit",
                    ),
                )
            )

        products.append(
            ProductRecord(
                product_name=xpath_first_text(product, "./hl7:name"),
                product_code=xpath_first_text(product, "./hl7:code/@code"),
                product_code_system=xpath_first_text(
                    product,
                    "./hl7:code/@codeSystem",
                ),
                dosage_form_code=xpath_first_text(product, "./hl7:formCode/@code"),
                dosage_form_display_name=xpath_first_text(
                    product,
                    "./hl7:formCode/@displayName",
                ),
                ingredients=ingredient_records,
            )
        )

    return products


def extract_labeler_names(root: etree._Element) -> list[str]:
    return xpath_all_text(
        root,
        ".//hl7:representedOrganization/hl7:name",
    )


def extract_routes(root: etree._Element) -> list[str]:
    return xpath_all_text(root, ".//hl7:routeCode/@displayName")


def extract_active_moieties(root: etree._Element) -> list[str]:
    return xpath_all_text(
        root,
        ".//hl7:activeMoiety/hl7:activeMoiety/hl7:name",
    )


def direct_section_text(section: etree._Element) -> str | None:
    """
    Extract only the text directly owned by this section.

    We intentionally use only ./text, not descendant section text, so parent sections
    do not duplicate all content from nested subsections.
    """
    text_nodes = section.xpath("./hl7:text", namespaces=NS)
    if not text_nodes:
        return None

    combined = " ".join(" ".join(node.itertext()) for node in text_nodes)
    return clean_text(combined)


def section_uid(section: etree._Element, fallback_index: int) -> str:
    root_id = xpath_first_text(section, "./hl7:id/@root")
    extension = xpath_first_text(section, "./hl7:id/@extension")

    if root_id and extension:
        return f"{root_id}:{extension}"
    if root_id:
        return root_id

    return f"synthetic_section_{fallback_index}"


def extract_sections(root: etree._Element) -> list[SectionRecord]:
    records: list[SectionRecord] = []
    order_counter = 0

    def walk(
        section: etree._Element,
        parent_uid: str | None,
        depth: int,
        parent_heading_path: list[str],
        parent_nearest_canonical_name: str | None,
        parent_nearest_canonical_uid: str | None,
        parent_retrieval_family: str | None,
        parent_retrieval_family_uid: str | None,
    ) -> None:
        nonlocal order_counter
        order_counter += 1

        uid = section_uid(section, order_counter)
        loinc_code = xpath_first_text(section, "./hl7:code/@code")
        code_display_name = xpath_first_text(section, "./hl7:code/@displayName")
        raw_title = xpath_first_text(section, "./hl7:title")
        normalized_title = normalize_section_title(raw_title)

        canonical_name, mapping_method = map_canonical_section(
            loinc_code=loinc_code,
            raw_title=raw_title,
        )

        # Nearest mapped section: self if directly mapped, otherwise inherit.
        if canonical_name is not None:
            nearest_canonical_name = canonical_name
            nearest_canonical_uid = uid
        else:
            nearest_canonical_name = parent_nearest_canonical_name
            nearest_canonical_uid = parent_nearest_canonical_uid

        # Retrieval family: only major sections should define the broad family.
        if canonical_name in RETRIEVAL_FAMILY_SECTIONS:
            retrieval_family = canonical_name
            retrieval_family_uid = uid
        else:
            retrieval_family = parent_retrieval_family
            retrieval_family_uid = parent_retrieval_family_uid

        text = direct_section_text(section)

        current_heading_path = [
            *parent_heading_path,
            raw_title or code_display_name or uid,
        ]

        child_sections = section.xpath(
            "./hl7:component/hl7:section",
            namespaces=NS,
        )

        records.append(
            SectionRecord(
                section_uid=uid,
                parent_section_uid=parent_uid,
                order_index=order_counter,
                depth=depth,
                loinc_code=loinc_code,
                code_display_name=code_display_name,
                raw_title=raw_title,
                normalized_title=normalized_title,
                canonical_section_name=canonical_name,
                mapping_method=mapping_method,
                nearest_canonical_section_name=nearest_canonical_name,
                nearest_canonical_source_section_uid=nearest_canonical_uid,
                retrieval_family=retrieval_family,
                retrieval_family_source_section_uid=retrieval_family_uid,
                is_unclassified=loinc_code == "42229-5",
                direct_text=text,
                direct_text_sha256=sha256_text(text),
                direct_char_count=len(text) if text else 0,
                child_count=len(child_sections),
                heading_path=current_heading_path,
            )
        )

        for child in child_sections:
            walk(
                section=child,
                parent_uid=uid,
                depth=depth + 1,
                parent_heading_path=current_heading_path,
                parent_nearest_canonical_name=nearest_canonical_name,
                parent_nearest_canonical_uid=nearest_canonical_uid,
                parent_retrieval_family=retrieval_family,
                parent_retrieval_family_uid=retrieval_family_uid,
            )

    top_level_sections = root.xpath(
        "./hl7:component/hl7:structuredBody/hl7:component/hl7:section",
        namespaces=NS,
    )

    for top_level_section in top_level_sections:
        walk(
            section=top_level_section,
            parent_uid=None,
            depth=0,
            parent_heading_path=[],
            parent_nearest_canonical_name=None,
            parent_nearest_canonical_uid=None,
            parent_retrieval_family=None,
            parent_retrieval_family_uid=None,
        )

    return records


def parse_spl_label(
    xml_path: Path,
    concept_name: str,
) -> dict[str, Any]:
    root = parse_xml(xml_path)

    document_metadata = extract_document_metadata(root)
    products = extract_products(root)
    sections = extract_sections(root)

    return {
        "concept_name": concept_name,
        "source_xml_path": str(xml_path),
        "document": document_metadata,
        "labeler_names": extract_labeler_names(root),
        "routes": extract_routes(root),
        "active_moieties": extract_active_moieties(root),
        "products": [
            {
                **asdict(product),
                "ingredients": [asdict(ingredient) for ingredient in product.ingredients],
            }
            for product in products
        ],
        "sections": [asdict(section) for section in sections],
    }