from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path

from medlabeliq.config.settings import settings
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set
from medlabeliq.parsing.spl_parser import parse_spl_label

LOGGER = logging.getLogger(__name__)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "concept_name",
        "set_id_locked",
        "version_locked",
        "set_id_from_xml",
        "version_from_xml",
        "document_code",
        "document_code_display_name",
        "title",
        "product_count",
        "ingredient_count",
        "route_count",
        "section_count",
        "direct_mapped_section_count",
        "direct_unmapped_section_count",
        "retrieval_family_covered_section_count",
        "retrieval_family_missing_section_count",
        "unclassified_section_count",
        "max_section_depth",
        "parsed_json_path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    seeds = load_locked_smoke_set(settings.smoke_set_path)
    output_dir = settings.interim_dir / "parsed_labels"
    summary_path = settings.interim_dir / "parsed_label_summary.csv"

    summary_rows: list[dict[str, object]] = []
    mapping_method_counts: Counter[str] = Counter()

    for seed in seeds:
        assert seed.set_id is not None
        assert seed.locked_spl_version is not None

        xml_path = (
            settings.raw_spl_dir
            / seed.set_id
            / f"v{seed.locked_spl_version}"
            / "label.xml"
        )

        parsed = parse_spl_label(
            xml_path=xml_path,
            concept_name=seed.concept_name,
        )

        output_path = (
            output_dir
            / f"{seed.concept_name}_{seed.set_id}_v{seed.locked_spl_version}.json"
        )
        write_json(parsed, output_path)

        sections = parsed["sections"]
        products = parsed["products"]

        direct_mapped_sections = [
            section
            for section in sections
            if section["canonical_section_name"] is not None
        ]
        direct_unmapped_sections = [
            section
            for section in sections
            if section["canonical_section_name"] is None
        ]
        retrieval_family_covered_sections = [
            section
            for section in sections
            if section["retrieval_family"] is not None
        ]
        retrieval_family_missing_sections = [
            section
            for section in sections
            if section["retrieval_family"] is None
        ]
        unclassified_sections = [
            section
            for section in sections
            if section["is_unclassified"]
        ]

        ingredient_count = sum(len(product["ingredients"]) for product in products)
        max_depth = max((section["depth"] for section in sections), default=0)

        for section in sections:
            mapping_method_counts[section["mapping_method"]] += 1

        summary_rows.append(
            {
                "concept_name": seed.concept_name,
                "set_id_locked": seed.set_id,
                "version_locked": seed.locked_spl_version,
                "set_id_from_xml": parsed["document"]["set_id"],
                "version_from_xml": parsed["document"]["version_number"],
                "document_code": parsed["document"]["document_code"],
                "document_code_display_name": parsed["document"][
                    "document_code_display_name"
                ],
                "title": parsed["document"]["title"],
                "product_count": len(products),
                "ingredient_count": ingredient_count,
                "route_count": len(parsed["routes"]),
                "section_count": len(sections),
                "direct_mapped_section_count": len(direct_mapped_sections),
                "direct_unmapped_section_count": len(direct_unmapped_sections),
                "retrieval_family_covered_section_count": len(
                    retrieval_family_covered_sections
                ),
                "retrieval_family_missing_section_count": len(
                    retrieval_family_missing_sections
                ),
                "unclassified_section_count": len(unclassified_sections),
                "max_section_depth": max_depth,
                "parsed_json_path": str(output_path),
            }
        )

        LOGGER.info(
            "Parsed %s: %s products, %s sections -> %s",
            seed.concept_name,
            len(products),
            len(sections),
            output_path,
        )

    write_summary_csv(summary_rows, summary_path)

    print("\nSTEP 4B PARSE SUMMARY")
    print("=" * 40)
    print(f"Labels parsed: {len(summary_rows)}")
    print(f"Summary written: {summary_path}")

    print("\nCanonical mapping methods across all sections:")
    for method, count in mapping_method_counts.items():
        print(f"  - {method}: {count}")

    print("\nPer-label section counts:")
    for row in summary_rows:
        print(
            f"  - {row['concept_name']}: "
            f"{row['section_count']} sections, "
            f"{row['direct_mapped_section_count']} directly mapped, "
            f"{row['direct_unmapped_section_count']} directly unmapped, "
            f"{row['retrieval_family_covered_section_count']} retrieval-family covered, "
            f"{row['retrieval_family_missing_section_count']} retrieval-family missing, "
            f"{row['unclassified_section_count']} unclassified"
        )


if __name__ == "__main__":
    main()