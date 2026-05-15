from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set


MEDLABELIQ_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/medlabeliq/phase1",
)


def deterministic_uuid(name: str) -> str:
    return str(uuid.uuid5(MEDLABELIQ_NAMESPACE, name))


def parsed_label_path(
    concept_name: str,
    set_id: str,
    version: int,
) -> Path:
    return (
        settings.interim_dir
        / "parsed_labels"
        / f"{concept_name}_{set_id}_v{version}.json"
    )


def read_download_manifest() -> dict[tuple[str, int], dict[str, str]]:
    path = settings.download_manifest_csv_path

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    manifest: dict[tuple[str, int], dict[str, str]] = {}

    for row in rows:
        key = (row["set_id"], int(row["locked_spl_version"]))
        manifest[key] = row

    return manifest


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def insert_label_document(cur, label_id: str, concept_name: str, document: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO label_document (
            label_id,
            concept_name,
            set_id,
            document_id_root,
            document_code,
            document_code_display_name,
            document_title
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (set_id) DO UPDATE SET
            concept_name = EXCLUDED.concept_name,
            document_id_root = EXCLUDED.document_id_root,
            document_code = EXCLUDED.document_code,
            document_code_display_name = EXCLUDED.document_code_display_name,
            document_title = EXCLUDED.document_title
        """,
        (
            label_id,
            concept_name,
            document["set_id"],
            document["document_id_root"],
            document["document_code"],
            document["document_code_display_name"],
            document["title"],
        ),
    )


def insert_label_version(
    cur,
    label_version_id: str,
    label_id: str,
    version_number: int,
    document: dict[str, Any],
    source_xml_path: str,
    xml_sha256: str | None,
) -> None:
    cur.execute(
        """
        INSERT INTO label_version (
            label_version_id,
            label_id,
            version_number,
            effective_time,
            source_xml_path,
            xml_sha256,
            is_locked_version
        )
        VALUES (%s, %s, %s, %s, %s, %s, TRUE)
        ON CONFLICT (label_id, version_number) DO UPDATE SET
            effective_time = EXCLUDED.effective_time,
            source_xml_path = EXCLUDED.source_xml_path,
            xml_sha256 = EXCLUDED.xml_sha256,
            is_locked_version = TRUE
        """,
        (
            label_version_id,
            label_id,
            version_number,
            document["effective_time"],
            source_xml_path,
            xml_sha256,
        ),
    )


def insert_products(
    cur,
    label_version_id: str,
    payload: dict[str, Any],
) -> None:
    routes = payload.get("routes", [])
    active_moieties = payload.get("active_moieties", [])

    for product_index, product in enumerate(payload["products"], start=1):
        product_id = deterministic_uuid(
            f"product:{label_version_id}:{product_index}:{product.get('product_code')}"
        )

        cur.execute(
            """
            INSERT INTO label_product (
                product_id,
                label_version_id,
                product_index,
                product_name,
                product_code,
                product_code_system,
                dosage_form_code,
                dosage_form_display_name,
                route_names,
                active_moieties
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (label_version_id, product_index) DO UPDATE SET
                product_name = EXCLUDED.product_name,
                product_code = EXCLUDED.product_code,
                product_code_system = EXCLUDED.product_code_system,
                dosage_form_code = EXCLUDED.dosage_form_code,
                dosage_form_display_name = EXCLUDED.dosage_form_display_name,
                route_names = EXCLUDED.route_names,
                active_moieties = EXCLUDED.active_moieties
            """,
            (
                product_id,
                label_version_id,
                product_index,
                product.get("product_name"),
                product.get("product_code"),
                product.get("product_code_system"),
                product.get("dosage_form_code"),
                product.get("dosage_form_display_name"),
                Jsonb(routes),
                Jsonb(active_moieties),
            ),
        )

        for ingredient_index, ingredient in enumerate(product["ingredients"], start=1):
            ingredient_id = deterministic_uuid(
                f"ingredient:{product_id}:{ingredient_index}:{ingredient.get('name')}"
            )

            cur.execute(
                """
                INSERT INTO product_ingredient (
                    ingredient_id,
                    product_id,
                    ingredient_index,
                    class_code,
                    ingredient_name,
                    unii_code,
                    strength_numerator_value,
                    strength_numerator_unit,
                    strength_denominator_value,
                    strength_denominator_unit
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (product_id, ingredient_index) DO UPDATE SET
                    class_code = EXCLUDED.class_code,
                    ingredient_name = EXCLUDED.ingredient_name,
                    unii_code = EXCLUDED.unii_code,
                    strength_numerator_value = EXCLUDED.strength_numerator_value,
                    strength_numerator_unit = EXCLUDED.strength_numerator_unit,
                    strength_denominator_value = EXCLUDED.strength_denominator_value,
                    strength_denominator_unit = EXCLUDED.strength_denominator_unit
                """,
                (
                    ingredient_id,
                    product_id,
                    ingredient_index,
                    ingredient.get("class_code"),
                    ingredient.get("name"),
                    ingredient.get("unii_code"),
                    ingredient.get("strength_numerator_value"),
                    ingredient.get("strength_numerator_unit"),
                    ingredient.get("strength_denominator_value"),
                    ingredient.get("strength_denominator_unit"),
                ),
            )


def insert_sections(
    cur,
    label_version_id: str,
    payload: dict[str, Any],
) -> None:
    sections = payload["sections"]

    source_uid_to_section_id = {
        section["section_uid"]: deterministic_uuid(
            f"section:{label_version_id}:{section['section_uid']}"
        )
        for section in sections
    }

    for section in sections:
        section_id = source_uid_to_section_id[section["section_uid"]]

        parent_source_uid = section["parent_section_uid"]
        parent_section_id = (
            source_uid_to_section_id[parent_source_uid]
            if parent_source_uid is not None
            else None
        )

        cur.execute(
            """
            INSERT INTO label_section (
                section_id,
                label_version_id,
                source_section_uid,
                parent_section_id,
                parent_source_section_uid,
                order_index,
                depth,
                loinc_code,
                code_display_name,
                raw_title,
                normalized_title,
                canonical_section_name,
                mapping_method,
                nearest_canonical_section_name,
                nearest_canonical_source_section_uid,
                retrieval_family,
                retrieval_family_source_section_uid,
                is_unclassified,
                direct_text,
                direct_text_sha256,
                direct_char_count,
                child_count,
                heading_path
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (label_version_id, source_section_uid) DO UPDATE SET
                parent_section_id = EXCLUDED.parent_section_id,
                parent_source_section_uid = EXCLUDED.parent_source_section_uid,
                order_index = EXCLUDED.order_index,
                depth = EXCLUDED.depth,
                loinc_code = EXCLUDED.loinc_code,
                code_display_name = EXCLUDED.code_display_name,
                raw_title = EXCLUDED.raw_title,
                normalized_title = EXCLUDED.normalized_title,
                canonical_section_name = EXCLUDED.canonical_section_name,
                mapping_method = EXCLUDED.mapping_method,
                nearest_canonical_section_name = EXCLUDED.nearest_canonical_section_name,
                nearest_canonical_source_section_uid = EXCLUDED.nearest_canonical_source_section_uid,
                retrieval_family = EXCLUDED.retrieval_family,
                retrieval_family_source_section_uid = EXCLUDED.retrieval_family_source_section_uid,
                is_unclassified = EXCLUDED.is_unclassified,
                direct_text = EXCLUDED.direct_text,
                direct_text_sha256 = EXCLUDED.direct_text_sha256,
                direct_char_count = EXCLUDED.direct_char_count,
                child_count = EXCLUDED.child_count,
                heading_path = EXCLUDED.heading_path
            """,
            (
                section_id,
                label_version_id,
                section["section_uid"],
                parent_section_id,
                parent_source_uid,
                section["order_index"],
                section["depth"],
                section["loinc_code"],
                section["code_display_name"],
                section["raw_title"],
                section["normalized_title"],
                section["canonical_section_name"],
                section["mapping_method"],
                section["nearest_canonical_section_name"],
                section["nearest_canonical_source_section_uid"],
                section["retrieval_family"],
                section["retrieval_family_source_section_uid"],
                section["is_unclassified"],
                section["direct_text"],
                section["direct_text_sha256"],
                section["direct_char_count"],
                section["child_count"],
                Jsonb(section["heading_path"]),
            ),
        )


def main() -> None:
    seeds = load_locked_smoke_set(settings.smoke_set_path)
    download_manifest = read_download_manifest()

    loaded_count = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for seed in seeds:
                assert seed.set_id is not None
                assert seed.locked_spl_version is not None

                path = parsed_label_path(
                    concept_name=seed.concept_name,
                    set_id=seed.set_id,
                    version=seed.locked_spl_version,
                )

                payload = load_json(path)
                document = payload["document"]

                label_id = deterministic_uuid(f"label_document:{seed.set_id}")
                label_version_id = deterministic_uuid(
                    f"label_version:{seed.set_id}:v{seed.locked_spl_version}"
                )

                manifest_row = download_manifest.get(
                    (seed.set_id, seed.locked_spl_version),
                    {},
                )

                insert_label_document(
                    cur=cur,
                    label_id=label_id,
                    concept_name=seed.concept_name,
                    document=document,
                )

                insert_label_version(
                    cur=cur,
                    label_version_id=label_version_id,
                    label_id=label_id,
                    version_number=seed.locked_spl_version,
                    document=document,
                    source_xml_path=payload["source_xml_path"],
                    xml_sha256=manifest_row.get("xml_sha256"),
                )

                insert_products(
                    cur=cur,
                    label_version_id=label_version_id,
                    payload=payload,
                )

                insert_sections(
                    cur=cur,
                    label_version_id=label_version_id,
                    payload=payload,
                )

                loaded_count += 1

                print(
                    f"Loaded {seed.concept_name}: "
                    f"{len(payload['products'])} products, "
                    f"{len(payload['sections'])} sections"
                )

    print(f"\nLoaded {loaded_count} parsed labels into PostgreSQL.")


if __name__ == "__main__":
    main()