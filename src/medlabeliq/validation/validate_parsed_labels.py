from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medlabeliq.config.settings import settings
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def validate_one_label(seed) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []

    assert seed.set_id is not None
    assert seed.locked_spl_version is not None

    path = parsed_label_path(
        concept_name=seed.concept_name,
        set_id=seed.set_id,
        version=seed.locked_spl_version,
    )

    if not path.exists():
        return [f"{seed.concept_name}: missing parsed JSON at {path}"], {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    document = payload["document"]
    products = payload["products"]
    sections = payload["sections"]

    if document["set_id"] != seed.set_id:
        errors.append(
            f"{seed.concept_name}: XML SET ID {document['set_id']} "
            f"does not match locked SET ID {seed.set_id}."
        )

    if str(document["version_number"]) != str(seed.locked_spl_version):
        errors.append(
            f"{seed.concept_name}: XML version {document['version_number']} "
            f"does not match locked version {seed.locked_spl_version}."
        )

    if not products:
        errors.append(f"{seed.concept_name}: no products extracted.")

    if not sections:
        errors.append(f"{seed.concept_name}: no sections extracted.")

    section_uids = [section["section_uid"] for section in sections]
    if len(section_uids) != len(set(section_uids)):
        errors.append(f"{seed.concept_name}: duplicate section_uid values found.")

    section_uid_set = set(section_uids)

    order_indexes = [section["order_index"] for section in sections]
    expected_order = list(range(1, len(sections) + 1))
    if order_indexes != expected_order:
        errors.append(
            f"{seed.concept_name}: section order_index values are not sequential."
        )

    for section in sections:
        uid = section["section_uid"]
        parent_uid = section["parent_section_uid"]
        depth = section["depth"]
        heading_path = section["heading_path"]
        direct_text = section["direct_text"]
        stored_hash = section["direct_text_sha256"]

        if parent_uid is not None and parent_uid not in section_uid_set:
            errors.append(
                f"{seed.concept_name}: section {uid} references missing parent {parent_uid}."
            )

        if parent_uid is None and depth != 0:
            errors.append(
                f"{seed.concept_name}: root section {uid} has nonzero depth {depth}."
            )

        if parent_uid is not None and depth == 0:
            errors.append(
                f"{seed.concept_name}: child section {uid} has zero depth."
            )

        if len(heading_path) != depth + 1:
            errors.append(
                f"{seed.concept_name}: section {uid} heading_path length "
                f"{len(heading_path)} does not equal depth + 1 ({depth + 1})."
            )

        if stored_hash != sha256_text(direct_text):
            errors.append(
                f"{seed.concept_name}: section {uid} direct-text checksum mismatch."
            )

        if section["mapping_method"] == "unmapped" and section["canonical_section_name"] is not None:
            errors.append(
                f"{seed.concept_name}: section {uid} has canonical name despite unmapped method."
            )

        if section["mapping_method"] != "unmapped" and section["canonical_section_name"] is None:
            errors.append(
                f"{seed.concept_name}: section {uid} has mapping method "
                f"{section['mapping_method']} but no canonical name."
            )

    summary = {
        "concept_name": seed.concept_name,
        "document_code_display_name": document["document_code_display_name"],
        "product_count": len(products),
        "section_count": len(sections),
        "mapped_section_count": sum(
            1 for section in sections
            if section["canonical_section_name"] is not None
        ),
        "unmapped_section_count": sum(
            1 for section in sections
            if section["canonical_section_name"] is None
        ),
        "max_depth": max((section["depth"] for section in sections), default=0),
    }

    return errors, summary


def main() -> None:
    seeds = load_locked_smoke_set(settings.smoke_set_path)

    all_errors: list[str] = []
    summaries: list[dict[str, object]] = []

    for seed in seeds:
        errors, summary = validate_one_label(seed)
        all_errors.extend(errors)

        if summary:
            summaries.append(summary)

    print("\nSTEP 4C PARSED-LABEL VALIDATION")
    print("=" * 44)

    for summary in summaries:
        print(
            f"- {summary['concept_name']}: "
            f"{summary['document_code_display_name']} | "
            f"{summary['product_count']} products | "
            f"{summary['section_count']} sections | "
            f"{summary['mapped_section_count']} mapped | "
            f"{summary['unmapped_section_count']} unmapped | "
            f"max depth {summary['max_depth']}"
        )

    if all_errors:
        print("\nVALIDATION STATUS: FAIL")
        print("=" * 44)
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: PASS")
    print("=" * 44)
    print("All parsed labels are structurally consistent.")


if __name__ == "__main__":
    main()