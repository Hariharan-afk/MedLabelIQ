from __future__ import annotations

import json
from pathlib import Path

from medlabeliq.config.settings import settings
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set


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

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    sections = payload["sections"]
    section_by_uid = {section["section_uid"]: section for section in sections}

    for section in sections:
        uid = section["section_uid"]

        canonical_name = section["canonical_section_name"]
        nearest_name = section["nearest_canonical_section_name"]
        nearest_uid = section["nearest_canonical_source_section_uid"]
        retrieval_family = section["retrieval_family"]
        retrieval_family_uid = section["retrieval_family_source_section_uid"]

        # If a section is directly mapped, nearest canonical section should be itself.
        if canonical_name is not None:
            if nearest_name != canonical_name:
                errors.append(
                    f"{seed.concept_name}: {uid} direct canonical {canonical_name} "
                    f"but nearest canonical is {nearest_name}."
                )

            if nearest_uid != uid:
                errors.append(
                    f"{seed.concept_name}: {uid} direct canonical section should "
                    f"point nearest_canonical_source_section_uid to itself."
                )

        # If nearest canonical UID exists, it must be a valid section UID.
        if nearest_uid is not None and nearest_uid not in section_by_uid:
            errors.append(
                f"{seed.concept_name}: {uid} has invalid nearest canonical UID {nearest_uid}."
            )

        # If retrieval family exists, source UID must also exist.
        if retrieval_family is not None:
            if retrieval_family_uid is None:
                errors.append(
                    f"{seed.concept_name}: {uid} has retrieval_family "
                    f"{retrieval_family} but no retrieval_family_source_section_uid."
                )
            elif retrieval_family_uid not in section_by_uid:
                errors.append(
                    f"{seed.concept_name}: {uid} has invalid retrieval family UID "
                    f"{retrieval_family_uid}."
                )

        # If retrieval family source UID exists, retrieval family must exist.
        if retrieval_family_uid is not None and retrieval_family is None:
            errors.append(
                f"{seed.concept_name}: {uid} has retrieval_family_source_section_uid "
                f"but retrieval_family is None."
            )

    summary = {
        "concept_name": seed.concept_name,
        "section_count": len(sections),
        "direct_mapped": sum(
            1 for section in sections
            if section["canonical_section_name"] is not None
        ),
        "nearest_canonical_present": sum(
            1 for section in sections
            if section["nearest_canonical_section_name"] is not None
        ),
        "retrieval_family_present": sum(
            1 for section in sections
            if section["retrieval_family"] is not None
        ),
        "retrieval_family_missing": sum(
            1 for section in sections
            if section["retrieval_family"] is None
        ),
    }

    return errors, summary


def main() -> None:
    seeds = load_locked_smoke_set(settings.smoke_set_path)

    all_errors: list[str] = []
    summaries: list[dict[str, object]] = []

    for seed in seeds:
        errors, summary = validate_one_label(seed)
        all_errors.extend(errors)
        summaries.append(summary)

    print("\nSECTION HIERARCHY VALIDATION")
    print("=" * 44)

    for summary in summaries:
        print(
            f"- {summary['concept_name']}: "
            f"{summary['section_count']} sections | "
            f"{summary['direct_mapped']} direct mapped | "
            f"{summary['nearest_canonical_present']} nearest-canonical present | "
            f"{summary['retrieval_family_present']} retrieval-family present | "
            f"{summary['retrieval_family_missing']} missing"
        )

    if all_errors:
        print("\nVALIDATION STATUS: FAIL")
        print("=" * 44)
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: PASS")
    print("=" * 44)
    print("Section hierarchy, nearest-canonical inheritance, and retrieval-family fields are consistent.")


if __name__ == "__main__":
    main()