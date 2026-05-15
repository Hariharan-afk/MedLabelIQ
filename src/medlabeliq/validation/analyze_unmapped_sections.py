from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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


def classify_unmapped_reason(section: dict[str, Any]) -> str:
    loinc_code = section["loinc_code"]
    normalized_title = section["normalized_title"]

    if loinc_code == "42229-5":
        return "unclassified_with_title" if normalized_title else "unclassified_no_title"

    if loinc_code and normalized_title:
        return "coded_but_not_in_mapper"

    if loinc_code and not normalized_title:
        return "coded_no_title_not_in_mapper"

    if not loinc_code and normalized_title:
        return "title_only_not_in_mapper"

    return "no_code_no_title"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seeds = load_locked_smoke_set(settings.smoke_set_path)

    unmapped_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()

    title_examples: dict[str, set[str]] = defaultdict(set)
    title_depths: dict[str, set[int]] = defaultdict(set)
    title_codes: dict[str, set[str]] = defaultdict(set)

    for seed in seeds:
        assert seed.set_id is not None
        assert seed.locked_spl_version is not None

        path = parsed_label_path(
            concept_name=seed.concept_name,
            set_id=seed.set_id,
            version=seed.locked_spl_version,
        )

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for section in payload["sections"]:
            if section["canonical_section_name"] is not None:
                continue

            reason = classify_unmapped_reason(section)

            row = {
                "concept_name": seed.concept_name,
                "section_uid": section["section_uid"],
                "depth": section["depth"],
                "loinc_code": section["loinc_code"],
                "code_display_name": section["code_display_name"],
                "raw_title": section["raw_title"],
                "normalized_title": section["normalized_title"],
                "reason": reason,
                "parent_section_uid": section["parent_section_uid"],
                "heading_path": " > ".join(section["heading_path"]),
                "direct_char_count": section["direct_char_count"],
                "child_count": section["child_count"],
            }

            unmapped_rows.append(row)
            reason_counts[reason] += 1

            normalized_title = section["normalized_title"] or "<NO_TITLE>"
            loinc_code = section["loinc_code"] or "<NO_CODE>"

            title_counts[normalized_title] += 1
            code_counts[loinc_code] += 1

            title_examples[normalized_title].add(seed.concept_name)
            title_depths[normalized_title].add(section["depth"])
            title_codes[normalized_title].add(loinc_code)

    frequency_rows: list[dict[str, Any]] = []
    for normalized_title, count in title_counts.most_common():
        frequency_rows.append(
            {
                "normalized_title": normalized_title,
                "count": count,
                "loinc_codes": " | ".join(sorted(title_codes[normalized_title])),
                "depths": " | ".join(str(depth) for depth in sorted(title_depths[normalized_title])),
                "example_concepts": " | ".join(sorted(title_examples[normalized_title])),
            }
        )

    details_path = settings.interim_dir / "unmapped_sections_detail.csv"
    frequency_path = settings.interim_dir / "unmapped_section_frequency.csv"

    write_csv(
        details_path,
        unmapped_rows,
        fieldnames=[
            "concept_name",
            "section_uid",
            "depth",
            "loinc_code",
            "code_display_name",
            "raw_title",
            "normalized_title",
            "reason",
            "parent_section_uid",
            "heading_path",
            "direct_char_count",
            "child_count",
        ],
    )

    write_csv(
        frequency_path,
        frequency_rows,
        fieldnames=[
            "normalized_title",
            "count",
            "loinc_codes",
            "depths",
            "example_concepts",
        ],
    )

    print("\nUNMAPPED SECTION ANALYSIS")
    print("=" * 40)
    print(f"Total unmapped sections: {len(unmapped_rows)}")

    print("\nUnmapped sections by reason:")
    for reason, count in reason_counts.most_common():
        print(f"  - {reason}: {count}")

    print("\nMost common unmapped normalized titles:")
    for row in frequency_rows[:30]:
        print(
            f"  - {row['normalized_title']}: {row['count']} "
            f"(codes: {row['loinc_codes']}; depths: {row['depths']}; "
            f"examples: {row['example_concepts']})"
        )

    print("\nReports written:")
    print(f"  - {details_path}")
    print(f"  - {frequency_path}")


if __name__ == "__main__":
    main()