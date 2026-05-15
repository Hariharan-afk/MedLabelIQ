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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seeds = load_locked_smoke_set(settings.smoke_set_path)

    gap_rows: list[dict[str, Any]] = []
    title_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    title_examples: dict[str, set[str]] = defaultdict(set)

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
            if section["retrieval_family"] is not None:
                continue

            normalized_title = section["normalized_title"] or "<NO_TITLE>"
            canonical_name = section["canonical_section_name"] or "<NO_CANONICAL>"

            row = {
                "concept_name": seed.concept_name,
                "section_uid": section["section_uid"],
                "depth": section["depth"],
                "loinc_code": section["loinc_code"],
                "code_display_name": section["code_display_name"],
                "raw_title": section["raw_title"],
                "normalized_title": normalized_title,
                "canonical_section_name": canonical_name,
                "nearest_canonical_section_name": section["nearest_canonical_section_name"],
                "heading_path": " > ".join(section["heading_path"]),
                "direct_char_count": section["direct_char_count"],
                "child_count": section["child_count"],
            }

            gap_rows.append(row)

            title_counts[normalized_title] += 1
            canonical_counts[canonical_name] += 1
            concept_counts[seed.concept_name] += 1
            title_examples[normalized_title].add(seed.concept_name)

    details_path = settings.interim_dir / "retrieval_family_gaps_detail.csv"
    summary_path = settings.interim_dir / "retrieval_family_gaps_frequency.csv"

    frequency_rows = []
    for title, count in title_counts.most_common():
        frequency_rows.append(
            {
                "normalized_title": title,
                "count": count,
                "example_concepts": " | ".join(sorted(title_examples[title])),
            }
        )

    write_csv(
        details_path,
        gap_rows,
        fieldnames=[
            "concept_name",
            "section_uid",
            "depth",
            "loinc_code",
            "code_display_name",
            "raw_title",
            "normalized_title",
            "canonical_section_name",
            "nearest_canonical_section_name",
            "heading_path",
            "direct_char_count",
            "child_count",
        ],
    )

    write_csv(
        summary_path,
        frequency_rows,
        fieldnames=[
            "normalized_title",
            "count",
            "example_concepts",
        ],
    )

    print("\nRETRIEVAL-FAMILY GAP ANALYSIS")
    print("=" * 44)
    print(f"Total retrieval-family missing sections: {len(gap_rows)}")

    print("\nMissing retrieval-family sections by concept:")
    for concept_name, count in concept_counts.most_common():
        print(f"  - {concept_name}: {count}")

    print("\nMissing sections by direct canonical name:")
    for canonical_name, count in canonical_counts.most_common():
        print(f"  - {canonical_name}: {count}")

    print("\nMost common missing normalized titles:")
    for row in frequency_rows[:40]:
        print(
            f"  - {row['normalized_title']}: {row['count']} "
            f"(examples: {row['example_concepts']})"
        )

    print("\nReports written:")
    print(f"  - {details_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()