from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medlabeliq.config.settings import settings
from medlabeliq.retrieval.lexical_search import SearchResult, search_chunks


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    drug: str
    search_family: str | None
    expected_families: list[str]
    expected_heading_any: list[str]
    expected_text_any: list[str]


def normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.lower().split())


def contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True

    normalized_text = normalize(text)
    return any(normalize(needle) in normalized_text for needle in needles)


def heading_text(result: SearchResult) -> str:
    return " > ".join(str(item) for item in result.heading_path)


def is_relevant(result: SearchResult, case: EvalCase) -> bool:
    if normalize(result.concept_name) != normalize(case.drug):
        return False

    if case.expected_families and result.retrieval_family not in case.expected_families:
        return False

    heading_ok = contains_any(heading_text(result), case.expected_heading_any)
    text_ok = contains_any(result.chunk_text, case.expected_text_any)

    # We accept heading match OR text match because some SPL leaf sections are
    # title-heavy, while others contain the evidence mainly in body text.
    return heading_ok or text_ok


def load_cases(path: Path) -> tuple[int, list[EvalCase]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    top_k = int(payload.get("top_k", 5))

    cases = [
        EvalCase(
            id=item["id"],
            query=item["query"],
            drug=item["drug"],
            search_family=item.get("search_family"),
            expected_families=item.get("expected_families", []),
            expected_heading_any=item.get("expected_heading_any", []),
            expected_text_any=item.get("expected_text_any", []),
        )
        for item in payload["cases"]
    ]

    return top_k, cases


def evaluate_case(case: EvalCase, top_k: int) -> dict[str, Any]:
    results = search_chunks(
        query=case.query,
        concept_name=case.drug,
        retrieval_family=case.search_family,
        limit=top_k,
    )

    relevant_rank: int | None = None
    top_result: SearchResult | None = results[0] if results else None

    for idx, result in enumerate(results, start=1):
        if is_relevant(result, case):
            relevant_rank = idx
            break

    return {
        "case_id": case.id,
        "query": case.query,
        "drug": case.drug,
        "search_family": case.search_family,
        "result_count": len(results),
        "relevant_rank": relevant_rank,
        "hit_at_1": relevant_rank == 1,
        "hit_at_k": relevant_rank is not None,
        "reciprocal_rank": (1.0 / relevant_rank) if relevant_rank else 0.0,
        "top_drug": top_result.concept_name if top_result else None,
        "top_family": top_result.retrieval_family if top_result else None,
        "top_heading": heading_text(top_result) if top_result else None,
        "top_rank_score": top_result.rank if top_result else None,
        "top_preview": top_result.chunk_text[:300].replace("\n", " ") if top_result else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "query",
        "drug",
        "search_family",
        "result_count",
        "relevant_rank",
        "hit_at_1",
        "hit_at_k",
        "reciprocal_rank",
        "top_drug",
        "top_family",
        "top_heading",
        "top_rank_score",
        "top_preview",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PostgreSQL lexical retrieval over section-aware chunks."
    )
    parser.add_argument(
        "--eval-set",
        default=str(settings.project_root / "data" / "evaluation" / "retrieval_eval_smoke.yaml"),
        help="Path to retrieval evaluation YAML file.",
    )
    parser.add_argument(
        "--output",
        default=str(settings.interim_dir / "lexical_retrieval_eval_results.csv"),
        help="Path to write per-case CSV results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_path = Path(args.eval_set)
    output_path = Path(args.output)

    top_k, cases = load_cases(eval_path)

    rows = [evaluate_case(case, top_k=top_k) for case in cases]

    total = len(rows)
    hit_at_1 = sum(1 for row in rows if row["hit_at_1"])
    hit_at_k = sum(1 for row in rows if row["hit_at_k"])
    mrr = sum(float(row["reciprocal_rank"]) for row in rows) / total if total else 0.0

    write_csv(output_path, rows)

    print("\nLEXICAL RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Eval set: {eval_path}")
    print(f"Cases: {total}")
    print(f"Top K: {top_k}")
    print(f"Hit@1: {hit_at_1}/{total} = {hit_at_1 / total:.3f}")
    print(f"Hit@{top_k}: {hit_at_k}/{total} = {hit_at_k / total:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(f"Results written: {output_path}")

    print("\nPer-case results:")
    for row in rows:
        status = "PASS" if row["hit_at_k"] else "FAIL"
        print(
            f"  - {status} | {row['case_id']} | "
            f"rank={row['relevant_rank']} | "
            f"top_family={row['top_family']} | "
            f"top_heading={row['top_heading']}"
        )

    failures = [row for row in rows if not row["hit_at_k"]]
    if failures:
        print("\nFailure cases to inspect:")
        for row in failures:
            print(
                f"  - {row['case_id']}: query='{row['query']}', "
                f"top_heading='{row['top_heading']}'"
            )


if __name__ == "__main__":
    main()