from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medlabeliq.config.settings import settings
from medlabeliq.generation.answer_generator import APPLICATION_SAFETY_NOTE
from medlabeliq.orchestration.qa_workflow import (
    answer_query_with_drug_resolution,
)
from medlabeliq.rxnorm.identity_answer import (
    RXNORM_IDENTITY_SAFETY_NOTE,
)


@dataclass(frozen=True)
class MultiSourceEvalCase:
    id: str
    query: str
    drug: str | None
    family: str | None

    expected_status: str
    expected_source: str | None
    expected_source_route_status: str | None
    expected_family_plan_status: str | None
    expected_retrieval_family: str | None

    expected_response_any: list[str]

    minimum_citations: int
    citation_prefix: str | None

    minimum_label_evidence: int
    maximum_label_evidence: int | None

    minimum_identity_evidence: int
    maximum_identity_evidence: int | None


def normalize(value: str | None) -> str:
    if value is None:
        return ""

    return " ".join(value.lower().split())


def contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True

    normalized_text = normalize(text)

    return any(
        normalize(needle) in normalized_text
        for needle in needles
    )


def load_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    return int(value)


def load_cases(path: Path) -> list[MultiSourceEvalCase]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    cases: list[MultiSourceEvalCase] = []

    for item in payload["cases"]:
        cases.append(
            MultiSourceEvalCase(
                id=item["id"],
                query=item["query"],
                drug=item.get("drug"),
                family=item.get("family"),
                expected_status=item["expected_status"],
                expected_source=item.get("expected_source"),
                expected_source_route_status=item.get(
                    "expected_source_route_status"
                ),
                expected_family_plan_status=item.get(
                    "expected_family_plan_status"
                ),
                expected_retrieval_family=item.get(
                    "expected_retrieval_family"
                ),
                expected_response_any=item.get(
                    "expected_response_any",
                    [],
                ),
                minimum_citations=int(
                    item.get("minimum_citations", 0)
                ),
                citation_prefix=item.get("citation_prefix"),
                minimum_label_evidence=int(
                    item.get("minimum_label_evidence", 0)
                ),
                maximum_label_evidence=load_optional_int(
                    item.get("maximum_label_evidence")
                ),
                minimum_identity_evidence=int(
                    item.get("minimum_identity_evidence", 0)
                ),
                maximum_identity_evidence=load_optional_int(
                    item.get("maximum_identity_evidence")
                ),
            )
        )

    return cases


def expected_safety_note_for_source(source: str | None) -> str:
    if source == "rxnorm_identity":
        return RXNORM_IDENTITY_SAFETY_NOTE

    return APPLICATION_SAFETY_NOTE


def evidence_ids_for_citation_validation(
    *,
    actual_source: str | None,
    label_evidence_items,
    identity_evidence_items,
) -> set[str]:
    if actual_source == "rxnorm_identity":
        return {
            item.evidence_id
            for item in identity_evidence_items
        }

    return {
        item.evidence_id
        for item in label_evidence_items
    }


def evaluate_case(case: MultiSourceEvalCase) -> dict[str, Any]:
    try:
        workflow_result = answer_query_with_drug_resolution(
            query=case.query,
            requested_drug=case.drug,
            retrieval_family=case.family,
        )

        generated = workflow_result.generated
        answer = generated.answer

        source_plan = workflow_result.source_plan
        family_plan = workflow_result.family_plan

        label_evidence_items = generated.evidence_pack.evidence_items
        identity_evidence_items = workflow_result.identity_evidence or []

        actual_source = (
            source_plan.selected_source
            if source_plan is not None
            else None
        )

        actual_source_route_status = (
            source_plan.status
            if source_plan is not None
            else None
        )

        actual_family_plan_status = (
            family_plan.status
            if family_plan is not None
            else None
        )

        actual_retrieval_family = workflow_result.retrieval_family

        combined_response_text = (
            f"{answer.answer}\n{answer.evidence_summary}"
        )

        status_pass = answer.status == case.expected_status

        source_pass = (
            True
            if case.expected_source is None
            else actual_source == case.expected_source
        )

        source_route_status_pass = (
            True
            if case.expected_source_route_status is None
            else (
                actual_source_route_status
                == case.expected_source_route_status
            )
        )

        family_plan_status_pass = (
            True
            if case.expected_family_plan_status is None
            else (
                actual_family_plan_status
                == case.expected_family_plan_status
            )
        )

        retrieval_family_pass = (
            True
            if case.expected_retrieval_family is None
            else (
                actual_retrieval_family
                == case.expected_retrieval_family
            )
        )

        response_text_pass = contains_any(
            combined_response_text,
            case.expected_response_any,
        )

        if case.expected_status == "answered":
            citation_count_pass = (
                len(answer.citations) >= case.minimum_citations
            )
            citations_policy_pass = len(answer.citations) > 0
        else:
            citation_count_pass = len(answer.citations) == 0
            citations_policy_pass = len(answer.citations) == 0

        valid_evidence_ids = evidence_ids_for_citation_validation(
            actual_source=actual_source,
            label_evidence_items=label_evidence_items,
            identity_evidence_items=identity_evidence_items,
        )

        citation_reference_pass = all(
            citation in valid_evidence_ids
            for citation in answer.citations
        )

        citation_prefix_pass = (
            True
            if case.citation_prefix is None
            else all(
                citation.startswith(case.citation_prefix)
                for citation in answer.citations
            )
        )

        label_evidence_count = len(label_evidence_items)
        identity_evidence_count = len(identity_evidence_items)

        minimum_label_evidence_pass = (
            label_evidence_count >= case.minimum_label_evidence
        )

        maximum_label_evidence_pass = (
            True
            if case.maximum_label_evidence is None
            else label_evidence_count <= case.maximum_label_evidence
        )

        minimum_identity_evidence_pass = (
            identity_evidence_count >= case.minimum_identity_evidence
        )

        maximum_identity_evidence_pass = (
            True
            if case.maximum_identity_evidence is None
            else identity_evidence_count <= case.maximum_identity_evidence
        )

        safety_note_pass = (
            answer.safety_note
            == expected_safety_note_for_source(actual_source)
        )

        overall_pass = all(
            [
                status_pass,
                source_pass,
                source_route_status_pass,
                family_plan_status_pass,
                retrieval_family_pass,
                response_text_pass,
                citation_count_pass,
                citations_policy_pass,
                citation_reference_pass,
                citation_prefix_pass,
                minimum_label_evidence_pass,
                maximum_label_evidence_pass,
                minimum_identity_evidence_pass,
                maximum_identity_evidence_pass,
                safety_note_pass,
            ]
        )

        return {
            "case_id": case.id,
            "query": case.query,
            "drug": case.drug,
            "family": case.family,
            "expected_status": case.expected_status,
            "actual_status": answer.status,
            "expected_source": case.expected_source,
            "actual_source": actual_source,
            "expected_source_route_status": case.expected_source_route_status,
            "actual_source_route_status": actual_source_route_status,
            "expected_family_plan_status": case.expected_family_plan_status,
            "actual_family_plan_status": actual_family_plan_status,
            "expected_retrieval_family": case.expected_retrieval_family,
            "actual_retrieval_family": actual_retrieval_family,
            "status_pass": status_pass,
            "source_pass": source_pass,
            "source_route_status_pass": source_route_status_pass,
            "family_plan_status_pass": family_plan_status_pass,
            "retrieval_family_pass": retrieval_family_pass,
            "response_text_pass": response_text_pass,
            "citation_count": len(answer.citations),
            "minimum_citations": case.minimum_citations,
            "citation_count_pass": citation_count_pass,
            "citations_policy_pass": citations_policy_pass,
            "citations": " | ".join(answer.citations),
            "citation_reference_pass": citation_reference_pass,
            "citation_prefix": case.citation_prefix,
            "citation_prefix_pass": citation_prefix_pass,
            "label_evidence_count": label_evidence_count,
            "minimum_label_evidence": case.minimum_label_evidence,
            "maximum_label_evidence": case.maximum_label_evidence,
            "minimum_label_evidence_pass": minimum_label_evidence_pass,
            "maximum_label_evidence_pass": maximum_label_evidence_pass,
            "identity_evidence_count": identity_evidence_count,
            "minimum_identity_evidence": case.minimum_identity_evidence,
            "maximum_identity_evidence": case.maximum_identity_evidence,
            "minimum_identity_evidence_pass": minimum_identity_evidence_pass,
            "maximum_identity_evidence_pass": maximum_identity_evidence_pass,
            "safety_note_pass": safety_note_pass,
            "answer": answer.answer,
            "evidence_summary": answer.evidence_summary,
            "overall_pass": overall_pass,
            "error": "",
        }

    except Exception as exc:
        return {
            "case_id": case.id,
            "query": case.query,
            "drug": case.drug,
            "family": case.family,
            "expected_status": case.expected_status,
            "actual_status": "ERROR",
            "expected_source": case.expected_source,
            "actual_source": None,
            "expected_source_route_status": case.expected_source_route_status,
            "actual_source_route_status": None,
            "expected_family_plan_status": case.expected_family_plan_status,
            "actual_family_plan_status": None,
            "expected_retrieval_family": case.expected_retrieval_family,
            "actual_retrieval_family": None,
            "status_pass": False,
            "source_pass": False,
            "source_route_status_pass": False,
            "family_plan_status_pass": False,
            "retrieval_family_pass": False,
            "response_text_pass": False,
            "citation_count": 0,
            "minimum_citations": case.minimum_citations,
            "citation_count_pass": False,
            "citations_policy_pass": False,
            "citations": "",
            "citation_reference_pass": False,
            "citation_prefix": case.citation_prefix,
            "citation_prefix_pass": False,
            "label_evidence_count": 0,
            "minimum_label_evidence": case.minimum_label_evidence,
            "maximum_label_evidence": case.maximum_label_evidence,
            "minimum_label_evidence_pass": False,
            "maximum_label_evidence_pass": False,
            "identity_evidence_count": 0,
            "minimum_identity_evidence": case.minimum_identity_evidence,
            "maximum_identity_evidence": case.maximum_identity_evidence,
            "minimum_identity_evidence_pass": False,
            "maximum_identity_evidence_pass": False,
            "safety_note_pass": False,
            "answer": "",
            "evidence_summary": "",
            "overall_pass": False,
            "error": repr(exc),
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "query",
        "drug",
        "family",
        "expected_status",
        "actual_status",
        "expected_source",
        "actual_source",
        "expected_source_route_status",
        "actual_source_route_status",
        "expected_family_plan_status",
        "actual_family_plan_status",
        "expected_retrieval_family",
        "actual_retrieval_family",
        "status_pass",
        "source_pass",
        "source_route_status_pass",
        "family_plan_status_pass",
        "retrieval_family_pass",
        "response_text_pass",
        "citation_count",
        "minimum_citations",
        "citation_count_pass",
        "citations_policy_pass",
        "citations",
        "citation_reference_pass",
        "citation_prefix",
        "citation_prefix_pass",
        "label_evidence_count",
        "minimum_label_evidence",
        "maximum_label_evidence",
        "minimum_label_evidence_pass",
        "maximum_label_evidence_pass",
        "identity_evidence_count",
        "minimum_identity_evidence",
        "maximum_identity_evidence",
        "minimum_identity_evidence_pass",
        "maximum_identity_evidence_pass",
        "safety_note_pass",
        "answer",
        "evidence_summary",
        "overall_pass",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MedLabelIQ multi-source orchestration across "
            "DailyMed label QA and RxNorm identity QA."
        )
    )

    parser.add_argument(
        "--eval-set",
        default=str(
            settings.project_root
            / "data"
            / "evaluation"
            / "multisource_orchestration_eval_smoke.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            settings.interim_dir
            / "multisource_orchestration_eval_smoke_results.csv"
        ),
    )

    return parser.parse_args()


def pass_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(
        1
        for row in rows
        if row[key]
    )


def subset_with_expected_value(
    rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row[key] is not None
    ]


def main() -> None:
    args = parse_args()

    eval_path = Path(args.eval_set)
    output_path = Path(args.output)

    cases = load_cases(eval_path)
    rows = [evaluate_case(case) for case in cases]

    total = len(rows)

    overall_pass_count = pass_count(rows, "overall_pass")
    status_pass_count = pass_count(rows, "status_pass")
    source_pass_count = pass_count(rows, "source_pass")
    citation_policy_pass_count = pass_count(
        rows,
        "citations_policy_pass",
    )
    citation_reference_pass_count = pass_count(
        rows,
        "citation_reference_pass",
    )
    safety_note_pass_count = pass_count(rows, "safety_note_pass")

    source_route_rows = subset_with_expected_value(
        rows,
        "expected_source_route_status",
    )
    source_route_pass_count = pass_count(
        source_route_rows,
        "source_route_status_pass",
    )

    family_plan_rows = subset_with_expected_value(
        rows,
        "expected_family_plan_status",
    )
    family_plan_pass_count = pass_count(
        family_plan_rows,
        "family_plan_status_pass",
    )

    retrieval_family_rows = subset_with_expected_value(
        rows,
        "expected_retrieval_family",
    )
    retrieval_family_pass_count = pass_count(
        retrieval_family_rows,
        "retrieval_family_pass",
    )

    identity_route_count = sum(
        1
        for row in rows
        if row["actual_source"] == "rxnorm_identity"
    )

    label_route_count = sum(
        1
        for row in rows
        if row["actual_source"] == "dailymed_label"
    )

    write_csv(output_path, rows)

    print("\nMULTI-SOURCE ORCHESTRATION EVALUATION")
    print("=" * 80)
    print(f"Eval set: {eval_path}")
    print(f"Cases: {total}")
    print(
        f"Overall pass: {overall_pass_count}/{total} "
        f"= {overall_pass_count / total:.3f}"
    )
    print(
        f"Status accuracy: {status_pass_count}/{total} "
        f"= {status_pass_count / total:.3f}"
    )
    print(
        f"Source-route accuracy: {source_pass_count}/{total} "
        f"= {source_pass_count / total:.3f}"
    )

    if source_route_rows:
        print(
            f"Source-route-status accuracy: "
            f"{source_route_pass_count}/{len(source_route_rows)} "
            f"= {source_route_pass_count / len(source_route_rows):.3f}"
        )

    if family_plan_rows:
        print(
            f"Family-plan-status accuracy: "
            f"{family_plan_pass_count}/{len(family_plan_rows)} "
            f"= {family_plan_pass_count / len(family_plan_rows):.3f}"
        )

    if retrieval_family_rows:
        print(
            f"Retrieval-family accuracy: "
            f"{retrieval_family_pass_count}/{len(retrieval_family_rows)} "
            f"= {retrieval_family_pass_count / len(retrieval_family_rows):.3f}"
        )

    print(
        f"Citation-policy pass: {citation_policy_pass_count}/{total} "
        f"= {citation_policy_pass_count / total:.3f}"
    )
    print(
        f"Citation-reference pass: {citation_reference_pass_count}/{total} "
        f"= {citation_reference_pass_count / total:.3f}"
    )
    print(
        f"Safety-note pass: {safety_note_pass_count}/{total} "
        f"= {safety_note_pass_count / total:.3f}"
    )
    print(f"Actual RxNorm identity routes: {identity_route_count}")
    print(f"Actual DailyMed label routes: {label_route_count}")
    print(f"\nResults written: {output_path}")

    print("\nPer-case results:")
    for row in rows:
        status = "PASS" if row["overall_pass"] else "FAIL"

        print(
            f"  - {status} | {row['case_id']} | "
            f"source={row['actual_source']} | "
            f"source_status={row['actual_source_route_status']} | "
            f"expected_status={row['expected_status']} | "
            f"actual_status={row['actual_status']} | "
            f"family={row['actual_retrieval_family']} | "
            f"citations={row['citations']} | "
            f"label_evidence={row['label_evidence_count']} | "
            f"identity_evidence={row['identity_evidence_count']}"
        )

    failures = [
        row
        for row in rows
        if not row["overall_pass"]
    ]

    if failures:
        print("\nFailure details:")
        for row in failures:
            print(
                f"  - {row['case_id']} | "
                f"status_pass={row['status_pass']} | "
                f"source_pass={row['source_pass']} | "
                f"source_route_status_pass="
                f"{row['source_route_status_pass']} | "
                f"family_plan_status_pass="
                f"{row['family_plan_status_pass']} | "
                f"retrieval_family_pass="
                f"{row['retrieval_family_pass']} | "
                f"response_text_pass={row['response_text_pass']} | "
                f"citation_count_pass={row['citation_count_pass']} | "
                f"citation_reference_pass="
                f"{row['citation_reference_pass']} | "
                f"citation_prefix_pass={row['citation_prefix_pass']} | "
                f"minimum_label_evidence_pass="
                f"{row['minimum_label_evidence_pass']} | "
                f"maximum_label_evidence_pass="
                f"{row['maximum_label_evidence_pass']} | "
                f"minimum_identity_evidence_pass="
                f"{row['minimum_identity_evidence_pass']} | "
                f"maximum_identity_evidence_pass="
                f"{row['maximum_identity_evidence_pass']} | "
                f"safety_note_pass={row['safety_note_pass']} | "
                f"error={row['error']}"
            )


if __name__ == "__main__":
    main()