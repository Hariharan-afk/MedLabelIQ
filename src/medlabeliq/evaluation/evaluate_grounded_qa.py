from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medlabeliq.config.settings import settings
from medlabeliq.generation.answer_generator import (
    APPLICATION_SAFETY_NOTE,
    answer_query,
)


@dataclass(frozen=True)
class QAEvalCase:
    id: str
    query: str
    drug: str | None
    family: str | None
    expected_status: str
    expected_response_any: list[str]
    expected_cited_heading_any: list[str]
    minimum_citations: int


def normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.lower().split())


def contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True

    normalized_text = normalize(text)
    return any(normalize(needle) in normalized_text for needle in needles)


def load_cases(path: Path) -> list[QAEvalCase]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    cases: list[QAEvalCase] = []

    for item in payload["cases"]:
        cases.append(
            QAEvalCase(
                id=item["id"],
                query=item["query"],
                drug=item.get("drug"),
                family=item.get("family"),
                expected_status=item["expected_status"],
                expected_response_any=item.get("expected_response_any", []),
                expected_cited_heading_any=item.get(
                    "expected_cited_heading_any", []
                ),
                minimum_citations=int(item.get("minimum_citations", 0)),
            )
        )

    return cases


def cited_headings(
    citations: list[str],
    evidence_items,
) -> list[str]:
    evidence_by_id = {
        item.evidence_id: item
        for item in evidence_items
    }

    headings: list[str] = []

    for citation in citations:
        item = evidence_by_id.get(citation)
        if item is not None:
            headings.append(item.heading)

    return headings


def evaluate_case(case: QAEvalCase) -> dict[str, Any]:
    try:
        generated = answer_query(
            query=case.query,
            concept_name=case.drug,
            retrieval_family=case.family,
        )

        answer = generated.answer
        evidence_pack = generated.evidence_pack

        proposed_status = (
            generated.proposed_answer.status
            if generated.proposed_answer is not None
            else None
        )

        verification_verdict = (
            generated.verification.verdict
            if generated.verification is not None
            else None
        )

        verification_overrode_answer = generated.verification_overrode_answer

        combined_response_text = (
            f"{answer.answer}\n{answer.evidence_summary}"
        )

        status_pass = answer.status == case.expected_status

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

        headings = cited_headings(
            answer.citations,
            evidence_pack.evidence_items,
        )

        if case.expected_cited_heading_any:
            cited_heading_pass = any(
                contains_any(heading, case.expected_cited_heading_any)
                for heading in headings
            )
        else:
            cited_heading_pass = True

        safety_note_pass = (
            answer.safety_note == APPLICATION_SAFETY_NOTE
        )

        overall_pass = all(
            [
                status_pass,
                response_text_pass,
                citation_count_pass,
                citations_policy_pass,
                cited_heading_pass,
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
            "proposed_status": proposed_status,
            "verification_verdict": verification_verdict,
            "verification_overrode_answer": verification_overrode_answer,
            "status_pass": status_pass,
            "response_text_pass": response_text_pass,
            "citation_count": len(answer.citations),
            "citation_count_pass": citation_count_pass,
            "citations_policy_pass": citations_policy_pass,
            "citations": " | ".join(answer.citations),
            "cited_headings": " | ".join(headings),
            "cited_heading_pass": cited_heading_pass,
            "safety_note_pass": safety_note_pass,
            "evidence_count": len(evidence_pack.evidence_items),
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
            "proposed_status": None,
            "verification_verdict": None,
            "verification_overrode_answer": False,
            "status_pass": False,
            "response_text_pass": False,
            "citation_count": 0,
            "citation_count_pass": False,
            "citations_policy_pass": False,
            "citations": "",
            "cited_headings": "",
            "cited_heading_pass": False,
            "safety_note_pass": False,
            "evidence_count": 0,
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
        "proposed_status",
        "verification_verdict",
        "verification_overrode_answer",
        "status_pass",
        "response_text_pass",
        "citation_count",
        "citation_count_pass",
        "citations_policy_pass",
        "citations",
        "cited_headings",
        "cited_heading_pass",
        "safety_note_pass",
        "evidence_count",
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
        description="Evaluate MedLabelIQ grounded answer generation."
    )

    parser.add_argument(
        "--eval-set",
        default=str(
            settings.project_root
            / "data"
            / "evaluation"
            / "qa_generation_eval_smoke.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            settings.interim_dir
            / "grounded_qa_eval_smoke_results.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_path = Path(args.eval_set)
    output_path = Path(args.output)

    cases = load_cases(eval_path)
    rows = [evaluate_case(case) for case in cases]

    total = len(rows)

    overall_pass_count = sum(
        1 for row in rows
        if row["overall_pass"]
    )

    status_pass_count = sum(
        1 for row in rows
        if row["status_pass"]
    )

    answered_cases = [
        row for row in rows
        if row["expected_status"] == "answered"
    ]

    abstention_cases = [
        row for row in rows
        if row["expected_status"] == "insufficient_evidence"
    ]

    answered_pass_count = sum(
        1 for row in answered_cases
        if row["overall_pass"]
    )

    abstention_pass_count = sum(
        1 for row in abstention_cases
        if row["overall_pass"]
    )

    citation_policy_pass_count = sum(
        1 for row in rows
        if row["citations_policy_pass"]
    )

    cited_heading_pass_count = sum(
        1 for row in rows
        if row["cited_heading_pass"]
    )

    safety_pass_count = sum(
        1 for row in rows
        if row["safety_note_pass"]
    )

    verifier_override_count = sum(
        1 for row in rows
        if row["verification_overrode_answer"]
    )

    verifier_supported_count = sum(
        1 for row in rows
        if row["verification_verdict"] == "supported"
    )

    verifier_insufficient_count = sum(
        1 for row in rows
        if row["verification_verdict"] == "insufficient"
    )

    verifier_refuted_count = sum(
        1 for row in rows
        if row["verification_verdict"] == "refuted"
    )

    write_csv(output_path, rows)

    print("\nGROUNDED QA EVALUATION")
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

    if answered_cases:
        print(
            f"Answered-case pass: {answered_pass_count}/{len(answered_cases)} "
            f"= {answered_pass_count / len(answered_cases):.3f}"
        )

    if abstention_cases:
        print(
            f"Abstention-case pass: {abstention_pass_count}/{len(abstention_cases)} "
            f"= {abstention_pass_count / len(abstention_cases):.3f}"
        )

    print(
        f"Citation-policy pass: {citation_policy_pass_count}/{total} "
        f"= {citation_policy_pass_count / total:.3f}"
    )

    print(
        f"Cited-heading pass: {cited_heading_pass_count}/{total} "
        f"= {cited_heading_pass_count / total:.3f}"
    )

    print(
        f"Safety-note pass: {safety_pass_count}/{total} "
        f"= {safety_pass_count / total:.3f}"
    )

    print("\nVerifier diagnostics:")
    print(f"  - Verifier overrides: {verifier_override_count}")
    print(f"  - Supported verdicts: {verifier_supported_count}")
    print(f"  - Insufficient verdicts: {verifier_insufficient_count}")
    print(f"  - Refuted verdicts: {verifier_refuted_count}")

    print(f"\nResults written: {output_path}")

    print("\nPer-case results:")
    for row in rows:
        status = "PASS" if row["overall_pass"] else "FAIL"

        print(
            f"  - {status} | {row['case_id']} | "
            f"expected={row['expected_status']} | "
            f"proposed={row['proposed_status']} | "
            f"verifier={row['verification_verdict']} | "
            f"overrode={row['verification_overrode_answer']} | "
            f"actual={row['actual_status']} | "
            f"citations={row['citations']} | "
            f"evidence_count={row['evidence_count']}"
        )

    failures = [
        row for row in rows
        if not row["overall_pass"]
    ]

    if failures:
        print("\nFailure details:")
        for row in failures:
            print(
                f"  - {row['case_id']} | "
                f"expected={row['expected_status']} | "
                f"proposed={row['proposed_status']} | "
                f"verifier={row['verification_verdict']} | "
                f"overrode={row['verification_overrode_answer']} | "
                f"actual={row['actual_status']} | "
                f"status_pass={row['status_pass']} | "
                f"response_text_pass={row['response_text_pass']} | "
                f"citation_count_pass={row['citation_count_pass']} | "
                f"cited_heading_pass={row['cited_heading_pass']} | "
                f"safety_note_pass={row['safety_note_pass']} | "
                f"error={row['error']}"
            )


if __name__ == "__main__":
    main()