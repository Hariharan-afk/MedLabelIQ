from __future__ import annotations

from dataclasses import dataclass

from medlabeliq.generation.answer_generator import (
    GeneratedAnswer,
    answer_query,
    build_deterministic_insufficient_answer,
)
from medlabeliq.generation.evidence_pack import (
    EvidencePack,
    build_evidence_pack,
)
from medlabeliq.orchestration.drug_filter_resolution import (
    DrugFilterResolution,
    resolve_optional_drug_filter,
)
from medlabeliq.orchestration.drug_mention_detection import (
    DrugMentionDetection,
    build_not_attempted_detection,
    detect_drug_mention_from_query,
)


@dataclass(frozen=True)
class QAWorkflowResult:
    generated: GeneratedAnswer
    drug_resolution: DrugFilterResolution
    drug_mention_detection: DrugMentionDetection
    retrieval_drug: str | None


@dataclass(frozen=True)
class RetrievalDebugWorkflowResult:
    evidence_pack: EvidencePack
    drug_resolution: DrugFilterResolution
    drug_mention_detection: DrugMentionDetection
    retrieval_drug: str | None


def build_empty_evidence_pack(
    *,
    query: str,
    retrieval_family: str | None,
) -> EvidencePack:
    return EvidencePack(
        query=query,
        concept_name=None,
        retrieval_family=retrieval_family,
        evidence_items=[],
    )


def answer_query_with_drug_resolution(
    query: str,
    *,
    requested_drug: str | None = None,
    retrieval_family: str | None = None,
    top_k: int | None = None,
) -> QAWorkflowResult:
    """
    QA workflow entry point with:
    - explicit drug-filter normalization,
    - automatic query-level drug mention detection when no explicit drug filter exists.

    Policy:
    - Explicit filter unresolved -> fail closed.
    - No explicit filter and auto-detection succeeds -> use detected drug as retrieval filter.
    - No explicit filter and auto-detection fails/ambiguous -> fall back to broad retrieval.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

    if requested_drug is not None and requested_drug.strip():
        drug_mention_detection = build_not_attempted_detection(query)

        if not drug_resolution.can_retrieve:
            empty_pack = build_empty_evidence_pack(
                query=query,
                retrieval_family=retrieval_family,
            )

            generated = build_deterministic_insufficient_answer(
                empty_pack,
            )

            return QAWorkflowResult(
                generated=generated,
                drug_resolution=drug_resolution,
                drug_mention_detection=drug_mention_detection,
                retrieval_drug=None,
            )

        retrieval_drug = drug_resolution.retrieval_drug

    else:
        drug_mention_detection = detect_drug_mention_from_query(query)

        retrieval_drug = (
            drug_mention_detection.retrieval_drug
            if drug_mention_detection.can_filter
            else None
        )

    generated = answer_query(
        query=query,
        concept_name=retrieval_drug,
        retrieval_family=retrieval_family,
        top_k=top_k,
    )

    return QAWorkflowResult(
        generated=generated,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_mention_detection,
        retrieval_drug=retrieval_drug,
    )


def build_debug_evidence_pack_with_drug_resolution(
    query: str,
    *,
    requested_drug: str | None = None,
    retrieval_family: str | None = None,
    top_k: int | None = None,
) -> RetrievalDebugWorkflowResult:
    """
    Retrieval-debug workflow entry point with the same explicit-filter and
    auto-detection behavior used by QA.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

    if requested_drug is not None and requested_drug.strip():
        drug_mention_detection = build_not_attempted_detection(query)

        if not drug_resolution.can_retrieve:
            empty_pack = build_empty_evidence_pack(
                query=query,
                retrieval_family=retrieval_family,
            )

            return RetrievalDebugWorkflowResult(
                evidence_pack=empty_pack,
                drug_resolution=drug_resolution,
                drug_mention_detection=drug_mention_detection,
                retrieval_drug=None,
            )

        retrieval_drug = drug_resolution.retrieval_drug

    else:
        drug_mention_detection = detect_drug_mention_from_query(query)

        retrieval_drug = (
            drug_mention_detection.retrieval_drug
            if drug_mention_detection.can_filter
            else None
        )

    evidence_pack = build_evidence_pack(
        query=query,
        concept_name=retrieval_drug,
        retrieval_family=retrieval_family,
        top_k=top_k,
    )

    return RetrievalDebugWorkflowResult(
        evidence_pack=evidence_pack,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_mention_detection,
        retrieval_drug=retrieval_drug,
    )