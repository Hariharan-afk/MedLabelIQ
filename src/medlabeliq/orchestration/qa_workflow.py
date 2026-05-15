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


@dataclass(frozen=True)
class QAWorkflowResult:
    generated: GeneratedAnswer
    drug_resolution: DrugFilterResolution


@dataclass(frozen=True)
class RetrievalDebugWorkflowResult:
    evidence_pack: EvidencePack
    drug_resolution: DrugFilterResolution


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
    QA workflow entry point with drug-filter normalization.

    If a requested drug filter cannot be mapped safely to exactly one indexed
    corpus concept, the workflow fails closed with deterministic abstention.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

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
        )

    generated = answer_query(
        query=query,
        concept_name=drug_resolution.retrieval_drug,
        retrieval_family=retrieval_family,
        top_k=top_k,
    )

    return QAWorkflowResult(
        generated=generated,
        drug_resolution=drug_resolution,
    )


def build_debug_evidence_pack_with_drug_resolution(
    query: str,
    *,
    requested_drug: str | None = None,
    retrieval_family: str | None = None,
    top_k: int | None = None,
) -> RetrievalDebugWorkflowResult:
    """
    Retrieval-debug workflow entry point with the same normalization behavior
    used by QA.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

    if not drug_resolution.can_retrieve:
        empty_pack = build_empty_evidence_pack(
            query=query,
            retrieval_family=retrieval_family,
        )

        return RetrievalDebugWorkflowResult(
            evidence_pack=empty_pack,
            drug_resolution=drug_resolution,
        )

    evidence_pack = build_evidence_pack(
        query=query,
        concept_name=drug_resolution.retrieval_drug,
        retrieval_family=retrieval_family,
        top_k=top_k,
    )

    return RetrievalDebugWorkflowResult(
        evidence_pack=evidence_pack,
        drug_resolution=drug_resolution,
    )