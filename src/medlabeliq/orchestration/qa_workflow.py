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
from medlabeliq.orchestration.retrieval_family_planner import (
    RetrievalFamilyPlan,
    plan_retrieval_family,
)


@dataclass(frozen=True)
class QAWorkflowResult:
    generated: GeneratedAnswer
    drug_resolution: DrugFilterResolution
    drug_mention_detection: DrugMentionDetection
    retrieval_drug: str | None
    family_plan: RetrievalFamilyPlan | None = None
    retrieval_family: str | None = None


@dataclass(frozen=True)
class RetrievalDebugWorkflowResult:
    evidence_pack: EvidencePack
    drug_resolution: DrugFilterResolution
    drug_mention_detection: DrugMentionDetection
    retrieval_drug: str | None
    family_plan: RetrievalFamilyPlan | None = None
    retrieval_family: str | None = None


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


def resolve_effective_family(
    *,
    query: str,
    requested_family: str | None,
) -> tuple[RetrievalFamilyPlan, str | None]:
    family_plan = plan_retrieval_family(
        query,
        requested_family=requested_family,
    )

    if requested_family is not None and requested_family.strip():
        return family_plan, requested_family

    if family_plan.can_filter:
        return family_plan, family_plan.planned_family

    return family_plan, None


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
    - automatic query-level drug mention detection,
    - retrieval-family planning when no explicit family filter is provided.

    Policy:
    - Explicit drug filter unresolved -> fail closed.
    - No explicit drug filter and auto-detection succeeds -> use detected drug.
    - No explicit drug filter and auto-detection fails/ambiguous -> broad drug retrieval.
    - Explicit family filter -> use it directly.
    - No explicit family filter and planner selects one safe family -> filter to it.
    - Group/ambiguous/no-route family plans -> broad family retrieval.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

    family_plan, effective_retrieval_family = resolve_effective_family(
        query=query,
        requested_family=retrieval_family,
    )

    if requested_drug is not None and requested_drug.strip():
        drug_mention_detection = build_not_attempted_detection(query)

        if not drug_resolution.can_retrieve:
            empty_pack = build_empty_evidence_pack(
                query=query,
                retrieval_family=effective_retrieval_family,
            )

            generated = build_deterministic_insufficient_answer(
                empty_pack,
            )

            return QAWorkflowResult(
                generated=generated,
                drug_resolution=drug_resolution,
                drug_mention_detection=drug_mention_detection,
                retrieval_drug=None,
                family_plan=family_plan,
                retrieval_family=effective_retrieval_family,
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
        retrieval_family=effective_retrieval_family,
        top_k=top_k,
    )

    return QAWorkflowResult(
        generated=generated,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_mention_detection,
        retrieval_drug=retrieval_drug,
        family_plan=family_plan,
        retrieval_family=effective_retrieval_family,
    )


def build_debug_evidence_pack_with_drug_resolution(
    query: str,
    *,
    requested_drug: str | None = None,
    retrieval_family: str | None = None,
    top_k: int | None = None,
) -> RetrievalDebugWorkflowResult:
    """
    Retrieval-debug workflow entry point with the same:
    - drug normalization,
    - query-level drug detection,
    - retrieval-family planning
    used by the QA answer flow.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

    family_plan, effective_retrieval_family = resolve_effective_family(
        query=query,
        requested_family=retrieval_family,
    )

    if requested_drug is not None and requested_drug.strip():
        drug_mention_detection = build_not_attempted_detection(query)

        if not drug_resolution.can_retrieve:
            empty_pack = build_empty_evidence_pack(
                query=query,
                retrieval_family=effective_retrieval_family,
            )

            return RetrievalDebugWorkflowResult(
                evidence_pack=empty_pack,
                drug_resolution=drug_resolution,
                drug_mention_detection=drug_mention_detection,
                retrieval_drug=None,
                family_plan=family_plan,
                retrieval_family=effective_retrieval_family,
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
        retrieval_family=effective_retrieval_family,
        top_k=top_k,
    )

    return RetrievalDebugWorkflowResult(
        evidence_pack=evidence_pack,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_mention_detection,
        retrieval_drug=retrieval_drug,
        family_plan=family_plan,
        retrieval_family=effective_retrieval_family,
    )