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
from medlabeliq.orchestration.source_router import (
    SourceRoutePlan,
    plan_source_route,
)
from medlabeliq.rxnorm.identity_answer import (
    answer_rxnorm_identity_query,
)
from medlabeliq.rxnorm.identity_models import (
    RxNormIdentityEvidence,
)


@dataclass(frozen=True)
class QAWorkflowResult:
    generated: GeneratedAnswer
    drug_resolution: DrugFilterResolution
    drug_mention_detection: DrugMentionDetection
    retrieval_drug: str | None
    family_plan: RetrievalFamilyPlan | None = None
    retrieval_family: str | None = None
    source_plan: SourceRoutePlan | None = None
    identity_evidence: list[RxNormIdentityEvidence] | None = None


@dataclass(frozen=True)
class RetrievalDebugWorkflowResult:
    evidence_pack: EvidencePack
    drug_resolution: DrugFilterResolution
    drug_mention_detection: DrugMentionDetection
    retrieval_drug: str | None
    family_plan: RetrievalFamilyPlan | None = None
    retrieval_family: str | None = None
    source_plan: SourceRoutePlan | None = None


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
    - retrieval-family planning,
    - source-aware routing,
    - RxNorm identity execution for identity/equivalence queries.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

    family_plan, effective_retrieval_family = resolve_effective_family(
        query=query,
        requested_family=retrieval_family,
    )

    if requested_drug is not None and requested_drug.strip():
        drug_mention_detection = build_not_attempted_detection(query)

        if not drug_resolution.can_retrieve:
            source_plan = plan_source_route(
                query,
                requested_family=retrieval_family,
                family_plan=family_plan,
                drug_resolution=drug_resolution,
                drug_mention_detection=drug_mention_detection,
            )

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
                source_plan=source_plan,
                identity_evidence=None,
            )

        retrieval_drug = drug_resolution.retrieval_drug

    else:
        drug_mention_detection = detect_drug_mention_from_query(query)

        retrieval_drug = (
            drug_mention_detection.retrieval_drug
            if drug_mention_detection.can_filter
            else None
        )

    source_plan = plan_source_route(
        query,
        requested_family=retrieval_family,
        family_plan=family_plan,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_mention_detection,
    )

    if (
        source_plan.status == "routed_rxnorm_identity"
        and source_plan.selected_source == "rxnorm_identity"
    ):
        identity_result = answer_rxnorm_identity_query(
            query,
            intent=source_plan.intent,
        )

        return QAWorkflowResult(
            generated=identity_result.generated,
            drug_resolution=drug_resolution,
            drug_mention_detection=drug_mention_detection,
            retrieval_drug=retrieval_drug,
            family_plan=family_plan,
            retrieval_family=effective_retrieval_family,
            source_plan=source_plan,
            identity_evidence=identity_result.evidence_items,
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
        source_plan=source_plan,
        identity_evidence=None,
    )


def build_debug_evidence_pack_with_drug_resolution(
    query: str,
    *,
    requested_drug: str | None = None,
    retrieval_family: str | None = None,
    top_k: int | None = None,
) -> RetrievalDebugWorkflowResult:
    """
    Retrieval-debug workflow entry point with:
    - drug normalization,
    - query-level drug detection,
    - retrieval-family planning,
    - source route planning.

    Identity execution is handled by the QA flow, not retrieval debug.
    """
    drug_resolution = resolve_optional_drug_filter(requested_drug)

    family_plan, effective_retrieval_family = resolve_effective_family(
        query=query,
        requested_family=retrieval_family,
    )

    if requested_drug is not None and requested_drug.strip():
        drug_mention_detection = build_not_attempted_detection(query)

        if not drug_resolution.can_retrieve:
            source_plan = plan_source_route(
                query,
                requested_family=retrieval_family,
                family_plan=family_plan,
                drug_resolution=drug_resolution,
                drug_mention_detection=drug_mention_detection,
            )

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
                source_plan=source_plan,
            )

        retrieval_drug = drug_resolution.retrieval_drug

    else:
        drug_mention_detection = detect_drug_mention_from_query(query)

        retrieval_drug = (
            drug_mention_detection.retrieval_drug
            if drug_mention_detection.can_filter
            else None
        )

    source_plan = plan_source_route(
        query,
        requested_family=retrieval_family,
        family_plan=family_plan,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_mention_detection,
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
        source_plan=source_plan,
    )