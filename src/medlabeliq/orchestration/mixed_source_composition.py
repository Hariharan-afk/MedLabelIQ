from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from medlabeliq.generation.answer_generator import (
    GeneratedAnswer,
    answer_query,
)
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack
from medlabeliq.orchestration.source_router import SourceRoutePlan
from medlabeliq.rxnorm.identity_answer import (
    answer_rxnorm_identity_query,
)
from medlabeliq.rxnorm.identity_models import (
    RxNormIdentityEvidence,
)


MIXED_SOURCE_COMPOSITION_SAFETY_NOTE = (
    "This answer combines RxNorm medication identity relationships with "
    "retrieved drug-label evidence and is not a substitute for advice from "
    "a qualified clinician or pharmacist."
)

MIXED_SOURCE_INSUFFICIENT_ANSWER = (
    "The query combines medication identity and clinical label claims, but "
    "the available evidence was not sufficient to answer both parts reliably."
)

MIXED_SOURCE_INSUFFICIENT_SUMMARY = (
    "At least one branch of the mixed-source workflow did not produce enough "
    "support to compose a reliable combined answer."
)


MixedSourceCompositionStatus = Literal[
    "composed_answered",
    "unsupported_decomposition",
    "missing_retrieval_drug",
    "missing_identity_intent",
    "identity_insufficient",
    "clinical_insufficient",
]


@dataclass(frozen=True)
class MixedSourceCompositionMetadata:
    status: MixedSourceCompositionStatus
    identity_query: str | None
    clinical_query: str | None
    identity_intent: str | None


@dataclass(frozen=True)
class MixedSourceCompositionExecution:
    generated: GeneratedAnswer
    identity_evidence: list[RxNormIdentityEvidence]
    metadata: MixedSourceCompositionMetadata


# =============================================================================
# Generic helpers
# =============================================================================

def empty_label_evidence_pack(query: str) -> EvidencePack:
    return EvidencePack(
        query=query,
        concept_name=None,
        retrieval_family=None,
        evidence_items=[],
    )


def ensure_question_mark(value: str) -> str:
    cleaned = value.strip().rstrip(" \t\n\r?.!")

    if not cleaned:
        return ""

    return f"{cleaned}?"


def capitalize_first(value: str) -> str:
    if not value:
        return value

    return value[0].upper() + value[1:]


def build_mixed_grounded_answer(
    *,
    status: Literal["answered", "insufficient_evidence"],
    answer_text: str,
    citations: list[str],
    evidence_summary: str,
) -> GroundedAnswer:
    return GroundedAnswer(
        status=status,
        answer=answer_text,
        citations=citations,
        evidence_summary=evidence_summary,
        safety_note=MIXED_SOURCE_COMPOSITION_SAFETY_NOTE,
    )


def build_mixed_insufficient_execution(
    *,
    query: str,
    status: MixedSourceCompositionStatus,
    identity_query: str | None,
    clinical_query: str | None,
    identity_intent: str | None,
    evidence_pack: EvidencePack | None = None,
    identity_evidence: list[RxNormIdentityEvidence] | None = None,
) -> MixedSourceCompositionExecution:
    answer = build_mixed_grounded_answer(
        status="insufficient_evidence",
        answer_text=MIXED_SOURCE_INSUFFICIENT_ANSWER,
        citations=[],
        evidence_summary=MIXED_SOURCE_INSUFFICIENT_SUMMARY,
    )

    generated = GeneratedAnswer(
        evidence_pack=evidence_pack or empty_label_evidence_pack(query),
        answer=answer,
        raw_model_output=None,
        verification=None,
        proposed_answer=answer,
        verification_overrode_answer=False,
        guardrail_triggered=False,
        guardrail_reason=None,
    )

    metadata = MixedSourceCompositionMetadata(
        status=status,
        identity_query=identity_query,
        clinical_query=clinical_query,
        identity_intent=identity_intent,
    )

    return MixedSourceCompositionExecution(
        generated=generated,
        identity_evidence=identity_evidence or [],
        metadata=metadata,
    )


# =============================================================================
# Decomposition helpers
# =============================================================================

def strongest_identity_intent(
    source_plan: SourceRoutePlan,
) -> str | None:
    if not source_plan.matches:
        return None

    strongest_match = max(
        source_plan.matches,
        key=lambda match: match.score,
    )

    return strongest_match.intent


def normalize_clinical_clause(
    clause: str,
    *,
    retrieval_drug: str,
) -> str:
    cleaned = clause.strip().rstrip(" \t\n\r?.!")
    normalized = " ".join(cleaned.casefold().split())

    if normalized in {
        "what is it used for",
        "what's it used for",
        "what is this drug used for",
        "what's this drug used for",
    }:
        return f"What is {retrieval_drug} used for?"

    rewritten = re.sub(
        r"\bthis drug\b",
        retrieval_drug,
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )

    rewritten = re.sub(
        r"\bit\b",
        retrieval_drug,
        rewritten,
        count=1,
        flags=re.IGNORECASE,
    )

    rewritten = capitalize_first(rewritten)

    return ensure_question_mark(rewritten)


def decompose_mixed_source_query(
    query: str,
    *,
    retrieval_drug: str,
    identity_intent: str | None,
) -> MixedSourceCompositionMetadata:
    pieces = re.split(
        r"\s+and\s+",
        query.strip(),
        maxsplit=1,
        flags=re.IGNORECASE,
    )

    if len(pieces) != 2:
        return MixedSourceCompositionMetadata(
            status="unsupported_decomposition",
            identity_query=None,
            clinical_query=None,
            identity_intent=identity_intent,
        )

    identity_clause = pieces[0].strip()
    clinical_clause = pieces[1].strip()

    identity_query = ensure_question_mark(
        capitalize_first(identity_clause)
    )

    clinical_query = normalize_clinical_clause(
        clinical_clause,
        retrieval_drug=retrieval_drug,
    )

    if not identity_query or not clinical_query:
        return MixedSourceCompositionMetadata(
            status="unsupported_decomposition",
            identity_query=identity_query or None,
            clinical_query=clinical_query or None,
            identity_intent=identity_intent,
        )

    return MixedSourceCompositionMetadata(
        status="composed_answered",
        identity_query=identity_query,
        clinical_query=clinical_query,
        identity_intent=identity_intent,
    )


# =============================================================================
# Public mixed-source execution entry point
# =============================================================================

def execute_mixed_source_composition(
    *,
    query: str,
    source_plan: SourceRoutePlan,
    retrieval_drug: str | None,
    retrieval_family: str | None,
    top_k: int | None,
) -> MixedSourceCompositionExecution:
    identity_intent = strongest_identity_intent(source_plan)

    if retrieval_drug is None:
        return build_mixed_insufficient_execution(
            query=query,
            status="missing_retrieval_drug",
            identity_query=None,
            clinical_query=None,
            identity_intent=identity_intent,
        )

    if identity_intent is None:
        return build_mixed_insufficient_execution(
            query=query,
            status="missing_identity_intent",
            identity_query=None,
            clinical_query=None,
            identity_intent=None,
        )

    metadata = decompose_mixed_source_query(
        query,
        retrieval_drug=retrieval_drug,
        identity_intent=identity_intent,
    )

    if metadata.status != "composed_answered":
        return build_mixed_insufficient_execution(
            query=query,
            status=metadata.status,
            identity_query=metadata.identity_query,
            clinical_query=metadata.clinical_query,
            identity_intent=metadata.identity_intent,
        )

    identity_result = answer_rxnorm_identity_query(
        metadata.identity_query,
        intent=identity_intent,
    )

    identity_answer = identity_result.generated.answer

    if identity_answer.status != "answered":
        return build_mixed_insufficient_execution(
            query=query,
            status="identity_insufficient",
            identity_query=metadata.identity_query,
            clinical_query=metadata.clinical_query,
            identity_intent=metadata.identity_intent,
            identity_evidence=identity_result.evidence_items,
        )

    clinical_generated = answer_query(
        query=metadata.clinical_query,
        concept_name=retrieval_drug,
        retrieval_family=retrieval_family,
        top_k=top_k,
    )

    clinical_answer = clinical_generated.answer

    if clinical_answer.status != "answered":
        return build_mixed_insufficient_execution(
            query=query,
            status="clinical_insufficient",
            identity_query=metadata.identity_query,
            clinical_query=metadata.clinical_query,
            identity_intent=metadata.identity_intent,
            evidence_pack=clinical_generated.evidence_pack,
            identity_evidence=identity_result.evidence_items,
        )

    combined_answer = build_mixed_grounded_answer(
        status="answered",
        answer_text=(
            f"{identity_answer.answer.rstrip()} "
            f"{clinical_answer.answer.rstrip()}"
        ),
        citations=[
            *identity_answer.citations,
            *clinical_answer.citations,
        ],
        evidence_summary=(
            f"RxNorm identity support: {identity_answer.evidence_summary} "
            f"Drug-label support: {clinical_answer.evidence_summary}"
        ),
    )

    generated = GeneratedAnswer(
        evidence_pack=clinical_generated.evidence_pack,
        answer=combined_answer,
        raw_model_output=None,
        verification=clinical_generated.verification,
        proposed_answer=combined_answer,
        verification_overrode_answer=(
            clinical_generated.verification_overrode_answer
        ),
        guardrail_triggered=clinical_generated.guardrail_triggered,
        guardrail_reason=clinical_generated.guardrail_reason,
    )

    final_metadata = MixedSourceCompositionMetadata(
        status="composed_answered",
        identity_query=metadata.identity_query,
        clinical_query=metadata.clinical_query,
        identity_intent=metadata.identity_intent,
    )

    return MixedSourceCompositionExecution(
        generated=generated,
        identity_evidence=identity_result.evidence_items,
        metadata=final_metadata,
    )