from __future__ import annotations

from dataclasses import dataclass

from medlabeliq.config.settings import settings
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.answer_verifier import verify_generated_answer
from medlabeliq.generation.claim_guardrails import (
    assess_guarantee_claim_support,
    assess_negative_treatment_claim_support,
)
from medlabeliq.generation.evidence_pack import (
    EvidencePack,
    build_evidence_pack,
)
from medlabeliq.generation.json_parser import parse_grounded_answer
from medlabeliq.generation.llm_client import generate_json_answer
from medlabeliq.generation.prompt_builder import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from medlabeliq.generation.verification_schema import AnswerVerification


APPLICATION_SAFETY_NOTE = (
    "This answer summarizes retrieved drug-label evidence and is not a "
    "substitute for advice from a qualified clinician or pharmacist."
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "The retrieved drug-label evidence is not sufficient to answer this "
    "question reliably."
)

INSUFFICIENT_EVIDENCE_SUMMARY = (
    "No retrieved evidence directly established the requested claim."
)


@dataclass(frozen=True)
class GeneratedAnswer:
    evidence_pack: EvidencePack
    answer: GroundedAnswer
    raw_model_output: str | None
    verification: AnswerVerification | None = None

    # Diagnostics
    proposed_answer: GroundedAnswer | None = None
    verification_overrode_answer: bool = False
    guardrail_triggered: bool = False
    guardrail_reason: str | None = None


def make_insufficient_answer() -> GroundedAnswer:
    return GroundedAnswer(
        status="insufficient_evidence",
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        citations=[],
        evidence_summary=INSUFFICIENT_EVIDENCE_SUMMARY,
        safety_note=APPLICATION_SAFETY_NOTE,
    )


def build_deterministic_insufficient_answer(
    evidence_pack: EvidencePack,
    *,
    raw_model_output: str | None = None,
    proposed_answer: GroundedAnswer | None = None,
    verification: AnswerVerification | None = None,
    verification_overrode_answer: bool = False,
    guardrail_triggered: bool = False,
    guardrail_reason: str | None = None,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        evidence_pack=evidence_pack,
        answer=make_insufficient_answer(),
        raw_model_output=raw_model_output,
        verification=verification,
        proposed_answer=proposed_answer,
        verification_overrode_answer=verification_overrode_answer,
        guardrail_triggered=guardrail_triggered,
        guardrail_reason=guardrail_reason,
    )


def answer_query(
    query: str,
    *,
    concept_name: str | None = None,
    retrieval_family: str | None = None,
    top_k: int | None = None,
) -> GeneratedAnswer:
    evidence_pack = build_evidence_pack(
        query=query,
        concept_name=concept_name,
        retrieval_family=retrieval_family,
        top_k=top_k,
    )

    if not evidence_pack.evidence_items:
        return build_deterministic_insufficient_answer(evidence_pack)

    user_prompt = build_user_prompt(
        query=query,
        evidence_pack=evidence_pack,
    )

    raw_output = generate_json_answer(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    parsed_answer = parse_grounded_answer(
        raw_output,
        allowed_evidence_ids=evidence_pack.evidence_ids,
    )

    proposed_answer = parsed_answer

    if parsed_answer.status == "insufficient_evidence":
        return build_deterministic_insufficient_answer(
            evidence_pack,
            raw_model_output=raw_output,
            proposed_answer=proposed_answer,
        )

    answered = parsed_answer.model_copy(
        update={"safety_note": APPLICATION_SAFETY_NOTE}
    )

    # ---------------------------------------------------------------------
    # Deterministic guarantee-claim guardrail
    # ---------------------------------------------------------------------
    guarantee_guardrail = assess_guarantee_claim_support(
        question=query,
        answer=answered,
        evidence_pack=evidence_pack,
    )

    if guarantee_guardrail.should_abstain:
        return build_deterministic_insufficient_answer(
            evidence_pack,
            raw_model_output=raw_output,
            proposed_answer=proposed_answer,
            guardrail_triggered=True,
            guardrail_reason=guarantee_guardrail.reason,
        )

    negative_treatment_guardrail = assess_negative_treatment_claim_support(
        question=query,
        answer=answered,
        evidence_pack=evidence_pack,
    )

    if negative_treatment_guardrail.should_abstain:
        return build_deterministic_insufficient_answer(
            evidence_pack,
            raw_model_output=raw_output,
            proposed_answer=proposed_answer,
            guardrail_triggered=True,
            guardrail_reason=negative_treatment_guardrail.reason,
        )

    # ---------------------------------------------------------------------
    # LLM-based evidence-support verifier
    # ---------------------------------------------------------------------
    verification: AnswerVerification | None = None

    if settings.answer_verifier_enabled:
        verification = verify_generated_answer(
            question=query,
            answer=answered,
            evidence_pack=evidence_pack,
        )

        if verification.verdict != "supported":
            return build_deterministic_insufficient_answer(
                evidence_pack,
                raw_model_output=raw_output,
                proposed_answer=proposed_answer,
                verification=verification,
                verification_overrode_answer=True,
            )

    return GeneratedAnswer(
        evidence_pack=evidence_pack,
        answer=answered,
        raw_model_output=raw_output,
        verification=verification,
        proposed_answer=proposed_answer,
        verification_overrode_answer=False,
        guardrail_triggered=False,
        guardrail_reason=None,
    )