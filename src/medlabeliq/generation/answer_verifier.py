from __future__ import annotations

from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack
from medlabeliq.generation.json_parser import extract_json_object
from medlabeliq.generation.llm_client import generate_json_answer
from medlabeliq.generation.verification_prompt import (
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_user_prompt,
)
from medlabeliq.generation.verification_schema import AnswerVerification


def verify_generated_answer(
    *,
    question: str,
    answer: GroundedAnswer,
    evidence_pack: EvidencePack,
) -> AnswerVerification:
    """
    Verify whether the generated answer is directly supported by the evidence it cited.
    """
    verifier_prompt = build_verifier_user_prompt(
        question=question,
        answer=answer,
        evidence_pack=evidence_pack,
    )

    raw_output = generate_json_answer(
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=verifier_prompt,
    )

    payload = extract_json_object(raw_output)
    verification = AnswerVerification.model_validate(payload)

    provided_evidence_ids = evidence_pack.evidence_ids

    invalid_ids = [
        evidence_id
        for evidence_id in verification.cited_evidence_used
        if evidence_id not in provided_evidence_ids
    ]

    if invalid_ids:
        raise ValueError(
            f"Verifier referenced evidence IDs not in the evidence pack: {invalid_ids}"
        )

    return verification