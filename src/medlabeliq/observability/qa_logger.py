from __future__ import annotations

from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection
from medlabeliq.generation.answer_generator import GeneratedAnswer


def log_qa_interaction(
    *,
    query: str,
    drug: str | None,
    family: str | None,
    top_k: int | None,
    include_evidence: bool,
    include_diagnostics: bool,
    generated: GeneratedAnswer,
    api_latency_ms: float,
    requested_drug: str | None = None,
    resolved_drug: str | None = None,
    drug_resolution_status: str | None = None,
    detected_drug_mention: str | None = None,
    drug_mention_detection_status: str | None = None,
    requested_family: str | None = None,
    family_plan_status: str | None = None,
    family_plan_intent: str | None = None,
) -> str:
    """
    Persist one QA request plus its evidence rows.

    Returns:
        request_log_id as a string.
    """
    request_log_uuid = uuid4()
    request_log_id = str(request_log_uuid)

    answer = generated.answer
    evidence_pack = generated.evidence_pack

    proposed_status = (
        generated.proposed_answer.status
        if generated.proposed_answer is not None
        else None
    )

    verification = generated.verification

    verification_verdict = (
        verification.verdict
        if verification is not None
        else None
    )

    verification_rationale = (
        verification.rationale
        if verification is not None
        else None
    )

    verification_evidence_used = (
        verification.cited_evidence_used
        if verification is not None
        else []
    )

    cited_ids = set(answer.citations)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO qa_request_log (
                    request_log_id,
                    query_text,
                    drug_filter,
                    requested_drug_filter,
                    resolved_drug_filter,
                    drug_resolution_status,
                    detected_drug_mention,
                    drug_mention_detection_status,
                    family_filter,
                    requested_family_filter,
                    family_plan_status,
                    family_plan_intent,
                    top_k,
                    include_evidence,
                    include_diagnostics,
                    final_status,
                    final_answer,
                    final_citations,
                    final_evidence_summary,
                    safety_note,
                    proposed_status,
                    verification_enabled,
                    verification_verdict,
                    verification_rationale,
                    verification_evidence_used,
                    verification_overrode_answer,
                    guardrail_triggered,
                    guardrail_reason,
                    evidence_count,
                    api_latency_ms
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                );
                """,
                (
                    request_log_uuid,
                    query,
                    drug,
                    requested_drug,
                    resolved_drug,
                    drug_resolution_status,
                    detected_drug_mention,
                    drug_mention_detection_status,
                    family,
                    requested_family,
                    family_plan_status,
                    family_plan_intent,
                    top_k,
                    include_evidence,
                    include_diagnostics,
                    answer.status,
                    answer.answer,
                    Jsonb(answer.citations),
                    answer.evidence_summary,
                    answer.safety_note,
                    proposed_status,
                    settings.answer_verifier_enabled,
                    verification_verdict,
                    verification_rationale,
                    Jsonb(verification_evidence_used),
                    generated.verification_overrode_answer,
                    generated.guardrail_triggered,
                    generated.guardrail_reason,
                    len(evidence_pack.evidence_items),
                    api_latency_ms,
                ),
            )

            for position, item in enumerate(
                evidence_pack.evidence_items,
                start=1,
            ):
                cur.execute(
                    """
                    INSERT INTO qa_evidence_log (
                        request_log_id,
                        evidence_id,
                        evidence_position,
                        chunk_id,
                        section_id,
                        concept_name,
                        retrieval_family,
                        canonical_section_name,
                        nearest_canonical_section_name,
                        heading,
                        set_id,
                        version_number,
                        hybrid_score,
                        lexical_rank,
                        dense_rank,
                        was_cited
                    )
                    VALUES (
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s
                    );
                    """,
                    (
                        request_log_uuid,
                        item.evidence_id,
                        position,
                        UUID(item.chunk_id),
                        UUID(item.section_id),
                        item.concept_name,
                        item.retrieval_family,
                        item.canonical_section_name,
                        item.nearest_canonical_section_name,
                        item.heading,
                        item.set_id,
                        item.version_number,
                        item.hybrid_score,
                        item.lexical_rank,
                        item.dense_rank,
                        item.evidence_id in cited_ids,
                    ),
                )

        conn.commit()

    return request_log_id