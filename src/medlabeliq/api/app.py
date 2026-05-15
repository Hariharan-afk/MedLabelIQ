from __future__ import annotations

import logging
import time

from typing import Any

from fastapi import FastAPI, HTTPException, status

from medlabeliq.api.schemas import (
    AnswerAPIResponse,
    AnswerRequest,
    DiagnosticsResponse,
    EvidenceItemResponse,
    GroundedAnswerResponse,
    HealthComponentResponse,
    HealthResponse,
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    RootResponse,
    VerificationResponse,
)
from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection
from medlabeliq.generation.answer_generator import answer_query
from medlabeliq.generation.evidence_pack import (
    EvidenceItem,
    build_evidence_pack,
)
from medlabeliq.qdrant_store.client import get_qdrant_client
from medlabeliq.observability.qa_logger import log_qa_interaction


app = FastAPI(
    title="MedLabelIQ API",
    version="0.1.0",
    description=(
        "Grounded medication-label QA API backed by DailyMed SPL ingestion, "
        "hybrid retrieval, Groq answer generation, verifier diagnostics, "
        "and deterministic abstention guardrails."
    ),
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------

def serialize_evidence_item(item: EvidenceItem) -> EvidenceItemResponse:
    return EvidenceItemResponse(
        evidence_id=item.evidence_id,
        chunk_id=item.chunk_id,
        section_id=item.section_id,
        drug=item.concept_name,
        retrieval_family=item.retrieval_family,
        canonical_section_name=item.canonical_section_name,
        nearest_canonical_section_name=item.nearest_canonical_section_name,
        heading=item.heading,
        set_id=item.set_id,
        version_number=item.version_number,
        source_label=item.source_label,
        chunk_text=item.chunk_text,
        hybrid_score=item.hybrid_score,
        lexical_rank=item.lexical_rank,
        dense_rank=item.dense_rank,
    )


def serialize_verification(verification) -> VerificationResponse | None:
    if verification is None:
        return None

    return VerificationResponse(
        verdict=verification.verdict,
        rationale=verification.rationale,
        cited_evidence_used=verification.cited_evidence_used,
    )


# ---------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------

@app.get(
    "/",
    response_model=RootResponse,
    tags=["Service"],
)
def root() -> RootResponse:
    return RootResponse(
        service="MedLabelIQ API",
        status="running",
        docs="/docs",
        endpoints=[
            "GET /health",
            "POST /qa/answer",
            "POST /retrieval/debug",
        ],
    )


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

def check_postgres() -> HealthComponentResponse:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok;")
                row = cur.fetchone()

        if row is None:
            return HealthComponentResponse(
                status="error",
                detail="PostgreSQL query returned no row.",
            )

        return HealthComponentResponse(
            status="ok",
            detail="PostgreSQL connection successful.",
        )

    except Exception as exc:
        return HealthComponentResponse(
            status="error",
            detail=f"{type(exc).__name__}: {exc}",
        )


def check_qdrant() -> HealthComponentResponse:
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(
            settings.qdrant_collection
        )

        return HealthComponentResponse(
            status="ok",
            detail=(
                f"Collection '{settings.qdrant_collection}' available "
                f"with {collection_info.points_count} points."
            ),
        )

    except Exception as exc:
        return HealthComponentResponse(
            status="error",
            detail=f"{type(exc).__name__}: {exc}",
        )


def check_llm_configuration() -> HealthComponentResponse:
    if not settings.llm_api_key:
        return HealthComponentResponse(
            status="not_configured",
            detail="LLM_API_KEY is empty.",
        )

    return HealthComponentResponse(
        status="ok",
        detail=(
            f"LLM configured with model '{settings.llm_model}' "
            f"via '{settings.llm_base_url}'."
        ),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
)
def health() -> HealthResponse:
    postgres_status = check_postgres()
    qdrant_status = check_qdrant()
    llm_status = check_llm_configuration()

    overall_ok = all(
        [
            postgres_status.status == "ok",
            qdrant_status.status == "ok",
            llm_status.status == "ok",
        ]
    )

    return HealthResponse(
        status="ok" if overall_ok else "degraded",
        service="MedLabelIQ API",
        postgres=postgres_status,
        qdrant=qdrant_status,
        llm=llm_status,
    )


# ---------------------------------------------------------------------
# QA answer endpoint
# ---------------------------------------------------------------------

@app.post(
    "/qa/answer",
    response_model=AnswerAPIResponse,
    response_model_exclude_none=True,
    tags=["QA"],
)
def answer_question(request: AnswerRequest) -> AnswerAPIResponse:
    started = time.perf_counter()

    try:
        generated = answer_query(
            query=request.query,
            concept_name=request.drug,
            retrieval_family=request.family,
            top_k=request.top_k,
        )

        api_latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        answer = generated.answer
        evidence_pack = generated.evidence_pack

        result = GroundedAnswerResponse(
            status=answer.status,
            answer=answer.answer,
            citations=answer.citations,
            evidence_summary=answer.evidence_summary,
            safety_note=answer.safety_note,
        )

        evidence_response = None
        if request.include_evidence:
            evidence_response = [
                serialize_evidence_item(item)
                for item in evidence_pack.evidence_items
            ]

        diagnostics_response = None
        if request.include_diagnostics:
            proposed_status = (
                generated.proposed_answer.status
                if generated.proposed_answer is not None
                else None
            )

            diagnostics_response = DiagnosticsResponse(
                evidence_count=len(evidence_pack.evidence_items),
                proposed_status=proposed_status,
                verification_enabled=settings.answer_verifier_enabled,
                verification=serialize_verification(
                    generated.verification
                ),
                verification_overrode_answer=(
                    generated.verification_overrode_answer
                ),
                guardrail_triggered=generated.guardrail_triggered,
                guardrail_reason=generated.guardrail_reason,
            )

        request_log_id: str | None = None

        try:
            request_log_id = log_qa_interaction(
                query=request.query,
                drug=request.drug,
                family=request.family,
                top_k=request.top_k,
                include_evidence=request.include_evidence,
                include_diagnostics=request.include_diagnostics,
                generated=generated,
                api_latency_ms=api_latency_ms,
            )
        except Exception as log_exc:
            logger.exception(
                "QA observability logging failed: %s",
                log_exc,
            )

        return AnswerAPIResponse(
            query=request.query,
            drug=request.drug,
            family=request.family,
            request_log_id=request_log_id,
            result=result,
            evidence=evidence_response,
            diagnostics=diagnostics_response,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


# ---------------------------------------------------------------------
# Retrieval debug endpoint
# ---------------------------------------------------------------------

@app.post(
    "/retrieval/debug",
    response_model=RetrievalDebugResponse,
    tags=["Retrieval"],
)
def retrieval_debug(
    request: RetrievalDebugRequest,
) -> RetrievalDebugResponse:
    try:
        evidence_pack = build_evidence_pack(
            query=request.query,
            concept_name=request.drug,
            retrieval_family=request.family,
            top_k=request.top_k,
        )

        evidence = [
            serialize_evidence_item(item)
            for item in evidence_pack.evidence_items
        ]

        return RetrievalDebugResponse(
            query=request.query,
            drug=request.drug,
            family=request.family,
            evidence_count=len(evidence),
            evidence=evidence,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc