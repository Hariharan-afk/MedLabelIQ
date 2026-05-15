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
    CorpusStatsResponse,
    DrugListResponse,
    DrugSummaryResponse,
    RetrievalFamilyListResponse,
    RetrievalFamilySummaryResponse,
    DrugNormalizationRequest,
    DrugNormalizationResponse,
    RxNormVersionResponse,
    DrugFilterResolutionResponse,
    RxNormConceptResponse,
    DrugMentionDetectionResponse,
    RetrievalFamilyPlanResponse,
    RetrievalFamilySignalMatchResponse,
    SourceRoutePlanResponse,
    SourceRouteSignalMatchResponse,
    RxNormIdentityEvidenceResponse,
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
from medlabeliq.corpus.metadata import (
    collect_corpus_stats,
    list_drug_summaries,
    list_retrieval_family_summaries,
)
from medlabeliq.rxnorm.client import RxNormClient
from medlabeliq.rxnorm.resolver import resolve_drug_term
from medlabeliq.orchestration.drug_filter_resolution import (
    DrugFilterResolution,
)
from medlabeliq.orchestration.qa_workflow import (
    answer_query_with_drug_resolution,
    build_debug_evidence_pack_with_drug_resolution,
)
from medlabeliq.orchestration.drug_mention_detection import (
    DrugMentionDetection,
)
from medlabeliq.orchestration.retrieval_family_planner import (
    RetrievalFamilyPlan,
)
from medlabeliq.orchestration.source_router import (
    SourceRoutePlan,
)
from medlabeliq.rxnorm.identity_models import (
    RxNormIdentityEvidence,
)

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


def serialize_drug_filter_resolution(
    resolution: DrugFilterResolution,
) -> DrugFilterResolutionResponse:
    selected_candidate = None

    if resolution.selected_candidate is not None:
        selected_candidate = RxNormConceptResponse(
            **resolution.selected_candidate.to_dict()
        )

    return DrugFilterResolutionResponse(
        requested_drug=resolution.requested_drug,
        status=resolution.status,
        retrieval_drug=resolution.retrieval_drug,
        corpus_matches=resolution.corpus_matches,
        selected_candidate=selected_candidate,
    )


def serialize_drug_mention_detection(
    detection: DrugMentionDetection,
) -> DrugMentionDetectionResponse:
    selected_candidate = None

    if detection.selected_candidate is not None:
        selected_candidate = RxNormConceptResponse(
            **detection.selected_candidate.to_dict()
        )

    return DrugMentionDetectionResponse(
        status=detection.status,
        detected_mention=detection.detected_mention,
        retrieval_drug=detection.retrieval_drug,
        corpus_matches=detection.corpus_matches,
        selected_candidate=selected_candidate,
    )


def serialize_retrieval_family_plan(
    plan: RetrievalFamilyPlan,
) -> RetrievalFamilyPlanResponse:
    return RetrievalFamilyPlanResponse(
        status=plan.status,
        intent=plan.intent,
        planned_family=plan.planned_family,
        candidate_families=plan.candidate_families,
        matches=[
            RetrievalFamilySignalMatchResponse(
                family=match.family,
                intent=match.intent,
                score=match.score,
                matched_signals=match.matched_signals,
            )
            for match in plan.matches
        ],
    )


def serialize_source_route_plan(
    plan: SourceRoutePlan,
) -> SourceRoutePlanResponse:
    return SourceRoutePlanResponse(
        status=plan.status,
        selected_source=plan.selected_source,
        intent=plan.intent,
        candidate_sources=plan.candidate_sources,
        matches=[
            SourceRouteSignalMatchResponse(
                source=match.source,
                intent=match.intent,
                score=match.score,
                matched_signals=match.matched_signals,
            )
            for match in plan.matches
        ],
    )


def serialize_rxnorm_identity_evidence(
    evidence: RxNormIdentityEvidence,
) -> RxNormIdentityEvidenceResponse:
    selected_candidate = None

    if evidence.selected_candidate is not None:
        selected_candidate = RxNormConceptResponse(
            **evidence.selected_candidate.to_dict()
        )

    return RxNormIdentityEvidenceResponse(
        evidence_id=evidence.evidence_id,
        term=evidence.term,
        resolution_status=evidence.resolution_status,
        selected_candidate=selected_candidate,
        related_ingredients=[
            RxNormConceptResponse(**ingredient.to_dict())
            for ingredient in evidence.related_ingredients
        ],
        related_brands=[
            RxNormConceptResponse(**brand.to_dict())
            for brand in evidence.related_brands
        ],
        summary=evidence.summary,
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
            "GET /drugs",
            "GET /families",
            "GET /corpus/stats",
            "GET /rxnorm/version",
            "POST /normalize/drug",
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
# Corpus metadata endpoints
# ---------------------------------------------------------------------

@app.get(
    "/drugs",
    response_model=DrugListResponse,
    tags=["Corpus"],
)
def drugs() -> DrugListResponse:
    try:
        rows = list_drug_summaries()

        drug_items = [
            DrugSummaryResponse(**row)
            for row in rows
        ]

        return DrugListResponse(
            count=len(drug_items),
            drugs=drug_items,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.get(
    "/families",
    response_model=RetrievalFamilyListResponse,
    tags=["Corpus"],
)
def families() -> RetrievalFamilyListResponse:
    try:
        rows = list_retrieval_family_summaries()

        family_items = [
            RetrievalFamilySummaryResponse(**row)
            for row in rows
        ]

        return RetrievalFamilyListResponse(
            count=len(family_items),
            families=family_items,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.get(
    "/corpus/stats",
    response_model=CorpusStatsResponse,
    response_model_exclude_none=True,
    tags=["Corpus"],
)
def corpus_stats() -> CorpusStatsResponse:
    try:
        stats = collect_corpus_stats()
        return CorpusStatsResponse(**stats)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

# ---------------------------------------------------------------------
# RxNorm normalization endpoints
# ---------------------------------------------------------------------

@app.get(
    "/rxnorm/version",
    response_model=RxNormVersionResponse,
    response_model_exclude_none=True,
    tags=["RxNorm"],
)
def rxnorm_version() -> RxNormVersionResponse:
    try:
        with RxNormClient() as client:
            version_payload = client.get_version()

        return RxNormVersionResponse(**version_payload)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post(
    "/normalize/drug",
    response_model=DrugNormalizationResponse,
    response_model_exclude_none=True,
    tags=["RxNorm"],
)
def normalize_drug(
    request: DrugNormalizationRequest,
) -> DrugNormalizationResponse:
    try:
        resolution = resolve_drug_term(request.term)
        return DrugNormalizationResponse(**resolution.to_dict())

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
        
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
        workflow_result = answer_query_with_drug_resolution(
            query=request.query,
            requested_drug=request.drug,
            retrieval_family=request.family,
            top_k=request.top_k,
        )

        generated = workflow_result.generated
        drug_resolution = workflow_result.drug_resolution
        drug_mention_detection = workflow_result.drug_mention_detection
        family_plan = workflow_result.family_plan
        source_plan = workflow_result.source_plan
        identity_evidence = workflow_result.identity_evidence

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

        drug_resolution_response = serialize_drug_filter_resolution(
            drug_resolution
        )

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
                drug_resolution=drug_resolution_response,
                drug_mention_detection=serialize_drug_mention_detection(
                    drug_mention_detection
                ),
                family_plan=(
                    serialize_retrieval_family_plan(family_plan)
                    if family_plan is not None
                    else None
                ),
                source_plan=(
                    serialize_source_route_plan(source_plan)
                    if source_plan is not None
                    else None
                ),
            )

        identity_evidence_response=None

        if request.include_evidence and identity_evidence is not None:
            identity_evidence_response = [
                serialize_rxnorm_identity_evidence(item)
                for item in identity_evidence
            ]

        request_log_id: str | None = None

        try:
            request_log_id = log_qa_interaction(
                query=request.query,
                drug=workflow_result.retrieval_drug,
                family=workflow_result.retrieval_family,
                top_k=request.top_k,
                include_evidence=request.include_evidence,
                include_diagnostics=request.include_diagnostics,
                generated=generated,
                api_latency_ms=api_latency_ms,
                requested_drug=request.drug,
                resolved_drug=workflow_result.retrieval_drug,
                drug_resolution_status=drug_resolution.status,
                detected_drug_mention=drug_mention_detection.detected_mention,
                drug_mention_detection_status=drug_mention_detection.status,
                requested_family=request.family,
                family_plan_status=(
                    family_plan.status
                    if family_plan is not None
                    else None
                ),
                family_plan_intent=(
                    family_plan.intent
                    if family_plan is not None
                    else None
                ),
                planned_source=(
                    source_plan.selected_source
                    if source_plan is not None
                    else None
                ),
                source_plan_status=(
                    source_plan.status
                    if source_plan is not None
                    else None
                ),
                source_plan_intent=(
                    source_plan.intent
                    if source_plan is not None
                    else None
                ),
            )
        except Exception as log_exc:
            logger.exception(
                "QA observability logging failed: %s",
                log_exc,
            )

        return AnswerAPIResponse(
            query=request.query,
            drug=request.drug,
            resolved_drug=workflow_result.retrieval_drug,
            family=request.family,
            planned_family=workflow_result.retrieval_family,
            planned_source=(
                source_plan.selected_source
                if source_plan is not None
                else None
            ),
            request_log_id=request_log_id,
            result=result,
            evidence=evidence_response,
            identity_evidence=identity_evidence_response,
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
    response_model_exclude_none=True,
    tags=["Retrieval"],
)
def retrieval_debug(
    request: RetrievalDebugRequest,
) -> RetrievalDebugResponse:
    try:
        workflow_result = build_debug_evidence_pack_with_drug_resolution(
            query=request.query,
            requested_drug=request.drug,
            retrieval_family=request.family,
            top_k=request.top_k,
        )

        evidence_pack = workflow_result.evidence_pack
        drug_resolution = workflow_result.drug_resolution
        drug_mention_detection = workflow_result.drug_mention_detection
        family_plan = workflow_result.family_plan
        source_plan = workflow_result.source_plan

        evidence = [
            serialize_evidence_item(item)
            for item in evidence_pack.evidence_items
        ]

        return RetrievalDebugResponse(
            query=request.query,
            drug=request.drug,
            resolved_drug=workflow_result.retrieval_drug,
            family=request.family,
            planned_family=workflow_result.retrieval_family,
            evidence_count=len(evidence),
            evidence=evidence,
            drug_resolution=serialize_drug_filter_resolution(
                drug_resolution
            ),
            drug_mention_detection=serialize_drug_mention_detection(
                drug_mention_detection
            ),
            family_plan=(
                serialize_retrieval_family_plan(family_plan)
                if family_plan is not None
                else None
            ),
            planned_source=(
                source_plan.selected_source
                if source_plan is not None
                else None
            ),
            source_plan=(
                serialize_source_route_plan(source_plan)
                if source_plan is not None
                else None
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc