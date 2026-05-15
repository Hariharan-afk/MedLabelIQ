from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------

class AnswerRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="Medication-label question to answer.",
        examples=["Can metformin cause dangerous acid buildup in the blood?"],
    )

    drug: str | None = Field(
        default=None,
        description=(
            "Optional drug filter. May be an indexed corpus concept, "
            "brand name, or noisy medication mention resolvable through RxNorm."
        ),
        examples=["Glucophage"],
    )

    family: str | None = Field(
        default=None,
        description=(
            "Optional retrieval-family filter, e.g. "
            "'warnings_and_precautions'."
        ),
        examples=["warnings_and_precautions"],
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional final evidence-pack size override.",
        examples=[5],
    )

    include_evidence: bool = Field(
        default=True,
        description="Whether to include retrieved evidence snippets in the API response.",
    )

    include_diagnostics: bool = Field(
        default=True,
        description=(
            "Whether to include verifier, guardrail, and proposal diagnostics "
            "in the API response."
        ),
    )


class RetrievalDebugRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="Query used for retrieval-only debugging.",
    )

    drug: str | None = Field(
        default=None,
        description=(
            "Optional drug filter. May be an indexed corpus concept, "
            "brand name, or noisy medication mention resolvable through RxNorm."
        ),
    )

    family: str | None = Field(
        default=None,
        description="Optional retrieval-family filter.",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional final evidence-pack size override.",
    )


# ---------------------------------------------------------------------
# Common response models
# ---------------------------------------------------------------------

class GroundedAnswerResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[str]
    evidence_summary: str
    safety_note: str


class EvidenceItemResponse(BaseModel):
    evidence_id: str
    chunk_id: str
    section_id: str

    drug: str
    retrieval_family: str
    canonical_section_name: str | None
    nearest_canonical_section_name: str | None

    heading: str
    set_id: str
    version_number: int
    source_label: str

    chunk_text: str

    hybrid_score: float
    lexical_rank: int | None
    dense_rank: int | None


class VerificationResponse(BaseModel):
    verdict: Literal["supported", "insufficient", "refuted"]
    rationale: str
    cited_evidence_used: list[str]


class DiagnosticsResponse(BaseModel):
    evidence_count: int

    proposed_status: Literal["answered", "insufficient_evidence"] | None
    verification_enabled: bool
    verification: VerificationResponse | None
    verification_overrode_answer: bool

    guardrail_triggered: bool
    guardrail_reason: str | None

    drug_resolution: DrugFilterResolutionResponse | None = None
    drug_mention_detection: DrugMentionDetectionResponse | None = None
    family_plan: RetrievalFamilyPlanResponse | None = None


class AnswerAPIResponse(BaseModel):
    query: str
    drug: str | None
    resolved_drug: str | None = None
    family: str | None
    planned_family: str | None = None
    request_log_id: str | None = None

    result: GroundedAnswerResponse
    evidence: list[EvidenceItemResponse] | None
    diagnostics: DiagnosticsResponse | None


class RetrievalDebugResponse(BaseModel):
    query: str
    drug: str | None
    resolved_drug: str | None = None
    family: str | None
    planned_family: str | None = None
    evidence_count: int
    evidence: list[EvidenceItemResponse]
    drug_resolution: DrugFilterResolutionResponse | None = None
    drug_mention_detection: DrugMentionDetectionResponse | None = None
    family_plan: RetrievalFamilyPlanResponse | None = None

# ---------------------------------------------------------------------
# Corpus metadata response models
# ---------------------------------------------------------------------

class DrugSummaryResponse(BaseModel):
    concept_name: str
    label_count: int
    label_version_count: int
    section_count: int
    chunk_count: int


class DrugListResponse(BaseModel):
    count: int
    drugs: list[DrugSummaryResponse]


class RetrievalFamilySummaryResponse(BaseModel):
    retrieval_family: str
    section_count: int
    chunk_count: int
    drug_count: int


class RetrievalFamilyListResponse(BaseModel):
    count: int
    families: list[RetrievalFamilySummaryResponse]


class CorpusBuildMetadataResponse(BaseModel):
    build_id: str
    built_at: datetime
    build_source: str
    seed_file_path: str

    drug_count: int
    label_document_count: int
    label_version_count: int
    product_count: int
    ingredient_count: int
    section_count: int
    retrievable_section_count: int
    chunk_count: int
    retrieval_family_count: int

    qdrant_collection: str
    qdrant_point_count: int
    embedding_model_name: str


class CorpusStatsResponse(BaseModel):
    drug_count: int
    label_document_count: int
    label_version_count: int
    product_count: int
    ingredient_count: int
    section_count: int
    retrievable_section_count: int
    chunk_count: int
    retrieval_family_count: int

    qdrant_collection: str
    qdrant_point_count: int | None
    embedding_model_name: str

    latest_build: CorpusBuildMetadataResponse | None

# ---------------------------------------------------------------------
# RxNorm normalization response models
# ---------------------------------------------------------------------

class DrugNormalizationRequest(BaseModel):
    term: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Drug term, brand name, or noisy medication mention to normalize.",
        examples=["Glucophage"],
    )


class RxNormConceptResponse(BaseModel):
    rxcui: str
    name: str | None
    synonym: str | None
    tty: str | None

    source: str | None = None
    score: float | None = None
    rank: int | None = None
    match_method: str | None = None


class RxNormCandidateResolutionResponse(BaseModel):
    candidate: RxNormConceptResponse
    related_ingredients: list[RxNormConceptResponse]
    corpus_matches: list[str]


class DrugNormalizationResponse(BaseModel):
    input_term: str
    status: Literal[
        "resolved",
        "ambiguous",
        "rxnorm_match_no_corpus_match",
        "no_rxnorm_match",
    ]

    corpus_concept: str | None
    corpus_matches: list[str]

    selected_candidate: RxNormConceptResponse | None
    candidates: list[RxNormCandidateResolutionResponse]


class RxNormVersionResponse(BaseModel):
    version: str | None
    api_version: str | None


class DrugFilterResolutionResponse(BaseModel):
    requested_drug: str | None
    status: Literal[
        "not_requested",
        "direct_corpus_match",
        "resolved",
        "ambiguous",
        "rxnorm_match_no_corpus_match",
        "no_rxnorm_match",
    ]
    retrieval_drug: str | None
    corpus_matches: list[str]
    selected_candidate: RxNormConceptResponse | None


class DrugMentionDetectionResponse(BaseModel):
    status: Literal[
        "not_attempted_explicit_filter_present",
        "direct_corpus_mention",
        "rxnorm_resolved_query_mention",
        "ambiguous",
        "no_mention_detected",
    ]
    detected_mention: str | None
    retrieval_drug: str | None
    corpus_matches: list[str]
    selected_candidate: RxNormConceptResponse | None


class RetrievalFamilySignalMatchResponse(BaseModel):
    family: str
    intent: str
    score: int
    matched_signals: list[str]


class RetrievalFamilyPlanResponse(BaseModel):
    status: Literal[
        "not_attempted_explicit_family_present",
        "routed_single_family",
        "candidate_family_group_unfiltered",
        "ambiguous",
        "no_route_detected",
    ]
    intent: str | None
    planned_family: str | None
    candidate_families: list[str]
    matches: list[RetrievalFamilySignalMatchResponse]
    
# ---------------------------------------------------------------------
# Health response models
# ---------------------------------------------------------------------

class HealthComponentResponse(BaseModel):
    status: Literal["ok", "error", "not_configured"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    postgres: HealthComponentResponse
    qdrant: HealthComponentResponse
    llm: HealthComponentResponse


class RootResponse(BaseModel):
    service: str
    status: str
    docs: str
    endpoints: list[str]