from __future__ import annotations

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
        description="Optional normalized drug concept filter, e.g. 'metformin'.",
        examples=["metformin"],
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
        description="Optional normalized drug concept filter.",
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


class AnswerAPIResponse(BaseModel):
    query: str
    drug: str | None
    family: str | None
    request_log_id: str | None = None

    result: GroundedAnswerResponse
    evidence: list[EvidenceItemResponse] | None
    diagnostics: DiagnosticsResponse | None


class RetrievalDebugResponse(BaseModel):
    query: str
    drug: str | None
    family: str | None
    evidence_count: int
    evidence: list[EvidenceItemResponse]


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