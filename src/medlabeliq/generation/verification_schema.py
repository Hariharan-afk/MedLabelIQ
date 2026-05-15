from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnswerVerification(BaseModel):
    """
    Verifies whether a generated answer is directly supported by its cited evidence.
    """

    verdict: Literal["supported", "insufficient", "refuted"] = Field(
        ...,
        description=(
            "supported = cited evidence directly establishes the answer; "
            "insufficient = cited evidence is related but does not establish it; "
            "refuted = cited evidence contradicts the answer."
        ),
    )

    rationale: str = Field(
        ...,
        description="Brief grounded explanation for the verdict.",
    )

    cited_evidence_used: list[str] = Field(
        default_factory=list,
        description="Evidence IDs the verifier actually considered, such as ['E1'].",
    )