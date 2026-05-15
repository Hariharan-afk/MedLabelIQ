from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GroundedAnswer(BaseModel):
    """
    Validated answer format returned by the LLM.
    """

    status: Literal["answered", "insufficient_evidence"]

    answer: str = Field(
        ...,
        description="Final user-facing answer grounded only in the evidence.",
    )

    citations: list[str] = Field(
        default_factory=list,
        description="Evidence IDs used, such as ['E1', 'E3'].",
    )

    evidence_summary: str = Field(
        ...,
        description="Brief explanation of which evidence supports the answer.",
    )

    safety_note: str = Field(
        ...,
        description="Application-controlled safety note. The model should return an empty string.",
    )

    @model_validator(mode="after")
    def validate_citations_for_answered_status(self):
        if self.status == "answered" and not self.citations:
            raise ValueError(
                "Answered responses must include at least one evidence citation."
            )

        if self.status == "insufficient_evidence" and self.citations:
            raise ValueError(
                "Insufficient-evidence responses should not include evidence citations."
            )

        return self