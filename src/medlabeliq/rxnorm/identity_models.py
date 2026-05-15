from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from medlabeliq.rxnorm.models import RxNormConcept


IdentityTermResolutionStatus = Literal[
    "resolved",
    "ambiguous",
    "no_rxnorm_match",
]


@dataclass(frozen=True)
class RxNormIdentityTermResolution:
    input_term: str
    status: IdentityTermResolutionStatus
    selected_candidate: RxNormConcept | None
    candidates: list[RxNormConcept]
    related_ingredients: list[RxNormConcept]
    related_brands: list[RxNormConcept]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_term": self.input_term,
            "status": self.status,
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "related_ingredients": [
                ingredient.to_dict()
                for ingredient in self.related_ingredients
            ],
            "related_brands": [
                brand.to_dict()
                for brand in self.related_brands
            ],
        }


@dataclass(frozen=True)
class RxNormIdentityEvidence:
    evidence_id: str
    term: str
    resolution_status: IdentityTermResolutionStatus
    selected_candidate: RxNormConcept | None
    related_ingredients: list[RxNormConcept]
    related_brands: list[RxNormConcept]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "term": self.term,
            "resolution_status": self.resolution_status,
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "related_ingredients": [
                ingredient.to_dict()
                for ingredient in self.related_ingredients
            ],
            "related_brands": [
                brand.to_dict()
                for brand in self.related_brands
            ],
            "summary": self.summary,
        }