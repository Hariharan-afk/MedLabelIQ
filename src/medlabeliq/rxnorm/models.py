from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


DrugResolutionStatus = Literal[
    "resolved",
    "ambiguous",
    "rxnorm_match_no_corpus_match",
    "no_rxnorm_match",
]


@dataclass(frozen=True)
class RxNormConcept:
    rxcui: str
    name: str | None
    synonym: str | None
    tty: str | None

    source: str | None = None
    score: float | None = None
    rank: int | None = None
    match_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rxcui": self.rxcui,
            "name": self.name,
            "synonym": self.synonym,
            "tty": self.tty,
            "source": self.source,
            "score": self.score,
            "rank": self.rank,
            "match_method": self.match_method,
        }


@dataclass(frozen=True)
class CandidateResolution:
    candidate: RxNormConcept
    related_ingredients: list[RxNormConcept]
    corpus_matches: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "related_ingredients": [
                ingredient.to_dict()
                for ingredient in self.related_ingredients
            ],
            "corpus_matches": self.corpus_matches,
        }


@dataclass(frozen=True)
class DrugTermResolution:
    input_term: str
    status: DrugResolutionStatus

    corpus_concept: str | None
    corpus_matches: list[str]

    selected_candidate: RxNormConcept | None
    candidates: list[CandidateResolution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_term": self.input_term,
            "status": self.status,
            "corpus_concept": self.corpus_concept,
            "corpus_matches": self.corpus_matches,
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
        }