from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from medlabeliq.rxnorm.models import DrugTermResolution, RxNormConcept
from medlabeliq.rxnorm.resolver import (
    list_indexed_corpus_drugs,
    normalize_match_text,
    resolve_drug_term,
)


DrugFilterResolutionStatus = Literal[
    "not_requested",
    "direct_corpus_match",
    "resolved",
    "ambiguous",
    "rxnorm_match_no_corpus_match",
    "no_rxnorm_match",
]


@dataclass(frozen=True)
class DrugFilterResolution:
    requested_drug: str | None
    status: DrugFilterResolutionStatus
    retrieval_drug: str | None
    corpus_matches: list[str]
    selected_candidate: RxNormConcept | None
    rxnorm_resolution: DrugTermResolution | None = None

    @property
    def can_retrieve(self) -> bool:
        return self.status in {
            "not_requested",
            "direct_corpus_match",
            "resolved",
        }


def find_direct_corpus_match(
    requested_drug: str,
    indexed_drugs: list[str],
) -> str | None:
    normalized_lookup = {
        normalize_match_text(drug): drug
        for drug in indexed_drugs
    }

    return normalized_lookup.get(
        normalize_match_text(requested_drug)
    )


def resolve_optional_drug_filter(
    requested_drug: str | None,
) -> DrugFilterResolution:
    """
    Resolve an optional QA/retrieval drug filter.

    Policy:
    - No drug filter -> continue with corpus-wide retrieval.
    - Direct indexed concept match -> use directly, avoid external RxNorm call.
    - RxNorm resolves to one indexed corpus concept -> use that concept.
    - Ambiguous / no corpus match / no RxNorm match -> fail closed.
    """
    if requested_drug is None or not requested_drug.strip():
        return DrugFilterResolution(
            requested_drug=None,
            status="not_requested",
            retrieval_drug=None,
            corpus_matches=[],
            selected_candidate=None,
            rxnorm_resolution=None,
        )

    cleaned_requested_drug = requested_drug.strip()

    indexed_drugs = list_indexed_corpus_drugs()

    direct_match = find_direct_corpus_match(
        cleaned_requested_drug,
        indexed_drugs,
    )

    if direct_match is not None:
        return DrugFilterResolution(
            requested_drug=cleaned_requested_drug,
            status="direct_corpus_match",
            retrieval_drug=direct_match,
            corpus_matches=[direct_match],
            selected_candidate=None,
            rxnorm_resolution=None,
        )

    rxnorm_resolution = resolve_drug_term(cleaned_requested_drug)

    if rxnorm_resolution.status == "resolved":
        return DrugFilterResolution(
            requested_drug=cleaned_requested_drug,
            status="resolved",
            retrieval_drug=rxnorm_resolution.corpus_concept,
            corpus_matches=rxnorm_resolution.corpus_matches,
            selected_candidate=rxnorm_resolution.selected_candidate,
            rxnorm_resolution=rxnorm_resolution,
        )

    if rxnorm_resolution.status == "ambiguous":
        return DrugFilterResolution(
            requested_drug=cleaned_requested_drug,
            status="ambiguous",
            retrieval_drug=None,
            corpus_matches=rxnorm_resolution.corpus_matches,
            selected_candidate=None,
            rxnorm_resolution=rxnorm_resolution,
        )

    if rxnorm_resolution.status == "rxnorm_match_no_corpus_match":
        return DrugFilterResolution(
            requested_drug=cleaned_requested_drug,
            status="rxnorm_match_no_corpus_match",
            retrieval_drug=None,
            corpus_matches=[],
            selected_candidate=None,
            rxnorm_resolution=rxnorm_resolution,
        )

    return DrugFilterResolution(
        requested_drug=cleaned_requested_drug,
        status="no_rxnorm_match",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=rxnorm_resolution,
    )