from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

from medlabeliq.db.connection import get_connection
from medlabeliq.rxnorm.client import RxNormClient
from medlabeliq.rxnorm.models import (
    CandidateResolution,
    DrugTermResolution,
    RxNormConcept,
)


INGREDIENT_TERM_TYPES = {"IN", "PIN"}


def normalize_match_text(value: str | None) -> str:
    if not value:
        return ""

    cleaned = value.casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def list_indexed_corpus_drugs() -> list[str]:
    """
    Return current corpus drug concepts from PostgreSQL.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT concept_name
                FROM label_document
                ORDER BY concept_name ASC;
                """
            )

            rows = cur.fetchall()

    return [
        str(row["concept_name"])
        for row in rows
        if row.get("concept_name")
    ]


def concept_from_properties(
    *,
    properties: dict[str, Any],
    match_method: str,
    source: str | None = None,
    score: float | None = None,
    rank: int | None = None,
) -> RxNormConcept | None:
    rxcui = properties.get("rxcui")
    if not rxcui:
        return None

    return RxNormConcept(
        rxcui=str(rxcui),
        name=(
            str(properties["name"])
            if properties.get("name") is not None
            else None
        ),
        synonym=(
            str(properties["synonym"])
            if properties.get("synonym") is not None
            else None
        ),
        tty=(
            str(properties["tty"])
            if properties.get("tty") is not None
            else None
        ),
        source=source,
        score=score,
        rank=rank,
        match_method=match_method,
    )


def concept_from_related_payload(
    payload: dict[str, Any],
) -> RxNormConcept | None:
    rxcui = payload.get("rxcui")
    if not rxcui:
        return None

    return RxNormConcept(
        rxcui=str(rxcui),
        name=(
            str(payload["name"])
            if payload.get("name") is not None
            else None
        ),
        synonym=(
            str(payload["synonym"])
            if payload.get("synonym") is not None
            else None
        ),
        tty=(
            str(payload["tty"])
            if payload.get("tty") is not None
            else None
        ),
    )


def unique_concepts_by_rxcui(
    concepts: list[RxNormConcept],
) -> list[RxNormConcept]:
    deduped: OrderedDict[str, RxNormConcept] = OrderedDict()

    for concept in concepts:
        deduped.setdefault(concept.rxcui, concept)

    return list(deduped.values())


def corpus_matches_for_concepts(
    *,
    concepts: list[RxNormConcept],
    indexed_drugs: list[str],
) -> list[str]:
    normalized_drug_lookup = {
        normalize_match_text(drug): drug
        for drug in indexed_drugs
    }

    matches: OrderedDict[str, None] = OrderedDict()

    for concept in concepts:
        for candidate_text in [concept.name, concept.synonym]:
            normalized = normalize_match_text(candidate_text)

            if normalized in normalized_drug_lookup:
                matches[normalized_drug_lookup[normalized]] = None

    return list(matches.keys())


def enrich_candidate_resolution(
    *,
    client: RxNormClient,
    candidate: RxNormConcept,
    indexed_drugs: list[str],
) -> CandidateResolution:
    ingredient_concepts: list[RxNormConcept] = []

    if candidate.tty in INGREDIENT_TERM_TYPES:
        ingredient_concepts.append(candidate)

    related_payloads = client.get_related_by_type(
        candidate.rxcui,
        term_types=["IN", "PIN"],
    )

    for payload in related_payloads:
        related_concept = concept_from_related_payload(payload)

        if related_concept is None:
            continue

        ingredient_concepts.append(related_concept)

    ingredient_concepts = unique_concepts_by_rxcui(
        ingredient_concepts
    )

    corpus_matches = corpus_matches_for_concepts(
        concepts=[
            candidate,
            *ingredient_concepts,
        ],
        indexed_drugs=indexed_drugs,
    )

    return CandidateResolution(
        candidate=candidate,
        related_ingredients=ingredient_concepts,
        corpus_matches=corpus_matches,
    )


def build_exact_or_normalized_candidates(
    *,
    client: RxNormClient,
    term: str,
) -> list[RxNormConcept]:
    rxcuis = client.find_rxcuis_by_string(
        term,
        search=2,
    )

    candidates: list[RxNormConcept] = []

    for rxcui in rxcuis:
        properties = client.get_concept_properties(rxcui)

        if properties is None:
            continue

        concept = concept_from_properties(
            properties=properties,
            match_method="exact_or_normalized",
        )

        if concept is not None:
            candidates.append(concept)

    return unique_concepts_by_rxcui(candidates)


def build_approximate_candidates(
    *,
    client: RxNormClient,
    term: str,
) -> list[RxNormConcept]:
    raw_candidates = client.get_approximate_matches(term)

    candidates: list[RxNormConcept] = []
    seen_rxcuis: set[str] = set()

    for raw_candidate in raw_candidates:
        raw_rxcui = raw_candidate.get("rxcui")
        if not raw_rxcui:
            continue

        rxcui = str(raw_rxcui)

        if rxcui in seen_rxcuis:
            continue

        seen_rxcuis.add(rxcui)

        properties = client.get_concept_properties(rxcui)
        if properties is None:
            continue

        raw_score = raw_candidate.get("score")
        raw_rank = raw_candidate.get("rank")

        try:
            score = (
                float(raw_score)
                if raw_score is not None
                else None
            )
        except (TypeError, ValueError):
            score = None

        try:
            rank = (
                int(raw_rank)
                if raw_rank is not None
                else None
            )
        except (TypeError, ValueError):
            rank = None

        concept = concept_from_properties(
            properties=properties,
            match_method="approximate",
            source=(
                str(raw_candidate["source"])
                if raw_candidate.get("source") is not None
                else None
            ),
            score=score,
            rank=rank,
        )

        if concept is not None:
            candidates.append(concept)

    return unique_concepts_by_rxcui(candidates)


def choose_selected_candidate(
    candidate_resolutions: list[CandidateResolution],
    corpus_concept: str,
) -> RxNormConcept | None:
    for resolution in candidate_resolutions:
        if corpus_concept in resolution.corpus_matches:
            return resolution.candidate

    return None


def resolve_drug_term(
    term: str,
) -> DrugTermResolution:
    """
    Resolve a drug mention against RxNorm and map it to an indexed
    MedLabelIQ corpus concept when possible.
    """
    cleaned_term = term.strip()

    indexed_drugs = list_indexed_corpus_drugs()

    with RxNormClient() as client:
        candidates = build_exact_or_normalized_candidates(
            client=client,
            term=cleaned_term,
        )

        if not candidates:
            candidates = build_approximate_candidates(
                client=client,
                term=cleaned_term,
            )

        if not candidates:
            return DrugTermResolution(
                input_term=cleaned_term,
                status="no_rxnorm_match",
                corpus_concept=None,
                corpus_matches=[],
                selected_candidate=None,
                candidates=[],
            )

        candidate_resolutions = [
            enrich_candidate_resolution(
                client=client,
                candidate=candidate,
                indexed_drugs=indexed_drugs,
            )
            for candidate in candidates
        ]

    all_corpus_matches: OrderedDict[str, None] = OrderedDict()

    for resolution in candidate_resolutions:
        for match in resolution.corpus_matches:
            all_corpus_matches[match] = None

    corpus_matches = list(all_corpus_matches.keys())

    if not corpus_matches:
        return DrugTermResolution(
            input_term=cleaned_term,
            status="rxnorm_match_no_corpus_match",
            corpus_concept=None,
            corpus_matches=[],
            selected_candidate=None,
            candidates=candidate_resolutions,
        )

    if len(corpus_matches) > 1:
        return DrugTermResolution(
            input_term=cleaned_term,
            status="ambiguous",
            corpus_concept=None,
            corpus_matches=corpus_matches,
            selected_candidate=None,
            candidates=candidate_resolutions,
        )

    corpus_concept = corpus_matches[0]

    return DrugTermResolution(
        input_term=cleaned_term,
        status="resolved",
        corpus_concept=corpus_concept,
        corpus_matches=corpus_matches,
        selected_candidate=choose_selected_candidate(
            candidate_resolutions,
            corpus_concept=corpus_concept,
        ),
        candidates=candidate_resolutions,
    )