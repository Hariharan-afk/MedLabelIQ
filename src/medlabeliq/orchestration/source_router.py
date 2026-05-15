from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from medlabeliq.orchestration.drug_filter_resolution import (
    DrugFilterResolution,
)
from medlabeliq.orchestration.drug_mention_detection import (
    DrugMentionDetection,
)
from medlabeliq.orchestration.retrieval_family_planner import (
    RetrievalFamilyPlan,
)


SourceRouteStatus = Literal[
    "routed_rxnorm_identity",
    "routed_dailymed_label",
    "ambiguous_mixed_source",
    "fallback_dailymed_label",
]

SourceName = Literal[
    "rxnorm_identity",
    "dailymed_label",
    "multi_source_composed",
]


@dataclass(frozen=True)
class SourceRouteSignalMatch:
    source: SourceName
    intent: str
    score: int
    matched_signals: list[str]


@dataclass(frozen=True)
class SourceRoutePlan:
    query: str
    status: SourceRouteStatus
    selected_source: SourceName
    intent: str | None
    candidate_sources: list[SourceName]
    matches: list[SourceRouteSignalMatch]


@dataclass(frozen=True)
class SourceSignalSpec:
    source: SourceName
    intent: str
    signal: str
    weight: int


IDENTITY_SIGNAL_SPECS: list[SourceSignalSpec] = [
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="brand_generic_equivalence",
        signal="same as",
        weight=6,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="brand_generic_equivalence",
        signal="equivalent to",
        weight=6,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="brand_generic_equivalence",
        signal="another name for",
        weight=6,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="generic_name_lookup",
        signal="generic name",
        weight=6,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="brand_name_lookup",
        signal="brand name",
        weight=6,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="brand_name_lookup",
        signal="brand of",
        weight=5,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="ingredient_identity",
        signal="active ingredient",
        weight=6,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="ingredient_identity",
        signal="ingredient in",
        weight=5,
    ),
    SourceSignalSpec(
        source="rxnorm_identity",
        intent="drug_identity_definition",
        signal="what is",
        weight=2,
    ),
]

MIN_RXNORM_IDENTITY_SCORE = 5


def normalize_text(value: str) -> str:
    cleaned = value.casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def phrase_occurs(
    *,
    phrase: str,
    normalized_query: str,
) -> bool:
    normalized_phrase = normalize_text(phrase)

    if not normalized_phrase:
        return False

    padded_query = f" {normalized_query} "
    padded_phrase = f" {normalized_phrase} "

    return padded_phrase in padded_query


def family_plan_indicates_label_evidence(
    family_plan: RetrievalFamilyPlan | None,
    *,
    requested_family: str | None,
) -> bool:
    if requested_family is not None and requested_family.strip():
        return True

    if family_plan is None:
        return False

    return family_plan.status in {
        "routed_single_family",
        "candidate_family_group_unfiltered",
    }


def build_identity_matches(
    query: str,
    *,
    family_plan: RetrievalFamilyPlan | None,
) -> list[SourceRouteSignalMatch]:
    normalized_query = normalize_text(query)

    grouped_signals: dict[str, list[str]] = {}
    grouped_scores: dict[str, int] = {}

    for spec in IDENTITY_SIGNAL_SPECS:
        # "what is" is only useful as an identity clue when the family planner
        # did not already detect a label-evidence intent like "used for".
        if (
            spec.signal == "what is"
            and family_plan is not None
            and family_plan.status
            in {
                "routed_single_family",
                "candidate_family_group_unfiltered",
            }
        ):
            continue

        if not phrase_occurs(
            phrase=spec.signal,
            normalized_query=normalized_query,
        ):
            continue

        grouped_signals.setdefault(spec.intent, []).append(spec.signal)
        grouped_scores[spec.intent] = (
            grouped_scores.get(spec.intent, 0) + spec.weight
        )

    matches = [
        SourceRouteSignalMatch(
            source="rxnorm_identity",
            intent=intent,
            score=grouped_scores[intent],
            matched_signals=grouped_signals[intent],
        )
        for intent in grouped_scores
    ]

    return sorted(
        matches,
        key=lambda item: (-item.score, item.intent),
    )


def strengthen_short_identity_question(
    *,
    query: str,
    family_plan: RetrievalFamilyPlan | None,
    drug_resolution: DrugFilterResolution | None,
    drug_mention_detection: DrugMentionDetection | None,
    matches: list[SourceRouteSignalMatch],
) -> list[SourceRouteSignalMatch]:
    """
    Boost very short 'What is X?' questions toward RxNorm identity routing
    when a drug filter or detected drug mention is present.
    """
    if family_plan is not None and family_plan.status in {
        "routed_single_family",
        "candidate_family_group_unfiltered",
    }:
        return matches

    normalized_query = normalize_text(query)

    if not normalized_query.startswith("what is "):
        return matches

    has_drug_context = False

    if (
        drug_resolution is not None
        and drug_resolution.retrieval_drug is not None
    ):
        has_drug_context = True

    if (
        drug_mention_detection is not None
        and drug_mention_detection.retrieval_drug is not None
    ):
        has_drug_context = True

    if not has_drug_context:
        return matches

    return [
        *matches,
        SourceRouteSignalMatch(
            source="rxnorm_identity",
            intent="drug_identity_definition",
            score=5,
            matched_signals=["what is + resolved drug context"],
        ),
    ]


def plan_source_route(
    query: str,
    *,
    requested_family: str | None,
    family_plan: RetrievalFamilyPlan | None,
    drug_resolution: DrugFilterResolution | None,
    drug_mention_detection: DrugMentionDetection | None,
) -> SourceRoutePlan:
    """
    Plan which knowledge source should ideally handle the query.

    Current routes:
    - dailymed_label: clinical label evidence and grounded RAG
    - rxnorm_identity: drug identity, brand/generic, and ingredient relations

    Step 30A only records the routing plan. Step 30B will execute the
    RxNorm identity branch.
    """
    identity_matches = build_identity_matches(
        query,
        family_plan=family_plan,
    )

    identity_matches = strengthen_short_identity_question(
        query=query,
        family_plan=family_plan,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_mention_detection,
        matches=identity_matches,
    )

    strongest_identity_score = (
        max(match.score for match in identity_matches)
        if identity_matches
        else 0
    )

    label_intent_detected = family_plan_indicates_label_evidence(
        family_plan,
        requested_family=requested_family,
    )

    if (
        strongest_identity_score >= MIN_RXNORM_IDENTITY_SCORE
        and label_intent_detected
    ):
        return SourceRoutePlan(
            query=query,
            status="ambiguous_mixed_source",
            selected_source="multi_source_composed",
            intent=None,
            candidate_sources=[
                "rxnorm_identity",
                "dailymed_label",
            ],
            matches=identity_matches,
        )

    if strongest_identity_score >= MIN_RXNORM_IDENTITY_SCORE:
        top_identity = max(
            identity_matches,
            key=lambda item: item.score,
        )

        return SourceRoutePlan(
            query=query,
            status="routed_rxnorm_identity",
            selected_source="rxnorm_identity",
            intent=top_identity.intent,
            candidate_sources=["rxnorm_identity"],
            matches=identity_matches,
        )

    if label_intent_detected:
        return SourceRoutePlan(
            query=query,
            status="routed_dailymed_label",
            selected_source="dailymed_label",
            intent=(
                family_plan.intent
                if family_plan is not None
                else None
            ),
            candidate_sources=["dailymed_label"],
            matches=[],
        )

    return SourceRoutePlan(
        query=query,
        status="fallback_dailymed_label",
        selected_source="dailymed_label",
        intent=None,
        candidate_sources=["dailymed_label"],
        matches=[],
    )