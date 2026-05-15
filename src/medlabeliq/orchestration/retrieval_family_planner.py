from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


RetrievalFamilyPlanStatus = Literal[
    "not_attempted_explicit_family_present",
    "routed_single_family",
    "candidate_family_group_unfiltered",
    "ambiguous",
    "no_route_detected",
]


@dataclass(frozen=True)
class RetrievalFamilySignalMatch:
    family: str
    intent: str
    score: int
    matched_signals: list[str]


@dataclass(frozen=True)
class RetrievalFamilyPlan:
    query: str
    status: RetrievalFamilyPlanStatus
    intent: str | None
    planned_family: str | None
    candidate_families: list[str]
    matches: list[RetrievalFamilySignalMatch]

    @property
    def can_filter(self) -> bool:
        return (
            self.status == "routed_single_family"
            and self.planned_family is not None
        )


@dataclass(frozen=True)
class SignalSpec:
    family: str
    intent: str
    signal: str
    weight: int


MIN_SINGLE_FAMILY_SCORE = 2


# =============================================================================
# Conservative family-specific routing rules
# =============================================================================

SINGLE_FAMILY_SIGNAL_SPECS: list[SignalSpec] = [
    # -------------------------------------------------------------------------
    # Indications / intended use
    # -------------------------------------------------------------------------
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="used for",
        weight=4,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="indicated for",
        weight=4,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="approved for",
        weight=4,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="treat",
        weight=3,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="treatment",
        weight=3,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="cure",
        weight=3,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="prevent",
        weight=3,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="prevention",
        weight=3,
    ),
    SignalSpec(
        family="indications_and_usage",
        intent="indication_or_use",
        signal="what is",
        weight=1,
    ),

    # -------------------------------------------------------------------------
    # Drug interactions
    # -------------------------------------------------------------------------
    SignalSpec(
        family="drug_interactions",
        intent="interaction",
        signal="interaction",
        weight=4,
    ),
    SignalSpec(
        family="drug_interactions",
        intent="interaction",
        signal="interact",
        weight=4,
    ),
    SignalSpec(
        family="drug_interactions",
        intent="interaction",
        signal="taken with",
        weight=4,
    ),
    SignalSpec(
        family="drug_interactions",
        intent="interaction",
        signal="take with",
        weight=4,
    ),
    SignalSpec(
        family="drug_interactions",
        intent="interaction",
        signal="together with",
        weight=4,
    ),
    SignalSpec(
        family="drug_interactions",
        intent="interaction",
        signal="combine",
        weight=3,
    ),
    SignalSpec(
        family="drug_interactions",
        intent="interaction",
        signal="combined with",
        weight=4,
    ),

    # -------------------------------------------------------------------------
    # Dosage / administration
    # -------------------------------------------------------------------------
    SignalSpec(
        family="dosage_and_administration",
        intent="dosage",
        signal="dose",
        weight=4,
    ),
    SignalSpec(
        family="dosage_and_administration",
        intent="dosage",
        signal="dosage",
        weight=4,
    ),
    SignalSpec(
        family="dosage_and_administration",
        intent="dosage",
        signal="how much",
        weight=4,
    ),
    SignalSpec(
        family="dosage_and_administration",
        intent="dosage",
        signal="how often",
        weight=4,
    ),
    SignalSpec(
        family="dosage_and_administration",
        intent="dosage",
        signal="once daily",
        weight=3,
    ),
    SignalSpec(
        family="dosage_and_administration",
        intent="dosage",
        signal="twice daily",
        weight=3,
    ),
    SignalSpec(
        family="dosage_and_administration",
        intent="dosage",
        signal="administer",
        weight=3,
    ),

    # -------------------------------------------------------------------------
    # Contraindications
    # -------------------------------------------------------------------------
    SignalSpec(
        family="contraindications",
        intent="contraindication",
        signal="contraindicated",
        weight=5,
    ),
    SignalSpec(
        family="contraindications",
        intent="contraindication",
        signal="should not take",
        weight=5,
    ),
    SignalSpec(
        family="contraindications",
        intent="contraindication",
        signal="do not take",
        weight=5,
    ),
    SignalSpec(
        family="contraindications",
        intent="contraindication",
        signal="who should not take",
        weight=5,
    ),

    # -------------------------------------------------------------------------
    # Adverse reactions / side effects
    # -------------------------------------------------------------------------
    SignalSpec(
        family="adverse_reactions",
        intent="adverse_reaction",
        signal="side effect",
        weight=5,
    ),
    SignalSpec(
        family="adverse_reactions",
        intent="adverse_reaction",
        signal="side effects",
        weight=5,
    ),
    SignalSpec(
        family="adverse_reactions",
        intent="adverse_reaction",
        signal="adverse reaction",
        weight=5,
    ),
    SignalSpec(
        family="adverse_reactions",
        intent="adverse_reaction",
        signal="adverse reactions",
        weight=5,
    ),
    SignalSpec(
        family="adverse_reactions",
        intent="adverse_reaction",
        signal="common reactions",
        weight=4,
    ),

    # -------------------------------------------------------------------------
    # Overdose
    # -------------------------------------------------------------------------
    SignalSpec(
        family="overdosage",
        intent="overdose",
        signal="overdose",
        weight=5,
    ),
    SignalSpec(
        family="overdosage",
        intent="overdose",
        signal="too much",
        weight=4,
    ),
    SignalSpec(
        family="overdosage",
        intent="overdose",
        signal="excess amount",
        weight=4,
    ),

    # -------------------------------------------------------------------------
    # Storage / handling
    # -------------------------------------------------------------------------
    SignalSpec(
        family="how_supplied_storage_and_handling",
        intent="storage",
        signal="store",
        weight=5,
    ),
    SignalSpec(
        family="how_supplied_storage_and_handling",
        intent="storage",
        signal="storage",
        weight=5,
    ),
    SignalSpec(
        family="how_supplied_storage_and_handling",
        intent="storage",
        signal="room temperature",
        weight=4,
    ),
    SignalSpec(
        family="how_supplied_storage_and_handling",
        intent="storage",
        signal="refrigerate",
        weight=4,
    ),
]


# =============================================================================
# Family groups that we detect but do not single-family filter yet
# =============================================================================

SAFETY_GROUP_SIGNALS = [
    "warning",
    "warnings",
    "risk",
    "dangerous",
    "serious",
    "fatal",
    "harm",
    "can cause",
    "could cause",
]

SAFETY_GROUP_FAMILIES = [
    "warnings_and_precautions",
    "warnings",
    "boxed_warning",
]

POPULATION_GROUP_SIGNALS = [
    "pregnant",
    "pregnancy",
    "breastfeeding",
    "nursing",
    "children",
    "child",
    "pediatric",
    "elderly",
    "geriatric",
    "unborn baby",
]

POPULATION_GROUP_FAMILIES = [
    "use_in_specific_populations",
    "warnings_and_precautions",
    "boxed_warning",
]


# =============================================================================
# Helpers
# =============================================================================

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


def build_not_attempted_family_plan(
    query: str,
) -> RetrievalFamilyPlan:
    return RetrievalFamilyPlan(
        query=query,
        status="not_attempted_explicit_family_present",
        intent=None,
        planned_family=None,
        candidate_families=[],
        matches=[],
    )


def score_single_family_matches(
    query: str,
) -> list[RetrievalFamilySignalMatch]:
    normalized_query = normalize_text(query)

    grouped_matches: dict[tuple[str, str], list[str]] = {}
    grouped_scores: dict[tuple[str, str], int] = {}

    for spec in SINGLE_FAMILY_SIGNAL_SPECS:
        if not phrase_occurs(
            phrase=spec.signal,
            normalized_query=normalized_query,
        ):
            continue

        key = (spec.family, spec.intent)

        grouped_matches.setdefault(key, []).append(spec.signal)
        grouped_scores[key] = grouped_scores.get(key, 0) + spec.weight

    results = [
        RetrievalFamilySignalMatch(
            family=family,
            intent=intent,
            score=grouped_scores[(family, intent)],
            matched_signals=grouped_matches[(family, intent)],
        )
        for family, intent in grouped_scores
    ]

    return sorted(
        results,
        key=lambda item: (-item.score, item.family),
    )


def detect_family_group(
    *,
    query: str,
    signals: list[str],
) -> list[str]:
    normalized_query = normalize_text(query)

    return [
        signal
        for signal in signals
        if phrase_occurs(
            phrase=signal,
            normalized_query=normalized_query,
        )
    ]


def detect_safety_group_signals(
    query: str,
) -> list[str]:
    """
    Detect safety-warning intent.

    This extends exact phrase matching with conservative regex patterns for
    common medication-risk questions such as:
    - Can Glucophage cause lactic acidosis?
    - Could metformin cause dangerous acidosis?
    - Does apixaban cause bleeding?
    """
    exact_signals = detect_family_group(
        query=query,
        signals=SAFETY_GROUP_SIGNALS,
    )

    normalized_query = normalize_text(query)

    regex_signals: list[str] = []

    cause_pattern = re.compile(
        r"\b(?:can|could|does|do|may|might|will|would)\s+"
        r"(?:[a-z0-9]+\s+){0,4}"
        r"cause\b"
    )

    if cause_pattern.search(normalized_query):
        regex_signals.append("__modal_drug_cause_pattern__")

    return [
        *exact_signals,
        *regex_signals,
    ]

# =============================================================================
# Planner
# =============================================================================

def plan_retrieval_family(
    query: str,
    *,
    requested_family: str | None = None,
) -> RetrievalFamilyPlan:
    """
    Build a conservative retrieval-family plan.

    Policy:
    - Explicit family provided -> do not auto-plan.
    - Strong single-family intent -> apply one retrieval family filter.
    - Safety/population groups -> detect the intent, but do not restrict retrieval
      to one family yet because clinically relevant evidence can live in multiple
      families.
    - Tied single-family scores -> mark ambiguous and do not filter.
    - No useful signal -> broad retrieval.
    """
    if requested_family is not None and requested_family.strip():
        return build_not_attempted_family_plan(query)

    single_family_matches = score_single_family_matches(query)

    safety_signals = detect_safety_group_signals(query)

    population_signals = detect_family_group(
        query=query,
        signals=POPULATION_GROUP_SIGNALS,
    )

    # -------------------------------------------------------------------------
    # Prefer strong, unambiguous single-family routing.
    # -------------------------------------------------------------------------
    if single_family_matches:
        top_match = single_family_matches[0]

        tied_top_matches = [
            match
            for match in single_family_matches
            if match.score == top_match.score
        ]

        if len(tied_top_matches) > 1:
            return RetrievalFamilyPlan(
                query=query,
                status="ambiguous",
                intent=None,
                planned_family=None,
                candidate_families=[
                    match.family
                    for match in tied_top_matches
                ],
                matches=single_family_matches,
            )

        if top_match.score >= MIN_SINGLE_FAMILY_SCORE:
            return RetrievalFamilyPlan(
                query=query,
                status="routed_single_family",
                intent=top_match.intent,
                planned_family=top_match.family,
                candidate_families=[top_match.family],
                matches=single_family_matches,
            )

    # -------------------------------------------------------------------------
    # Detect multi-family clinical groups, but remain unfiltered for recall.
    # -------------------------------------------------------------------------
    if safety_signals:
        return RetrievalFamilyPlan(
            query=query,
            status="candidate_family_group_unfiltered",
            intent="safety_warning",
            planned_family=None,
            candidate_families=SAFETY_GROUP_FAMILIES,
            matches=[
                RetrievalFamilySignalMatch(
                    family="__safety_family_group__",
                    intent="safety_warning",
                    score=len(safety_signals),
                    matched_signals=safety_signals,
                )
            ],
        )

    if population_signals:
        return RetrievalFamilyPlan(
            query=query,
            status="candidate_family_group_unfiltered",
            intent="population_specific_use",
            planned_family=None,
            candidate_families=POPULATION_GROUP_FAMILIES,
            matches=[
                RetrievalFamilySignalMatch(
                    family="__population_family_group__",
                    intent="population_specific_use",
                    score=len(population_signals),
                    matched_signals=population_signals,
                )
            ],
        )

    return RetrievalFamilyPlan(
        query=query,
        status="no_route_detected",
        intent=None,
        planned_family=None,
        candidate_families=[],
        matches=[],
    )