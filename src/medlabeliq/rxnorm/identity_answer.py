from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from medlabeliq.generation.answer_generator import GeneratedAnswer
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack
from medlabeliq.rxnorm.client import RxNormClient
from medlabeliq.rxnorm.identity_models import (
    RxNormIdentityEvidence,
    RxNormIdentityTermResolution,
)
from medlabeliq.rxnorm.models import RxNormConcept
from medlabeliq.rxnorm.resolver import (
    build_approximate_candidates,
    build_exact_or_normalized_candidates,
    concept_from_related_payload,
    unique_concepts_by_rxcui,
)


RXNORM_IDENTITY_SAFETY_NOTE = (
    "This answer summarizes RxNorm medication identity relationships and is "
    "not a substitute for advice from a qualified clinician or pharmacist."
)

RXNORM_IDENTITY_INSUFFICIENT_ANSWER = (
    "The available RxNorm identity data was not sufficient to answer this "
    "question reliably."
)

RXNORM_IDENTITY_INSUFFICIENT_SUMMARY = (
    "RxNorm did not provide a sufficiently clear medication identity mapping "
    "for the requested claim."
)

INGREDIENT_TERM_TYPES = {"IN", "PIN"}
BRAND_TERM_TYPES = {"BN"}


@dataclass(frozen=True)
class RxNormIdentityAnswerResult:
    generated: GeneratedAnswer
    evidence_items: list[RxNormIdentityEvidence]


# =============================================================================
# Shared helpers
# =============================================================================

def empty_identity_evidence_pack(query: str) -> EvidencePack:
    return EvidencePack(
        query=query,
        concept_name=None,
        retrieval_family=None,
        evidence_items=[],
    )


def build_identity_generated_answer(
    *,
    query: str,
    status: str,
    answer_text: str,
    citations: list[str],
    evidence_summary: str,
) -> GeneratedAnswer:
    grounded = GroundedAnswer(
        status=status,
        answer=answer_text,
        citations=citations,
        evidence_summary=evidence_summary,
        safety_note=RXNORM_IDENTITY_SAFETY_NOTE,
    )

    return GeneratedAnswer(
        evidence_pack=empty_identity_evidence_pack(query),
        answer=grounded,
        raw_model_output=None,
        verification=None,
        proposed_answer=grounded,
        verification_overrode_answer=False,
        guardrail_triggered=False,
        guardrail_reason=None,
    )


def build_identity_insufficient_result(
    *,
    query: str,
    evidence_items: list[RxNormIdentityEvidence] | None = None,
) -> RxNormIdentityAnswerResult:
    generated = build_identity_generated_answer(
        query=query,
        status="insufficient_evidence",
        answer_text=RXNORM_IDENTITY_INSUFFICIENT_ANSWER,
        citations=[],
        evidence_summary=RXNORM_IDENTITY_INSUFFICIENT_SUMMARY,
    )

    return RxNormIdentityAnswerResult(
        generated=generated,
        evidence_items=evidence_items or [],
    )


def clean_extracted_term(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.strip(" \t\n\r?!.:,;\"'")
    cleaned = re.sub(r"^(?:the\s+)", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def concept_display_name(concept: RxNormConcept | None) -> str:
    if concept is None:
        return "an unresolved RxNorm concept"

    if concept.name:
        return concept.name

    return concept.rxcui


def unique_nonempty_names(concepts: Iterable[RxNormConcept]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for concept in concepts:
        if not concept.name:
            continue

        if concept.name in seen:
            continue

        seen.add(concept.name)
        names.append(concept.name)

    return names


def preferred_ingredient_names(
    ingredients: list[RxNormConcept],
) -> list[str]:
    ingredient_level_names = unique_nonempty_names(
        concept
        for concept in ingredients
        if concept.tty == "IN"
    )

    if ingredient_level_names:
        return ingredient_level_names

    return unique_nonempty_names(ingredients)


def ingredient_rxcui_set(
    resolution: RxNormIdentityTermResolution,
) -> set[str]:
    return {
        ingredient.rxcui
        for ingredient in resolution.related_ingredients
    }


def related_concepts_by_type(
    *,
    client: RxNormClient,
    rxcui: str,
    term_types: list[str],
) -> list[RxNormConcept]:
    payloads = client.get_related_by_type(
        rxcui,
        term_types=term_types,
    )

    concepts: list[RxNormConcept] = []

    for payload in payloads:
        concept = concept_from_related_payload(payload)

        if concept is not None:
            concepts.append(concept)

    return unique_concepts_by_rxcui(concepts)


# =============================================================================
# RxNorm term resolution for identity questions
# =============================================================================

def resolve_identity_term(
    term: str,
) -> RxNormIdentityTermResolution:
    cleaned_term = clean_extracted_term(term)

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
            return RxNormIdentityTermResolution(
                input_term=cleaned_term,
                status="no_rxnorm_match",
                selected_candidate=None,
                candidates=[],
                related_ingredients=[],
                related_brands=[],
            )

        if len(candidates) > 1:
            return RxNormIdentityTermResolution(
                input_term=cleaned_term,
                status="ambiguous",
                selected_candidate=None,
                candidates=candidates,
                related_ingredients=[],
                related_brands=[],
            )

        selected_candidate = candidates[0]

        related_ingredients: list[RxNormConcept] = []
        related_brands: list[RxNormConcept] = []

        if selected_candidate.tty in INGREDIENT_TERM_TYPES:
            related_ingredients.append(selected_candidate)

        if selected_candidate.tty in BRAND_TERM_TYPES:
            related_brands.append(selected_candidate)

        related_ingredients.extend(
            related_concepts_by_type(
                client=client,
                rxcui=selected_candidate.rxcui,
                term_types=["IN", "PIN"],
            )
        )

        related_brands.extend(
            related_concepts_by_type(
                client=client,
                rxcui=selected_candidate.rxcui,
                term_types=["BN"],
            )
        )

    return RxNormIdentityTermResolution(
        input_term=cleaned_term,
        status="resolved",
        selected_candidate=selected_candidate,
        candidates=candidates,
        related_ingredients=unique_concepts_by_rxcui(
            related_ingredients
        ),
        related_brands=unique_concepts_by_rxcui(
            related_brands
        ),
    )


def summarize_identity_resolution(
    resolution: RxNormIdentityTermResolution,
) -> str:
    if resolution.status == "no_rxnorm_match":
        return (
            f"RxNorm did not return an identity match for "
            f"'{resolution.input_term}'."
        )

    if resolution.status == "ambiguous":
        candidate_names = unique_nonempty_names(resolution.candidates)
        candidate_text = ", ".join(candidate_names) or "multiple candidates"

        return (
            f"RxNorm returned multiple possible identity matches for "
            f"'{resolution.input_term}': {candidate_text}."
        )

    selected_name = concept_display_name(resolution.selected_candidate)
    selected_tty = (
        resolution.selected_candidate.tty
        if resolution.selected_candidate is not None
        else "unknown"
    )
    selected_rxcui = (
        resolution.selected_candidate.rxcui
        if resolution.selected_candidate is not None
        else "unknown"
    )

    ingredient_names = preferred_ingredient_names(
        resolution.related_ingredients
    )
    brand_names = unique_nonempty_names(
        resolution.related_brands
    )

    ingredient_text = (
        ", ".join(ingredient_names)
        if ingredient_names
        else "none returned"
    )

    brand_text = (
        ", ".join(brand_names)
        if brand_names
        else "none returned"
    )

    return (
        f"RxNorm resolved '{resolution.input_term}' to "
        f"{selected_name} ({selected_tty}, RxCUI {selected_rxcui}). "
        f"Related ingredient concept(s): {ingredient_text}. "
        f"Related brand concept(s): {brand_text}."
    )


def build_identity_evidence(
    *,
    evidence_id: str,
    resolution: RxNormIdentityTermResolution,
) -> RxNormIdentityEvidence:
    return RxNormIdentityEvidence(
        evidence_id=evidence_id,
        term=resolution.input_term,
        resolution_status=resolution.status,
        selected_candidate=resolution.selected_candidate,
        related_ingredients=resolution.related_ingredients,
        related_brands=resolution.related_brands,
        summary=summarize_identity_resolution(resolution),
    )


# =============================================================================
# Query parsing
# =============================================================================

EQUIVALENCE_PATTERNS = [
    re.compile(
        r"^\s*(?:is|are)\s+(?P<left>.+?)\s+"
        r"(?:the\s+)?same\s+as\s+(?P<right>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:is|are)\s+(?P<left>.+?)\s+"
        r"equivalent\s+to\s+(?P<right>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:is|are)\s+(?P<left>.+?)\s+"
        r"another\s+name\s+for\s+(?P<right>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
]

GENERIC_NAME_PATTERNS = [
    re.compile(
        r"^\s*(?:what\s+is|what's)\s+(?:the\s+)?"
        r"generic\s+name\s+of\s+(?P<term>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:what\s+is|what's)\s+(?:the\s+)?"
        r"generic\s+for\s+(?P<term>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
]

ACTIVE_INGREDIENT_PATTERNS = [
    re.compile(
        r"^\s*(?:what\s+is|what's)\s+(?:the\s+)?"
        r"active\s+ingredient\s+(?:in|of)\s+(?P<term>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:what\s+)?ingredient\s+(?:is\s+)?"
        r"(?:in|of)\s+(?P<term>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
]

BRAND_NAME_OF_PATTERNS = [
    re.compile(
        r"^\s*(?:what\s+is|what's)\s+(?:the\s+)?"
        r"brand\s+name\s+of\s+(?P<term>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
]

IS_BRAND_NAME_PATTERNS = [
    re.compile(
        r"^\s*(?:is|are)\s+(?P<term>.+?)\s+"
        r"(?:a\s+)?brand\s+name\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
]

DEFINITION_PATTERNS = [
    re.compile(
        r"^\s*(?:what\s+is|what's)\s+(?P<term>.+?)\s*[?!.]*\s*$",
        flags=re.IGNORECASE,
    ),
]


def parse_equivalence_terms(query: str) -> tuple[str, str] | None:
    for pattern in EQUIVALENCE_PATTERNS:
        match = pattern.match(query)

        if match is None:
            continue

        left = clean_extracted_term(match.group("left"))
        right = clean_extracted_term(match.group("right"))

        if left and right:
            return left, right

    return None


def parse_single_term(
    query: str,
    *,
    patterns: list[re.Pattern[str]],
) -> str | None:
    for pattern in patterns:
        match = pattern.match(query)

        if match is None:
            continue

        term = clean_extracted_term(match.group("term"))

        if term:
            return term

    return None


# =============================================================================
# Intent-specific answer builders
# =============================================================================

def answer_equivalence_query(
    query: str,
) -> RxNormIdentityAnswerResult:
    parsed = parse_equivalence_terms(query)

    if parsed is None:
        return build_identity_insufficient_result(query=query)

    left_term, right_term = parsed

    left_resolution = resolve_identity_term(left_term)
    right_resolution = resolve_identity_term(right_term)

    evidence_items = [
        build_identity_evidence(
            evidence_id="R1",
            resolution=left_resolution,
        ),
        build_identity_evidence(
            evidence_id="R2",
            resolution=right_resolution,
        ),
    ]

    if (
        left_resolution.status != "resolved"
        or right_resolution.status != "resolved"
    ):
        return build_identity_insufficient_result(
            query=query,
            evidence_items=evidence_items,
        )

    shared_ingredient_rxcuis = (
        ingredient_rxcui_set(left_resolution)
        & ingredient_rxcui_set(right_resolution)
    )

    left_display = concept_display_name(
        left_resolution.selected_candidate
    )
    right_display = concept_display_name(
        right_resolution.selected_candidate
    )

    if shared_ingredient_rxcuis:
        shared_names = preferred_ingredient_names(
            [
                ingredient
                for ingredient in left_resolution.related_ingredients
                if ingredient.rxcui in shared_ingredient_rxcuis
            ]
        )

        shared_text = ", ".join(shared_names) or "a shared ingredient concept"

        generated = build_identity_generated_answer(
            query=query,
            status="answered",
            answer_text=(
                f"Yes. RxNorm maps {left_display} and {right_display} "
                f"to the same ingredient concept: {shared_text}."
            ),
            citations=["R1", "R2"],
            evidence_summary=(
                "The two RxNorm term resolutions share at least one ingredient "
                "RxCUI, supporting a same-ingredient identity relationship."
            ),
        )

        return RxNormIdentityAnswerResult(
            generated=generated,
            evidence_items=evidence_items,
        )

    generated = build_identity_generated_answer(
        query=query,
        status="answered",
        answer_text=(
            f"No. The RxNorm identity mappings for {left_display} and "
            f"{right_display} do not share the same ingredient concept."
        ),
        citations=["R1", "R2"],
        evidence_summary=(
            "The two RxNorm term resolutions were valid, but no shared "
            "ingredient RxCUI was found."
        ),
    )

    return RxNormIdentityAnswerResult(
        generated=generated,
        evidence_items=evidence_items,
    )


def answer_generic_name_query(
    query: str,
) -> RxNormIdentityAnswerResult:
    term = parse_single_term(
        query,
        patterns=GENERIC_NAME_PATTERNS,
    )

    if term is None:
        return build_identity_insufficient_result(query=query)

    resolution = resolve_identity_term(term)

    evidence_items = [
        build_identity_evidence(
            evidence_id="R1",
            resolution=resolution,
        )
    ]

    if resolution.status != "resolved":
        return build_identity_insufficient_result(
            query=query,
            evidence_items=evidence_items,
        )

    ingredient_names = preferred_ingredient_names(
        resolution.related_ingredients
    )

    if not ingredient_names:
        return build_identity_insufficient_result(
            query=query,
            evidence_items=evidence_items,
        )

    generated = build_identity_generated_answer(
        query=query,
        status="answered",
        answer_text=(
            f"The RxNorm generic ingredient associated with {term} is "
            f"{', '.join(ingredient_names)}."
        ),
        citations=["R1"],
        evidence_summary=(
            "RxNorm resolved the requested term and returned a related "
            "ingredient concept."
        ),
    )

    return RxNormIdentityAnswerResult(
        generated=generated,
        evidence_items=evidence_items,
    )


def answer_active_ingredient_query(
    query: str,
) -> RxNormIdentityAnswerResult:
    term = parse_single_term(
        query,
        patterns=ACTIVE_INGREDIENT_PATTERNS,
    )

    if term is None:
        return build_identity_insufficient_result(query=query)

    resolution = resolve_identity_term(term)

    evidence_items = [
        build_identity_evidence(
            evidence_id="R1",
            resolution=resolution,
        )
    ]

    if resolution.status != "resolved":
        return build_identity_insufficient_result(
            query=query,
            evidence_items=evidence_items,
        )

    ingredient_names = preferred_ingredient_names(
        resolution.related_ingredients
    )

    if not ingredient_names:
        return build_identity_insufficient_result(
            query=query,
            evidence_items=evidence_items,
        )

    generated = build_identity_generated_answer(
        query=query,
        status="answered",
        answer_text=(
            f"The RxNorm active ingredient associated with {term} is "
            f"{', '.join(ingredient_names)}."
        ),
        citations=["R1"],
        evidence_summary=(
            "RxNorm resolved the requested term and returned related "
            "ingredient concept data."
        ),
    )

    return RxNormIdentityAnswerResult(
        generated=generated,
        evidence_items=evidence_items,
    )


def answer_brand_name_query(
    query: str,
) -> RxNormIdentityAnswerResult:
    brand_of_term = parse_single_term(
        query,
        patterns=BRAND_NAME_OF_PATTERNS,
    )

    if brand_of_term is not None:
        resolution = resolve_identity_term(brand_of_term)

        evidence_items = [
            build_identity_evidence(
                evidence_id="R1",
                resolution=resolution,
            )
        ]

        if resolution.status != "resolved":
            return build_identity_insufficient_result(
                query=query,
                evidence_items=evidence_items,
            )

        brand_names = unique_nonempty_names(
            resolution.related_brands
        )

        if not brand_names:
            return build_identity_insufficient_result(
                query=query,
                evidence_items=evidence_items,
            )

        generated = build_identity_generated_answer(
            query=query,
            status="answered",
            answer_text=(
                f"RxNorm lists {', '.join(brand_names)} as brand-name "
                f"concept(s) related to {brand_of_term}."
            ),
            citations=["R1"],
            evidence_summary=(
                "RxNorm resolved the requested term and returned related "
                "brand-name concept data."
            ),
        )

        return RxNormIdentityAnswerResult(
            generated=generated,
            evidence_items=evidence_items,
        )

    is_brand_term = parse_single_term(
        query,
        patterns=IS_BRAND_NAME_PATTERNS,
    )

    if is_brand_term is None:
        return build_identity_insufficient_result(query=query)

    resolution = resolve_identity_term(is_brand_term)

    evidence_items = [
        build_identity_evidence(
            evidence_id="R1",
            resolution=resolution,
        )
    ]

    if (
        resolution.status != "resolved"
        or resolution.selected_candidate is None
    ):
        return build_identity_insufficient_result(
            query=query,
            evidence_items=evidence_items,
        )

    selected = resolution.selected_candidate

    if selected.tty == "BN":
        answer_text = (
            f"Yes. RxNorm identifies {concept_display_name(selected)} "
            f"as a brand-name medication concept."
        )
    else:
        answer_text = (
            f"No. RxNorm identifies {concept_display_name(selected)} as a "
            f"{selected.tty or 'non-brand'} concept, not a brand-name concept."
        )

    generated = build_identity_generated_answer(
        query=query,
        status="answered",
        answer_text=answer_text,
        citations=["R1"],
        evidence_summary=(
            "RxNorm concept properties identify whether the resolved term "
            "is represented as a brand-name concept."
        ),
    )

    return RxNormIdentityAnswerResult(
        generated=generated,
        evidence_items=evidence_items,
    )


def answer_definition_query(
    query: str,
) -> RxNormIdentityAnswerResult:
    term = parse_single_term(
        query,
        patterns=DEFINITION_PATTERNS,
    )

    if term is None:
        return build_identity_insufficient_result(query=query)

    resolution = resolve_identity_term(term)

    evidence_items = [
        build_identity_evidence(
            evidence_id="R1",
            resolution=resolution,
        )
    ]

    if (
        resolution.status != "resolved"
        or resolution.selected_candidate is None
    ):
        return build_identity_insufficient_result(
            query=query,
            evidence_items=evidence_items,
        )

    selected = resolution.selected_candidate
    ingredient_names = preferred_ingredient_names(
        resolution.related_ingredients
    )

    if selected.tty == "BN" and ingredient_names:
        answer_text = (
            f"RxNorm identifies {concept_display_name(selected)} as a "
            f"brand-name medication concept associated with the ingredient "
            f"{', '.join(ingredient_names)}."
        )
    elif selected.tty == "IN":
        answer_text = (
            f"RxNorm identifies {concept_display_name(selected)} as an "
            f"ingredient medication concept."
        )
    elif selected.tty == "PIN":
        answer_text = (
            f"RxNorm identifies {concept_display_name(selected)} as a "
            f"precise-ingredient medication concept."
        )
    else:
        answer_text = (
            f"RxNorm identifies {concept_display_name(selected)} as a "
            f"{selected.tty or 'medication'} concept."
        )

    generated = build_identity_generated_answer(
        query=query,
        status="answered",
        answer_text=answer_text,
        citations=["R1"],
        evidence_summary=(
            "RxNorm concept properties and related ingredient concepts were "
            "used to summarize the medication identity."
        ),
    )

    return RxNormIdentityAnswerResult(
        generated=generated,
        evidence_items=evidence_items,
    )


# =============================================================================
# Public entry point
# =============================================================================

def answer_rxnorm_identity_query(
    query: str,
    *,
    intent: str | None,
) -> RxNormIdentityAnswerResult:
    if intent == "brand_generic_equivalence":
        return answer_equivalence_query(query)

    if intent == "generic_name_lookup":
        return answer_generic_name_query(query)

    if intent == "ingredient_identity":
        return answer_active_ingredient_query(query)

    if intent == "brand_name_lookup":
        return answer_brand_name_query(query)

    if intent == "drug_identity_definition":
        return answer_definition_query(query)

    return build_identity_insufficient_result(query=query)