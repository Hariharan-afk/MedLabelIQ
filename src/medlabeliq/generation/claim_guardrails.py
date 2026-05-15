from __future__ import annotations

import re
from dataclasses import dataclass

from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack


# ---------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimGuardrailDecision:
    should_abstain: bool
    reason: str | None = None


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(marker in normalized for marker in markers)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _get_cited_evidence_text(
    *,
    answer: GroundedAnswer,
    evidence_pack: EvidencePack,
) -> str:
    cited_ids = set(answer.citations)

    return "\n".join(
        item.chunk_text
        for item in evidence_pack.evidence_items
        if item.evidence_id in cited_ids
    )


# ---------------------------------------------------------------------
# Guardrail 1: Guarantee-style claims
# ---------------------------------------------------------------------

GUARANTEE_QUERY_MARKERS = (
    "guarantee",
    "guarantees",
    "guaranteed",
)


def assess_guarantee_claim_support(
    *,
    question: str,
    answer: GroundedAnswer,
    evidence_pack: EvidencePack,
) -> ClaimGuardrailDecision:
    """
    Deterministic guardrail for guarantee-style claims.

    Example:
    - "Does metformin guarantee weight loss?"

    Loosely related clinical-study evidence should not be treated as
    direct support for a guarantee-level conclusion unless the cited
    evidence itself explicitly addresses guarantee-level certainty.
    """
    if answer.status != "answered":
        return ClaimGuardrailDecision(should_abstain=False)

    if not _contains_any_marker(question, GUARANTEE_QUERY_MARKERS):
        return ClaimGuardrailDecision(should_abstain=False)

    cited_text = _get_cited_evidence_text(
        answer=answer,
        evidence_pack=evidence_pack,
    )

    evidence_explicitly_addresses_guarantee = _contains_any_marker(
        cited_text,
        GUARANTEE_QUERY_MARKERS,
    )

    if evidence_explicitly_addresses_guarantee:
        return ClaimGuardrailDecision(should_abstain=False)

    return ClaimGuardrailDecision(
        should_abstain=True,
        reason=(
            "The user asked a guarantee-style claim, but the cited label "
            "evidence did not explicitly establish or address guarantee-level certainty."
        ),
    )


# ---------------------------------------------------------------------
# Guardrail 2: Negative treatment claims by omission
# ---------------------------------------------------------------------

THERAPEUTIC_USE_QUERY_MARKERS = (
    "treat",
    "treats",
    "treated",
    "cure",
    "cures",
    "used for",
    "help with",
    "help treat",
)

NEGATIVE_ANSWER_MARKERS = (
    "no,",
    "no.",
    "does not",
    "do not",
    "doesn't",
    "is not",
    "are not",
    "not used",
    "not indicated",
    "not for",
    "cannot",
    "can't",
)

EXPLICIT_NEGATIVE_EVIDENCE_MARKERS = (
    "does not",
    "do not",
    "is not",
    "are not",
    "not used",
    "not indicated",
    "not for",
    "cannot",
    "can't",
)


def _extract_treatment_target(question: str) -> str | None:
    """
    Extract the condition/object of a therapeutic-use question.

    Examples:
    - "Does apixaban treat bacterial infections?"
      -> "bacterial infections"
    - "Does albuterol cure pneumonia?"
      -> "pneumonia"
    - "Does omeprazole help treat lactose intolerance?"
      -> "lactose intolerance"

    This is intentionally lightweight and designed for our evaluation/query style.
    """
    normalized = _normalize_text(question).rstrip("?.")

    patterns = [
        r"\b(?:does|do|can|could|will|would|is|are)\s+.+?\s+treat\s+(.+)$",
        r"\b(?:does|do|can|could|will|would|is|are)\s+.+?\s+cure\s+(.+)$",
        r"\b(?:does|do|can|could|will|would|is|are)\s+.+?\s+help treat\s+(.+)$",
        r"\b(?:does|do|can|could|will|would|is|are)\s+.+?\s+help with\s+(.+)$",
        r"\b(?:is|are)\s+.+?\s+used for\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            target = match.group(1).strip()
            return target or None

    return None


def _target_keywords(target: str) -> list[str]:
    """
    Convert extracted target phrase into content keywords.

    We remove only very light stop words so phrases like:
    - "bacterial infections" -> ["bacterial", "infections"]
    - "the flu" -> ["flu"]
    """
    stop_words = {
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "with",
        "in",
        "on",
        "from",
    }

    words = re.findall(r"[a-zA-Z0-9]+", target.lower())

    return [
        word
        for word in words
        if word not in stop_words and len(word) > 1
    ]


def _evidence_mentions_target(
    *,
    cited_text: str,
    target: str | None,
) -> bool:
    """
    Require the cited evidence to mention the therapeutic target itself.

    This prevents:
    - question: "Does apixaban treat bacterial infections?"
    - evidence: "Apixaban is not for use in APS..."
    from being treated as direct negative support.
    """
    if not target:
        return False

    normalized_evidence = _normalize_text(cited_text)
    normalized_target = _normalize_text(target)

    # Strongest signal: exact target phrase appears.
    if normalized_target in normalized_evidence:
        return True

    # Fallback: all meaningful target keywords appear somewhere.
    keywords = _target_keywords(target)
    if not keywords:
        return False

    return all(keyword in normalized_evidence for keyword in keywords)


def assess_negative_treatment_claim_support(
    *,
    question: str,
    answer: GroundedAnswer,
    evidence_pack: EvidencePack,
) -> ClaimGuardrailDecision:
    """
    Guardrail for negative therapeutic-use claims inferred only by omission.

    Example:
    - Question: "Does apixaban treat bacterial infections?"
    - Answer: "No, apixaban does not treat bacterial infections."

    A list of a drug's approved uses does not directly prove that every
    unlisted condition is not treated. To answer a negative treatment claim,
    the cited evidence must:
    1. Explicitly mention the target condition/object asked about.
    2. Contain explicit negative/exclusion language.
    """
    if answer.status != "answered":
        return ClaimGuardrailDecision(should_abstain=False)

    question_is_therapeutic_use_claim = _contains_any_marker(
        question,
        THERAPEUTIC_USE_QUERY_MARKERS,
    )

    if not question_is_therapeutic_use_claim:
        return ClaimGuardrailDecision(should_abstain=False)

    answer_is_negative = _contains_any_marker(
        answer.answer,
        NEGATIVE_ANSWER_MARKERS,
    )

    if not answer_is_negative:
        return ClaimGuardrailDecision(should_abstain=False)

    cited_text = _get_cited_evidence_text(
        answer=answer,
        evidence_pack=evidence_pack,
    )

    target = _extract_treatment_target(question)

    evidence_mentions_target = _evidence_mentions_target(
        cited_text=cited_text,
        target=target,
    )

    if not evidence_mentions_target:
        return ClaimGuardrailDecision(
            should_abstain=True,
            reason=(
                "The answer makes a negative treatment-use claim, but the cited "
                "label evidence does not mention the therapeutic target asked about."
            ),
        )

    evidence_explicitly_states_negative_use = _contains_any_marker(
        cited_text,
        EXPLICIT_NEGATIVE_EVIDENCE_MARKERS,
    )

    if not evidence_explicitly_states_negative_use:
        return ClaimGuardrailDecision(
            should_abstain=True,
            reason=(
                "The answer makes a negative treatment-use claim, but the cited "
                "label evidence does not explicitly establish that negative claim."
            ),
        )

    return ClaimGuardrailDecision(should_abstain=False)