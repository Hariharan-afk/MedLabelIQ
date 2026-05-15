from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from medlabeliq.rxnorm.models import DrugTermResolution, RxNormConcept
from medlabeliq.rxnorm.resolver import (
    list_indexed_corpus_drugs,
    normalize_match_text,
    resolve_drug_term,
)


DrugMentionDetectionStatus = Literal[
    "not_attempted_explicit_filter_present",
    "direct_corpus_mention",
    "rxnorm_resolved_query_mention",
    "ambiguous",
    "no_mention_detected",
]


@dataclass(frozen=True)
class DrugMentionCandidateResolution:
    mention_text: str
    resolution: DrugTermResolution


@dataclass(frozen=True)
class DrugMentionDetection:
    query: str
    status: DrugMentionDetectionStatus
    detected_mention: str | None
    retrieval_drug: str | None
    corpus_matches: list[str]
    selected_candidate: RxNormConcept | None
    candidate_resolutions: list[DrugMentionCandidateResolution]

    @property
    def can_filter(self) -> bool:
        return self.status in {
            "direct_corpus_mention",
            "rxnorm_resolved_query_mention",
        }


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

CANDIDATE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "can",
    "cause",
    "causes",
    "caused",
    "could",
    "do",
    "does",
    "drug",
    "for",
    "from",
    "have",
    "help",
    "if",
    "in",
    "is",
    "it",
    "may",
    "medication",
    "medicine",
    "of",
    "or",
    "should",
    "the",
    "this",
    "to",
    "treat",
    "treats",
    "use",
    "used",
    "using",
    "what",
    "when",
    "will",
    "with",
}

MAX_CANDIDATE_SPANS = 18
MAX_SPAN_TOKENS = 3


def build_not_attempted_detection(
    query: str,
) -> DrugMentionDetection:
    return DrugMentionDetection(
        query=query,
        status="not_attempted_explicit_filter_present",
        detected_mention=None,
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        candidate_resolutions=[],
    )


def normalized_phrase_occurs_in_query(
    *,
    phrase: str,
    query: str,
) -> bool:
    normalized_phrase = normalize_match_text(phrase)
    normalized_query = normalize_match_text(query)

    if not normalized_phrase or not normalized_query:
        return False

    padded_phrase = f" {normalized_phrase} "
    padded_query = f" {normalized_query} "

    return padded_phrase in padded_query


def detect_direct_corpus_mentions(
    *,
    query: str,
    indexed_drugs: list[str],
) -> list[str]:
    matches: list[str] = []

    for drug in indexed_drugs:
        if normalized_phrase_occurs_in_query(
            phrase=drug,
            query=query,
        ):
            matches.append(drug)

    return matches


def extract_candidate_spans(query: str) -> list[str]:
    """
    Extract conservative candidate mention spans from query text.

    Policy:
    - tokenize alphabetic / hyphenated text,
    - try single-token spans first,
    - then 2-token and 3-token spans,
    - skip spans that are too short or entirely stopwords,
    - cap outbound RxNorm attempts to avoid excessive API calls.
    """
    tokens = TOKEN_PATTERN.findall(query)

    if not tokens:
        return []

    candidate_spans: list[str] = []
    seen_normalized: set[str] = set()

    for span_size in range(1, MAX_SPAN_TOKENS + 1):
        for start_idx in range(0, len(tokens) - span_size + 1):
            span_tokens = tokens[start_idx : start_idx + span_size]
            raw_span = " ".join(span_tokens)
            normalized_span = normalize_match_text(raw_span)

            if not normalized_span:
                continue

            normalized_parts = normalized_span.split()

            if all(
                part in CANDIDATE_STOPWORDS
                for part in normalized_parts
            ):
                continue

            compact_length = len(normalized_span.replace(" ", ""))
            if compact_length < 4:
                continue

            if normalized_span in seen_normalized:
                continue

            seen_normalized.add(normalized_span)
            candidate_spans.append(raw_span)

            if len(candidate_spans) >= MAX_CANDIDATE_SPANS:
                return candidate_spans

    return candidate_spans


def choose_candidate_for_corpus_drug(
    *,
    candidate_resolutions: list[DrugMentionCandidateResolution],
    corpus_drug: str,
) -> DrugMentionCandidateResolution | None:
    for candidate_resolution in candidate_resolutions:
        if corpus_drug in candidate_resolution.resolution.corpus_matches:
            return candidate_resolution

    return None


def detect_drug_mention_from_query(
    query: str,
) -> DrugMentionDetection:
    """
    Detect one safe drug concept from the query text.

    Resolution order:
    1. Direct indexed corpus mention, e.g. 'metformin'.
    2. Candidate phrase extraction + RxNorm mapping, e.g.:
       - 'Glucophage' -> metformin
       - 'metformn' -> metformin
    3. If multiple corpus drugs are found, return ambiguous and do not filter.
    4. If no safe mention is found, return no_mention_detected.
    """
    indexed_drugs = list_indexed_corpus_drugs()

    direct_matches = detect_direct_corpus_mentions(
        query=query,
        indexed_drugs=indexed_drugs,
    )

    if len(direct_matches) == 1:
        return DrugMentionDetection(
            query=query,
            status="direct_corpus_mention",
            detected_mention=direct_matches[0],
            retrieval_drug=direct_matches[0],
            corpus_matches=direct_matches,
            selected_candidate=None,
            candidate_resolutions=[],
        )

    if len(direct_matches) > 1:
        return DrugMentionDetection(
            query=query,
            status="ambiguous",
            detected_mention=None,
            retrieval_drug=None,
            corpus_matches=direct_matches,
            selected_candidate=None,
            candidate_resolutions=[],
        )

    candidate_spans = extract_candidate_spans(query)

    candidate_resolutions: list[DrugMentionCandidateResolution] = []

    resolved_candidate_resolutions: list[
        DrugMentionCandidateResolution
    ] = []

    fallback_candidate_resolutions: list[
        DrugMentionCandidateResolution
    ] = []

    resolved_corpus_matches: OrderedDict[str, None] = OrderedDict()
    fallback_corpus_matches: OrderedDict[str, None] = OrderedDict()

    for candidate_span in candidate_spans:
        resolution = resolve_drug_term(candidate_span)

        if not resolution.corpus_matches:
            continue

        candidate_resolution = DrugMentionCandidateResolution(
            mention_text=candidate_span,
            resolution=resolution,
        )

        candidate_resolutions.append(candidate_resolution)

        # Prefer clearly resolved candidate spans over ambiguous fuzzy spans.
        # This prevents noisy text like "name and" from overriding a strong
        # brand-name detection such as "Glucophage" -> "metformin".
        if resolution.status == "resolved":
            resolved_candidate_resolutions.append(candidate_resolution)

            for corpus_match in resolution.corpus_matches:
                resolved_corpus_matches[corpus_match] = None
        else:
            fallback_candidate_resolutions.append(candidate_resolution)

            for corpus_match in resolution.corpus_matches:
                fallback_corpus_matches[corpus_match] = None

    # If at least one clearly resolved span exists, use only those matches
    # to decide the retrieval drug. Ambiguous fuzzy spans remain available
    # in diagnostics but no longer contaminate the final drug decision.
    if resolved_corpus_matches:
        unique_corpus_matches = list(resolved_corpus_matches.keys())
        candidate_pool = resolved_candidate_resolutions
    else:
        unique_corpus_matches = list(fallback_corpus_matches.keys())
        candidate_pool = fallback_candidate_resolutions

    if len(unique_corpus_matches) == 1:
        corpus_drug = unique_corpus_matches[0]

        chosen_candidate = choose_candidate_for_corpus_drug(
            candidate_resolutions=candidate_pool,
            corpus_drug=corpus_drug,
        )

        selected_candidate = (
            chosen_candidate.resolution.selected_candidate
            if chosen_candidate is not None
            else None
        )

        detected_mention = (
            chosen_candidate.mention_text
            if chosen_candidate is not None
            else None
        )

        return DrugMentionDetection(
            query=query,
            status="rxnorm_resolved_query_mention",
            detected_mention=detected_mention,
            retrieval_drug=corpus_drug,
            corpus_matches=unique_corpus_matches,
            selected_candidate=selected_candidate,
            candidate_resolutions=candidate_resolutions,
        )

    if len(unique_corpus_matches) > 1:
        return DrugMentionDetection(
            query=query,
            status="ambiguous",
            detected_mention=None,
            retrieval_drug=None,
            corpus_matches=unique_corpus_matches,
            selected_candidate=None,
            candidate_resolutions=candidate_resolutions,
        )

    return DrugMentionDetection(
        query=query,
        status="no_mention_detected",
        detected_mention=None,
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        candidate_resolutions=[],
    )