from __future__ import annotations

import medlabeliq.orchestration.drug_mention_detection as module
from medlabeliq.rxnorm.models import DrugTermResolution, RxNormConcept


def test_direct_corpus_mention_is_detected_without_rxnorm(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    def should_not_be_called(term: str):
        raise AssertionError("RxNorm should not run for direct corpus mentions.")

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        should_not_be_called,
    )

    result = module.detect_drug_mention_from_query(
        "Can metformin cause lactic acidosis?"
    )

    assert result.status == "direct_corpus_mention"
    assert result.detected_mention == "metformin"
    assert result.retrieval_drug == "metformin"
    assert result.can_filter is True


def test_brand_name_in_query_resolves_via_rxnorm(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    selected_candidate = RxNormConcept(
        rxcui="151827",
        name="Glucophage",
        synonym="",
        tty="BN",
        match_method="exact_or_normalized",
    )

    def fake_resolve(term: str) -> DrugTermResolution:
        if term == "Glucophage":
            return DrugTermResolution(
                input_term=term,
                status="resolved",
                corpus_concept="metformin",
                corpus_matches=["metformin"],
                selected_candidate=selected_candidate,
                candidates=[],
            )

        return DrugTermResolution(
            input_term=term,
            status="no_rxnorm_match",
            corpus_concept=None,
            corpus_matches=[],
            selected_candidate=None,
            candidates=[],
        )

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        fake_resolve,
    )

    result = module.detect_drug_mention_from_query(
        "Can Glucophage cause dangerous acid buildup in the blood?"
    )

    assert result.status == "rxnorm_resolved_query_mention"
    assert result.detected_mention == "Glucophage"
    assert result.retrieval_drug == "metformin"
    assert result.selected_candidate is not None
    assert result.selected_candidate.name == "Glucophage"


def test_typo_in_query_resolves_via_rxnorm(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    selected_candidate = RxNormConcept(
        rxcui="6809",
        name="metformin",
        synonym="",
        tty="IN",
        source="GS",
        score=8.4,
        rank=1,
        match_method="approximate",
    )

    def fake_resolve(term: str) -> DrugTermResolution:
        if term == "metformn":
            return DrugTermResolution(
                input_term=term,
                status="resolved",
                corpus_concept="metformin",
                corpus_matches=["metformin"],
                selected_candidate=selected_candidate,
                candidates=[],
            )

        return DrugTermResolution(
            input_term=term,
            status="no_rxnorm_match",
            corpus_concept=None,
            corpus_matches=[],
            selected_candidate=None,
            candidates=[],
        )

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        fake_resolve,
    )

    result = module.detect_drug_mention_from_query(
        "Can metformn cause dangerous acid buildup in the blood?"
    )

    assert result.status == "rxnorm_resolved_query_mention"
    assert result.detected_mention == "metformn"
    assert result.retrieval_drug == "metformin"


def test_multiple_direct_drugs_are_marked_ambiguous(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    result = module.detect_drug_mention_from_query(
        "Can metformin interact with apixaban?"
    )

    assert result.status == "ambiguous"
    assert result.retrieval_drug is None
    assert result.corpus_matches == ["metformin", "apixaban"]


def test_no_detectable_drug_mention(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        lambda term: DrugTermResolution(
            input_term=term,
            status="no_rxnorm_match",
            corpus_concept=None,
            corpus_matches=[],
            selected_candidate=None,
            candidates=[],
        ),
    )

    result = module.detect_drug_mention_from_query(
        "Can this medication cause side effects?"
    )

    assert result.status == "no_mention_detected"
    assert result.retrieval_drug is None
    assert result.can_filter is False


def test_resolved_brand_match_wins_over_noisy_ambiguous_span(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "atorvastatin"],
    )

    glucophage_candidate = RxNormConcept(
        rxcui="151827",
        name="Glucophage",
        synonym="",
        tty="BN",
        match_method="exact_or_normalized",
    )

    def fake_resolve(term: str) -> DrugTermResolution:
        if term == "Glucophage":
            return DrugTermResolution(
                input_term=term,
                status="resolved",
                corpus_concept="metformin",
                corpus_matches=["metformin"],
                selected_candidate=glucophage_candidate,
                candidates=[],
            )

        if term in {"name and", "name and what"}:
            return DrugTermResolution(
                input_term=term,
                status="ambiguous",
                corpus_concept=None,
                corpus_matches=["metformin", "atorvastatin"],
                selected_candidate=None,
                candidates=[],
            )

        return DrugTermResolution(
            input_term=term,
            status="no_rxnorm_match",
            corpus_concept=None,
            corpus_matches=[],
            selected_candidate=None,
            candidates=[],
        )

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        fake_resolve,
    )

    result = module.detect_drug_mention_from_query(
        "Is Glucophage a brand name and what is it used for?"
    )

    assert result.status == "rxnorm_resolved_query_mention"
    assert result.detected_mention == "Glucophage"
    assert result.retrieval_drug == "metformin"
    assert result.corpus_matches == ["metformin"]