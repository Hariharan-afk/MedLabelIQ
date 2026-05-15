from __future__ import annotations

import medlabeliq.orchestration.drug_filter_resolution as module
from medlabeliq.rxnorm.models import DrugTermResolution, RxNormConcept


def test_direct_corpus_match_skips_rxnorm(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    def should_not_be_called(term: str):
        raise AssertionError("RxNorm should not be called for direct corpus matches.")

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        should_not_be_called,
    )

    result = module.resolve_optional_drug_filter("Metformin")

    assert result.status == "direct_corpus_match"
    assert result.retrieval_drug == "metformin"
    assert result.can_retrieve is True


def test_rxnorm_brand_resolution_maps_to_corpus(monkeypatch) -> None:
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

    fake_resolution = DrugTermResolution(
        input_term="Glucophage",
        status="resolved",
        corpus_concept="metformin",
        corpus_matches=["metformin"],
        selected_candidate=selected_candidate,
        candidates=[],
    )

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        lambda term: fake_resolution,
    )

    result = module.resolve_optional_drug_filter("Glucophage")

    assert result.status == "resolved"
    assert result.retrieval_drug == "metformin"
    assert result.corpus_matches == ["metformin"]
    assert result.selected_candidate is not None
    assert result.selected_candidate.name == "Glucophage"


def test_unknown_rxnorm_drug_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    fake_resolution = DrugTermResolution(
        input_term="not-a-drug",
        status="no_rxnorm_match",
        corpus_concept=None,
        corpus_matches=[],
        selected_candidate=None,
        candidates=[],
    )

    monkeypatch.setattr(
        module,
        "resolve_drug_term",
        lambda term: fake_resolution,
    )

    result = module.resolve_optional_drug_filter("not-a-drug")

    assert result.status == "no_rxnorm_match"
    assert result.retrieval_drug is None
    assert result.can_retrieve is False