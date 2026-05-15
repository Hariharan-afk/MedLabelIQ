from __future__ import annotations

import medlabeliq.rxnorm.resolver as resolver
from medlabeliq.rxnorm.models import RxNormConcept


class FakeRxNormClient:
    def __enter__(self) -> "FakeRxNormClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def find_rxcuis_by_string(
        self,
        term: str,
        *,
        search: int = 2,
    ) -> list[str]:
        if term == "metformin":
            return ["6809"]

        if term == "Glucophage":
            return ["brand-rxcui"]

        return []

    def get_approximate_matches(
        self,
        term: str,
        *,
        max_entries: int | None = None,
    ) -> list[dict]:
        if term == "metformn":
            return [
                {
                    "rxcui": "6809",
                    "score": "13.0",
                    "rank": "1",
                    "source": "RXNORM",
                }
            ]

        return []

    def get_concept_properties(
        self,
        rxcui: str,
    ) -> dict | None:
        if rxcui == "6809":
            return {
                "rxcui": "6809",
                "name": "metformin",
                "synonym": "",
                "tty": "IN",
            }

        if rxcui == "brand-rxcui":
            return {
                "rxcui": "brand-rxcui",
                "name": "Glucophage",
                "synonym": "",
                "tty": "BN",
            }

        return None

    def get_related_by_type(
        self,
        rxcui: str,
        *,
        term_types,
    ) -> list[dict]:
        if rxcui == "brand-rxcui":
            return [
                {
                    "rxcui": "6809",
                    "name": "metformin",
                    "synonym": "",
                    "tty": "IN",
                }
            ]

        return []


def test_resolve_direct_ingredient_to_corpus(monkeypatch) -> None:
    monkeypatch.setattr(
        resolver,
        "RxNormClient",
        FakeRxNormClient,
    )

    monkeypatch.setattr(
        resolver,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    result = resolver.resolve_drug_term("metformin")

    assert result.status == "resolved"
    assert result.corpus_concept == "metformin"
    assert result.corpus_matches == ["metformin"]
    assert result.selected_candidate is not None
    assert result.selected_candidate.rxcui == "6809"


def test_resolve_brand_to_ingredient_corpus_match(monkeypatch) -> None:
    monkeypatch.setattr(
        resolver,
        "RxNormClient",
        FakeRxNormClient,
    )

    monkeypatch.setattr(
        resolver,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    result = resolver.resolve_drug_term("Glucophage")

    assert result.status == "resolved"
    assert result.corpus_concept == "metformin"
    assert result.corpus_matches == ["metformin"]
    assert result.selected_candidate is not None
    assert result.selected_candidate.name == "Glucophage"


def test_resolve_misspelling_via_approximate_match(monkeypatch) -> None:
    monkeypatch.setattr(
        resolver,
        "RxNormClient",
        FakeRxNormClient,
    )

    monkeypatch.setattr(
        resolver,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    result = resolver.resolve_drug_term("metformn")

    assert result.status == "resolved"
    assert result.corpus_concept == "metformin"
    assert result.corpus_matches == ["metformin"]
    assert result.selected_candidate is not None
    assert result.selected_candidate.match_method == "approximate"


def test_no_rxnorm_match(monkeypatch) -> None:
    monkeypatch.setattr(
        resolver,
        "RxNormClient",
        FakeRxNormClient,
    )

    monkeypatch.setattr(
        resolver,
        "list_indexed_corpus_drugs",
        lambda: ["metformin", "apixaban"],
    )

    result = resolver.resolve_drug_term("definitely-not-a-drug")

    assert result.status == "no_rxnorm_match"
    assert result.corpus_concept is None
    assert result.corpus_matches == []