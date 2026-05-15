from __future__ import annotations

import medlabeliq.rxnorm.identity_answer as module
from medlabeliq.rxnorm.identity_models import (
    RxNormIdentityTermResolution,
)
from medlabeliq.rxnorm.models import RxNormConcept


def concept(
    rxcui: str,
    name: str,
    tty: str,
) -> RxNormConcept:
    return RxNormConcept(
        rxcui=rxcui,
        name=name,
        synonym="",
        tty=tty,
    )


def resolved_term(
    *,
    term: str,
    selected: RxNormConcept,
    ingredients: list[RxNormConcept],
    brands: list[RxNormConcept] | None = None,
) -> RxNormIdentityTermResolution:
    return RxNormIdentityTermResolution(
        input_term=term,
        status="resolved",
        selected_candidate=selected,
        candidates=[selected],
        related_ingredients=ingredients,
        related_brands=brands or [],
    )


def test_equivalence_answer_uses_shared_ingredient(monkeypatch) -> None:
    apixaban = concept("1364430", "apixaban", "IN")
    eliquis = concept("1364447", "Eliquis", "BN")

    resolutions = {
        "Eliquis": resolved_term(
            term="Eliquis",
            selected=eliquis,
            ingredients=[apixaban],
            brands=[eliquis],
        ),
        "apixaban": resolved_term(
            term="apixaban",
            selected=apixaban,
            ingredients=[apixaban],
        ),
    }

    monkeypatch.setattr(
        module,
        "resolve_identity_term",
        lambda term: resolutions[term],
    )

    result = module.answer_rxnorm_identity_query(
        "Is Eliquis the same as apixaban?",
        intent="brand_generic_equivalence",
    )

    assert result.generated.answer.status == "answered"
    assert "Yes." in result.generated.answer.answer
    assert "apixaban" in result.generated.answer.answer
    assert result.generated.answer.citations == ["R1", "R2"]
    assert len(result.evidence_items) == 2


def test_generic_name_answer_returns_ingredient(monkeypatch) -> None:
    metformin = concept("6809", "metformin", "IN")
    glucophage = concept("151827", "Glucophage", "BN")

    monkeypatch.setattr(
        module,
        "resolve_identity_term",
        lambda term: resolved_term(
            term=term,
            selected=glucophage,
            ingredients=[metformin],
            brands=[glucophage],
        ),
    )

    result = module.answer_rxnorm_identity_query(
        "What is the generic name of Glucophage?",
        intent="generic_name_lookup",
    )

    assert result.generated.answer.status == "answered"
    assert "metformin" in result.generated.answer.answer
    assert result.generated.answer.citations == ["R1"]
    assert len(result.evidence_items) == 1


def test_definition_answer_describes_brand_and_ingredient(monkeypatch) -> None:
    metformin = concept("6809", "metformin", "IN")
    glucophage = concept("151827", "Glucophage", "BN")

    monkeypatch.setattr(
        module,
        "resolve_identity_term",
        lambda term: resolved_term(
            term=term,
            selected=glucophage,
            ingredients=[metformin],
            brands=[glucophage],
        ),
    )

    result = module.answer_rxnorm_identity_query(
        "What is Glucophage?",
        intent="drug_identity_definition",
    )

    assert result.generated.answer.status == "answered"
    assert "brand-name medication concept" in result.generated.answer.answer
    assert "metformin" in result.generated.answer.answer
    assert result.generated.answer.citations == ["R1"]