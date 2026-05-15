from __future__ import annotations

import medlabeliq.orchestration.mixed_source_composition as module
from medlabeliq.generation.answer_generator import GeneratedAnswer
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack
from medlabeliq.orchestration.source_router import (
    SourceRoutePlan,
    SourceRouteSignalMatch,
)
from medlabeliq.rxnorm.identity_answer import (
    RxNormIdentityAnswerResult,
)
from medlabeliq.rxnorm.identity_models import RxNormIdentityEvidence


def test_decomposition_rewrites_pronoun_to_retrieval_drug() -> None:
    metadata = module.decompose_mixed_source_query(
        "Is Eliquis the same as apixaban and can it prevent stroke?",
        retrieval_drug="apixaban",
        identity_intent="brand_generic_equivalence",
    )

    assert metadata.status == "composed_answered"
    assert metadata.identity_query == "Is Eliquis the same as apixaban?"
    assert metadata.clinical_query == "Can apixaban prevent stroke?"


def test_decomposition_rewrites_used_for_clause() -> None:
    metadata = module.decompose_mixed_source_query(
        "Is Glucophage the same as metformin and what is it used for?",
        retrieval_drug="metformin",
        identity_intent="brand_generic_equivalence",
    )

    assert metadata.status == "composed_answered"
    assert metadata.identity_query == "Is Glucophage the same as metformin?"
    assert metadata.clinical_query == "What is metformin used for?"


def test_mixed_execution_combines_identity_and_label_citations(monkeypatch) -> None:
    source_plan = SourceRoutePlan(
        query="Is Eliquis the same as apixaban and can it prevent stroke?",
        status="ambiguous_mixed_source",
        selected_source="multi_source_composed",
        intent=None,
        candidate_sources=[
            "rxnorm_identity",
            "dailymed_label",
        ],
        matches=[
            SourceRouteSignalMatch(
                source="rxnorm_identity",
                intent="brand_generic_equivalence",
                score=6,
                matched_signals=["same as"],
            )
        ],
    )

    identity_answer = GroundedAnswer(
        status="answered",
        answer="Yes. Eliquis and apixaban share the same ingredient.",
        citations=["R1", "R2"],
        evidence_summary="Shared RxNorm ingredient mapping.",
        safety_note="identity note",
    )

    identity_generated = GeneratedAnswer(
        evidence_pack=EvidencePack(
            query="identity",
            concept_name=None,
            retrieval_family=None,
            evidence_items=[],
        ),
        answer=identity_answer,
        raw_model_output=None,
        proposed_answer=identity_answer,
    )

    identity_evidence = [
        RxNormIdentityEvidence(
            evidence_id="R1",
            term="Eliquis",
            resolution_status="resolved",
            selected_candidate=None,
            related_ingredients=[],
            related_brands=[],
            summary="Identity evidence 1.",
        ),
        RxNormIdentityEvidence(
            evidence_id="R2",
            term="apixaban",
            resolution_status="resolved",
            selected_candidate=None,
            related_ingredients=[],
            related_brands=[],
            summary="Identity evidence 2.",
        ),
    ]

    clinical_answer = GroundedAnswer(
        status="answered",
        answer="Apixaban is labeled to reduce the risk of stroke.",
        citations=["E1"],
        evidence_summary="DailyMed indication evidence.",
        safety_note="label note",
    )

    clinical_generated = GeneratedAnswer(
        evidence_pack=EvidencePack(
            query="clinical",
            concept_name="apixaban",
            retrieval_family="indications_and_usage",
            evidence_items=[],
        ),
        answer=clinical_answer,
        raw_model_output=None,
        proposed_answer=clinical_answer,
    )

    monkeypatch.setattr(
        module,
        "answer_rxnorm_identity_query",
        lambda query, intent: RxNormIdentityAnswerResult(
            generated=identity_generated,
            evidence_items=identity_evidence,
        ),
    )

    monkeypatch.setattr(
        module,
        "answer_query",
        lambda **kwargs: clinical_generated,
    )

    result = module.execute_mixed_source_composition(
        query="Is Eliquis the same as apixaban and can it prevent stroke?",
        source_plan=source_plan,
        retrieval_drug="apixaban",
        retrieval_family="indications_and_usage",
        top_k=None,
    )

    assert result.generated.answer.status == "answered"
    assert result.generated.answer.citations == ["R1", "R2", "E1"]
    assert result.metadata.status == "composed_answered"
    assert result.metadata.clinical_query == "Can apixaban prevent stroke?"
    assert result.identity_evidence == identity_evidence