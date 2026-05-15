from __future__ import annotations

import medlabeliq.orchestration.qa_workflow as workflow
from medlabeliq.generation.answer_generator import GeneratedAnswer
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack
from medlabeliq.orchestration.drug_filter_resolution import (
    DrugFilterResolution,
)
from medlabeliq.orchestration.drug_mention_detection import (
    DrugMentionDetection,
)
from medlabeliq.orchestration.retrieval_family_planner import (
    RetrievalFamilyPlan,
)
from medlabeliq.orchestration.source_router import SourceRoutePlan
from medlabeliq.rxnorm.identity_answer import (
    RxNormIdentityAnswerResult,
)
from medlabeliq.rxnorm.identity_models import RxNormIdentityEvidence


def make_generated_identity_answer() -> GeneratedAnswer:
    answer = GroundedAnswer(
        status="answered",
        answer="Yes. Eliquis and apixaban map to the same ingredient.",
        citations=["R1", "R2"],
        evidence_summary="Shared RxNorm ingredient mapping.",
        safety_note="RxNorm identity safety note.",
    )

    return GeneratedAnswer(
        evidence_pack=EvidencePack(
            query="Is Eliquis the same as apixaban?",
            concept_name=None,
            retrieval_family=None,
            evidence_items=[],
        ),
        answer=answer,
        raw_model_output=None,
        proposed_answer=answer,
    )


def test_workflow_executes_rxnorm_identity_branch(monkeypatch) -> None:
    drug_resolution = DrugFilterResolution(
        requested_drug=None,
        status="not_requested",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
    )

    detection = DrugMentionDetection(
        query="Is Eliquis the same as apixaban?",
        status="direct_corpus_mention",
        detected_mention="apixaban",
        retrieval_drug="apixaban",
        corpus_matches=["apixaban"],
        selected_candidate=None,
        candidate_resolutions=[],
    )

    family_plan = RetrievalFamilyPlan(
        query="Is Eliquis the same as apixaban?",
        status="no_route_detected",
        intent=None,
        planned_family=None,
        candidate_families=[],
        matches=[],
    )

    source_plan = SourceRoutePlan(
        query="Is Eliquis the same as apixaban?",
        status="routed_rxnorm_identity",
        selected_source="rxnorm_identity",
        intent="brand_generic_equivalence",
        candidate_sources=["rxnorm_identity"],
        matches=[],
    )

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: drug_resolution,
    )

    monkeypatch.setattr(
        workflow,
        "resolve_effective_family",
        lambda query, requested_family: (family_plan, None),
    )

    monkeypatch.setattr(
        workflow,
        "detect_drug_mention_from_query",
        lambda query: detection,
    )

    monkeypatch.setattr(
        workflow,
        "plan_source_route",
        lambda *args, **kwargs: source_plan,
    )

    monkeypatch.setattr(
        workflow,
        "answer_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DailyMed answer_query should not run.")
        ),
    )

    identity_evidence = [
        RxNormIdentityEvidence(
            evidence_id="R1",
            term="Eliquis",
            resolution_status="resolved",
            selected_candidate=None,
            related_ingredients=[],
            related_brands=[],
            summary="Test RxNorm evidence.",
        )
    ]

    monkeypatch.setattr(
        workflow,
        "answer_rxnorm_identity_query",
        lambda query, intent: RxNormIdentityAnswerResult(
            generated=make_generated_identity_answer(),
            evidence_items=identity_evidence,
        ),
    )

    result = workflow.answer_query_with_drug_resolution(
        "Is Eliquis the same as apixaban?"
    )

    assert result.generated.answer.status == "answered"
    assert result.source_plan is not None
    assert result.source_plan.selected_source == "rxnorm_identity"
    assert result.identity_evidence == identity_evidence