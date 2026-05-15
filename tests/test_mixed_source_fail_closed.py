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
from medlabeliq.orchestration.mixed_source_composition import (
    MixedSourceCompositionExecution,
    MixedSourceCompositionMetadata,
)
from medlabeliq.orchestration.retrieval_family_planner import (
    RetrievalFamilyPlan,
)
from medlabeliq.orchestration.source_router import SourceRoutePlan
from medlabeliq.rxnorm.identity_models import RxNormIdentityEvidence


def test_ambiguous_mixed_source_executes_composition(monkeypatch) -> None:
    query = "Is Eliquis the same as apixaban and can it prevent stroke?"

    drug_resolution = DrugFilterResolution(
        requested_drug=None,
        status="not_requested",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
    )

    drug_detection = DrugMentionDetection(
        query=query,
        status="direct_corpus_mention",
        detected_mention="apixaban",
        retrieval_drug="apixaban",
        corpus_matches=["apixaban"],
        selected_candidate=None,
        candidate_resolutions=[],
    )

    family_plan = RetrievalFamilyPlan(
        query=query,
        status="routed_single_family",
        intent="indication_or_use",
        planned_family="indications_and_usage",
        candidate_families=["indications_and_usage"],
        matches=[],
    )

    source_plan = SourceRoutePlan(
        query=query,
        status="ambiguous_mixed_source",
        selected_source="multi_source_composed",
        intent=None,
        candidate_sources=[
            "rxnorm_identity",
            "dailymed_label",
        ],
        matches=[],
    )

    answer = GroundedAnswer(
        status="answered",
        answer=(
            "Yes. Eliquis and apixaban share an ingredient. "
            "Apixaban is labeled to reduce stroke risk."
        ),
        citations=["R1", "E1"],
        evidence_summary="Combined RxNorm and DailyMed support.",
        safety_note="mixed safety note",
    )

    generated = GeneratedAnswer(
        evidence_pack=EvidencePack(
            query=query,
            concept_name="apixaban",
            retrieval_family="indications_and_usage",
            evidence_items=[],
        ),
        answer=answer,
        raw_model_output=None,
        proposed_answer=answer,
    )

    mixed_metadata = MixedSourceCompositionMetadata(
        status="composed_answered",
        identity_query="Is Eliquis the same as apixaban?",
        clinical_query="Can apixaban prevent stroke?",
        identity_intent="brand_generic_equivalence",
    )

    identity_evidence = [
        RxNormIdentityEvidence(
            evidence_id="R1",
            term="Eliquis",
            resolution_status="resolved",
            selected_candidate=None,
            related_ingredients=[],
            related_brands=[],
            summary="Identity evidence.",
        )
    ]

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: drug_resolution,
    )

    monkeypatch.setattr(
        workflow,
        "resolve_effective_family",
        lambda query, requested_family: (
            family_plan,
            "indications_and_usage",
        ),
    )

    monkeypatch.setattr(
        workflow,
        "detect_drug_mention_from_query",
        lambda query: drug_detection,
    )

    monkeypatch.setattr(
        workflow,
        "plan_source_route",
        lambda *args, **kwargs: source_plan,
    )

    monkeypatch.setattr(
        workflow,
        "execute_mixed_source_composition",
        lambda **kwargs: MixedSourceCompositionExecution(
            generated=generated,
            identity_evidence=identity_evidence,
            metadata=mixed_metadata,
        ),
    )

    result = workflow.answer_query_with_drug_resolution(query)

    assert result.generated.answer.status == "answered"
    assert result.source_plan is not None
    assert result.source_plan.status == "ambiguous_mixed_source"
    assert result.source_plan.selected_source == "multi_source_composed"
    assert result.retrieval_family == "indications_and_usage"
    assert result.identity_evidence == identity_evidence
    assert result.mixed_source_composition == mixed_metadata