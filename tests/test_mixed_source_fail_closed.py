from __future__ import annotations

import medlabeliq.orchestration.qa_workflow as workflow
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


def test_ambiguous_mixed_source_fails_closed(monkeypatch) -> None:
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
        selected_source="dailymed_label",
        intent=None,
        candidate_sources=[
            "rxnorm_identity",
            "dailymed_label",
        ],
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
        "answer_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DailyMed answer_query should not run.")
        ),
    )

    monkeypatch.setattr(
        workflow,
        "answer_rxnorm_identity_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RxNorm identity answer should not run.")
        ),
    )

    result = workflow.answer_query_with_drug_resolution(query)

    assert result.generated.answer.status == "insufficient_evidence"
    assert result.source_plan is not None
    assert result.source_plan.status == "ambiguous_mixed_source"
    assert result.retrieval_family == "indications_and_usage"
    assert result.identity_evidence is None