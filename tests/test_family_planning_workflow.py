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


def make_generated_answer() -> GeneratedAnswer:
    answer = GroundedAnswer(
        status="answered",
        answer="Test answer.",
        citations=["E1"],
        evidence_summary="Test evidence.",
        safety_note="Test safety note.",
    )

    return GeneratedAnswer(
        evidence_pack=EvidencePack(
            query="test",
            concept_name="omeprazole",
            retrieval_family="indications_and_usage",
            evidence_items=[],
        ),
        answer=answer,
        raw_model_output=None,
    )


def make_not_requested_drug_resolution() -> DrugFilterResolution:
    return DrugFilterResolution(
        requested_drug=None,
        status="not_requested",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
    )


def make_no_detection(query: str) -> DrugMentionDetection:
    return DrugMentionDetection(
        query=query,
        status="no_mention_detected",
        detected_mention=None,
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        candidate_resolutions=[],
    )


def test_workflow_uses_planned_family_when_no_explicit_family(monkeypatch) -> None:
    query = "What is omeprazole used for?"

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: make_not_requested_drug_resolution(),
    )

    monkeypatch.setattr(
        workflow,
        "detect_drug_mention_from_query",
        lambda query: make_no_detection(query),
    )

    family_plan = RetrievalFamilyPlan(
        query=query,
        status="routed_single_family",
        intent="indication_or_use",
        planned_family="indications_and_usage",
        candidate_families=["indications_and_usage"],
        matches=[],
    )

    monkeypatch.setattr(
        workflow,
        "plan_retrieval_family",
        lambda query, requested_family=None: family_plan,
    )

    captured: dict[str, str | None] = {}

    def fake_answer_query(
        query: str,
        *,
        concept_name: str | None = None,
        retrieval_family: str | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        captured["retrieval_family"] = retrieval_family
        return make_generated_answer()

    monkeypatch.setattr(
        workflow,
        "answer_query",
        fake_answer_query,
    )

    result = workflow.answer_query_with_drug_resolution(query)

    assert captured["retrieval_family"] == "indications_and_usage"
    assert result.retrieval_family == "indications_and_usage"
    assert result.family_plan is not None
    assert result.family_plan.status == "routed_single_family"


def test_workflow_keeps_requested_explicit_family(monkeypatch) -> None:
    query = "What is omeprazole used for?"

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: make_not_requested_drug_resolution(),
    )

    monkeypatch.setattr(
        workflow,
        "detect_drug_mention_from_query",
        lambda query: make_no_detection(query),
    )

    explicit_plan = RetrievalFamilyPlan(
        query=query,
        status="not_attempted_explicit_family_present",
        intent=None,
        planned_family=None,
        candidate_families=[],
        matches=[],
    )

    monkeypatch.setattr(
        workflow,
        "plan_retrieval_family",
        lambda query, requested_family=None: explicit_plan,
    )

    captured: dict[str, str | None] = {}

    def fake_answer_query(
        query: str,
        *,
        concept_name: str | None = None,
        retrieval_family: str | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        captured["retrieval_family"] = retrieval_family
        return make_generated_answer()

    monkeypatch.setattr(
        workflow,
        "answer_query",
        fake_answer_query,
    )

    result = workflow.answer_query_with_drug_resolution(
        query,
        retrieval_family="clinical_studies",
    )

    assert captured["retrieval_family"] == "clinical_studies"
    assert result.retrieval_family == "clinical_studies"
    assert result.family_plan is not None
    assert (
        result.family_plan.status
        == "not_attempted_explicit_family_present"
    )