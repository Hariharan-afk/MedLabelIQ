from __future__ import annotations

import medlabeliq.orchestration.qa_workflow as workflow
from medlabeliq.generation.answer_generator import GeneratedAnswer
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack
from medlabeliq.orchestration.drug_filter_resolution import DrugFilterResolution
from medlabeliq.orchestration.drug_mention_detection import DrugMentionDetection


def make_generated_answer() -> GeneratedAnswer:
    answer = GroundedAnswer(
        status="answered",
        answer="Yes. Evidence supports the answer.",
        citations=["E1"],
        evidence_summary="Test evidence summary.",
        safety_note="Test safety note.",
    )

    return GeneratedAnswer(
        evidence_pack=EvidencePack(
            query="test query",
            concept_name="metformin",
            retrieval_family=None,
            evidence_items=[],
        ),
        answer=answer,
        raw_model_output=None,
    )


def test_workflow_passes_resolved_drug_to_answer_query(monkeypatch) -> None:
    drug_resolution = DrugFilterResolution(
        requested_drug="Glucophage",
        status="resolved",
        retrieval_drug="metformin",
        corpus_matches=["metformin"],
        selected_candidate=None,
        rxnorm_resolution=None,
    )

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: drug_resolution,
    )

    captured: dict[str, str | None] = {}

    def fake_answer_query(
        query: str,
        *,
        concept_name: str | None = None,
        retrieval_family: str | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        captured["concept_name"] = concept_name
        return make_generated_answer()

    monkeypatch.setattr(
        workflow,
        "answer_query",
        fake_answer_query,
    )

    result = workflow.answer_query_with_drug_resolution(
        "Can Glucophage cause lactic acidosis?",
        requested_drug="Glucophage",
    )

    assert captured["concept_name"] == "metformin"
    assert result.drug_resolution.retrieval_drug == "metformin"
    assert result.generated.answer.status == "answered"


def test_workflow_fails_closed_for_unresolved_requested_drug(monkeypatch) -> None:
    drug_resolution = DrugFilterResolution(
        requested_drug="not-a-drug",
        status="no_rxnorm_match",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
    )

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: drug_resolution,
    )

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("answer_query should not run for unresolved drug filters.")

    monkeypatch.setattr(
        workflow,
        "answer_query",
        should_not_be_called,
    )

    result = workflow.answer_query_with_drug_resolution(
        "Does this treat infection?",
        requested_drug="not-a-drug",
    )

    assert result.generated.answer.status == "insufficient_evidence"
    assert result.generated.evidence_pack.evidence_items == []
    assert result.drug_resolution.can_retrieve is False

def test_workflow_uses_auto_detected_drug_when_no_explicit_filter(monkeypatch) -> None:
    drug_resolution = DrugFilterResolution(
        requested_drug=None,
        status="not_requested",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
    )

    drug_detection = DrugMentionDetection(
        query="Can Glucophage cause lactic acidosis?",
        status="rxnorm_resolved_query_mention",
        detected_mention="Glucophage",
        retrieval_drug="metformin",
        corpus_matches=["metformin"],
        selected_candidate=None,
        candidate_resolutions=[],
    )

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: drug_resolution,
    )

    monkeypatch.setattr(
        workflow,
        "detect_drug_mention_from_query",
        lambda query: drug_detection,
    )

    captured: dict[str, str | None] = {}

    def fake_answer_query(
        query: str,
        *,
        concept_name: str | None = None,
        retrieval_family: str | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        captured["concept_name"] = concept_name
        return make_generated_answer()

    monkeypatch.setattr(
        workflow,
        "answer_query",
        fake_answer_query,
    )

    result = workflow.answer_query_with_drug_resolution(
        "Can Glucophage cause lactic acidosis?"
    )

    assert captured["concept_name"] == "metformin"
    assert result.retrieval_drug == "metformin"
    assert result.drug_mention_detection.status == "rxnorm_resolved_query_mention"


def test_workflow_falls_back_to_broad_retrieval_when_no_drug_detected(monkeypatch) -> None:
    drug_resolution = DrugFilterResolution(
        requested_drug=None,
        status="not_requested",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
    )

    drug_detection = DrugMentionDetection(
        query="Can this medication cause side effects?",
        status="no_mention_detected",
        detected_mention=None,
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        candidate_resolutions=[],
    )

    monkeypatch.setattr(
        workflow,
        "resolve_optional_drug_filter",
        lambda requested_drug: drug_resolution,
    )

    monkeypatch.setattr(
        workflow,
        "detect_drug_mention_from_query",
        lambda query: drug_detection,
    )

    captured: dict[str, str | None] = {}

    def fake_answer_query(
        query: str,
        *,
        concept_name: str | None = None,
        retrieval_family: str | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        captured["concept_name"] = concept_name
        return make_generated_answer()

    monkeypatch.setattr(
        workflow,
        "answer_query",
        fake_answer_query,
    )

    result = workflow.answer_query_with_drug_resolution(
        "Can this medication cause side effects?"
    )

    assert captured["concept_name"] is None
    assert result.retrieval_drug is None
    assert result.drug_mention_detection.status == "no_mention_detected"