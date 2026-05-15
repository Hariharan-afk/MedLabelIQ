from __future__ import annotations

from fastapi.testclient import TestClient

from medlabeliq.api.app import app
import medlabeliq.api.app as api_app
from medlabeliq.api.schemas import HealthComponentResponse
from medlabeliq.generation.answer_generator import (
    APPLICATION_SAFETY_NOTE,
    GeneratedAnswer,
)
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidenceItem, EvidencePack
from medlabeliq.generation.verification_schema import AnswerVerification


client = TestClient(app)


def make_fake_evidence_pack() -> EvidencePack:
    return EvidencePack(
        query="Can metformin cause lactic acidosis?",
        concept_name="metformin",
        retrieval_family="warnings_and_precautions",
        evidence_items=[
            EvidenceItem(
                evidence_id="E1",
                chunk_id="22d7d2e6-2af5-59af-b930-0a10417ee027",
                section_id="1ada10ad-d795-50ea-a7b8-b53309194ce5",
                concept_name="metformin",
                retrieval_family="warnings_and_precautions",
                canonical_section_name=None,
                nearest_canonical_section_name="warnings_and_precautions",
                heading_path=[
                    "5 WARNINGS AND PRECAUTIONS",
                    "5.1 Lactic Acidosis",
                ],
                set_id="test-set-id",
                version_number=15,
                chunk_text=(
                    "Metformin-associated lactic acidosis has occurred."
                ),
                hybrid_score=0.01,
                lexical_rank=None,
                dense_rank=1,
            ),
        ],
    )


def make_fake_generated_answer() -> GeneratedAnswer:
    answer = GroundedAnswer(
        status="answered",
        answer="Yes. Metformin can cause lactic acidosis.",
        citations=["E1"],
        evidence_summary="The cited warning section directly supports the answer.",
        safety_note=APPLICATION_SAFETY_NOTE,
    )

    verification = AnswerVerification(
        verdict="supported",
        rationale="The cited evidence directly supports the answer.",
        cited_evidence_used=["E1"],
    )

    return GeneratedAnswer(
        evidence_pack=make_fake_evidence_pack(),
        answer=answer,
        raw_model_output='{"status":"answered"}',
        verification=verification,
        proposed_answer=answer,
        verification_overrode_answer=False,
        guardrail_triggered=False,
        guardrail_reason=None,
    )


def test_health_endpoint_ok(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "check_postgres",
        lambda: HealthComponentResponse(
            status="ok",
            detail="Postgres test ok.",
        ),
    )

    monkeypatch.setattr(
        api_app,
        "check_qdrant",
        lambda: HealthComponentResponse(
            status="ok",
            detail="Qdrant test ok.",
        ),
    )

    monkeypatch.setattr(
        api_app,
        "check_llm_configuration",
        lambda: HealthComponentResponse(
            status="ok",
            detail="LLM configured.",
        ),
    )

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["postgres"]["status"] == "ok"
    assert payload["qdrant"]["status"] == "ok"
    assert payload["llm"]["status"] == "ok"


def test_answer_endpoint_returns_grounded_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "answer_query",
        lambda **kwargs: make_fake_generated_answer(),
    )

    monkeypatch.setattr(
        api_app,
        "log_qa_interaction",
        lambda **kwargs: "test-request-log-id",
    )

    response = client.post(
        "/qa/answer",
        json={
            "query": "Can metformin cause lactic acidosis?",
            "drug": "metformin",
            "family": "warnings_and_precautions",
            "include_evidence": True,
            "include_diagnostics": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["request_log_id"] == "test-request-log-id"
    assert payload["result"]["status"] == "answered"
    assert payload["result"]["citations"] == ["E1"]

    assert len(payload["evidence"]) == 1
    assert payload["evidence"][0]["evidence_id"] == "E1"

    assert payload["diagnostics"]["verification"]["verdict"] == "supported"
    assert payload["diagnostics"]["guardrail_triggered"] is False


def test_retrieval_debug_endpoint_returns_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "build_evidence_pack",
        lambda **kwargs: make_fake_evidence_pack(),
    )

    response = client.post(
        "/retrieval/debug",
        json={
            "query": "lactic acidosis",
            "drug": "metformin",
            "family": "warnings_and_precautions",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["evidence_count"] == 1
    assert payload["evidence"][0]["evidence_id"] == "E1"
    assert payload["evidence"][0]["drug"] == "metformin"