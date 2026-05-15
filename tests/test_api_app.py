from __future__ import annotations

from fastapi.testclient import TestClient

import medlabeliq.api.app as api_app
from medlabeliq.api.app import app
from medlabeliq.api.schemas import HealthComponentResponse
from medlabeliq.generation.answer_generator import (
    APPLICATION_SAFETY_NOTE,
    GeneratedAnswer,
)
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidenceItem, EvidencePack
from medlabeliq.generation.verification_schema import AnswerVerification
from medlabeliq.orchestration.drug_filter_resolution import (
    DrugFilterResolution,
)
from medlabeliq.orchestration.drug_mention_detection import (
    DrugMentionDetection,
)
from medlabeliq.orchestration.qa_workflow import (
    QAWorkflowResult,
    RetrievalDebugWorkflowResult,
)
from medlabeliq.rxnorm.models import DrugTermResolution, RxNormConcept


client = TestClient(app)


# =============================================================================
# Shared fixtures/helpers
# =============================================================================

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


def make_explicit_filter_resolution() -> DrugFilterResolution:
    return DrugFilterResolution(
        requested_drug="Glucophage",
        status="resolved",
        retrieval_drug="metformin",
        corpus_matches=["metformin"],
        selected_candidate=None,
        rxnorm_resolution=None,
    )


def make_explicit_filter_detection() -> DrugMentionDetection:
    return DrugMentionDetection(
        query="Can Glucophage cause lactic acidosis?",
        status="not_attempted_explicit_filter_present",
        detected_mention=None,
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        candidate_resolutions=[],
    )


def make_auto_detection_result() -> DrugMentionDetection:
    return DrugMentionDetection(
        query="Can Glucophage cause lactic acidosis?",
        status="rxnorm_resolved_query_mention",
        detected_mention="Glucophage",
        retrieval_drug="metformin",
        corpus_matches=["metformin"],
        selected_candidate=None,
        candidate_resolutions=[],
    )


def make_not_requested_filter_resolution() -> DrugFilterResolution:
    return DrugFilterResolution(
        requested_drug=None,
        status="not_requested",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
    )


# =============================================================================
# Health endpoint
# =============================================================================

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


# =============================================================================
# QA endpoints
# =============================================================================

def test_answer_endpoint_returns_grounded_answer_for_explicit_drug_filter(
    monkeypatch,
) -> None:
    fake_resolution = make_explicit_filter_resolution()
    fake_detection = make_explicit_filter_detection()

    monkeypatch.setattr(
        api_app,
        "answer_query_with_drug_resolution",
        lambda **kwargs: QAWorkflowResult(
            generated=make_fake_generated_answer(),
            drug_resolution=fake_resolution,
            drug_mention_detection=fake_detection,
            retrieval_drug="metformin",
        ),
    )

    monkeypatch.setattr(
        api_app,
        "log_qa_interaction",
        lambda **kwargs: "test-request-log-id",
    )

    response = client.post(
        "/qa/answer",
        json={
            "query": "Can Glucophage cause lactic acidosis?",
            "drug": "Glucophage",
            "family": "warnings_and_precautions",
            "include_evidence": True,
            "include_diagnostics": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["request_log_id"] == "test-request-log-id"
    assert payload["drug"] == "Glucophage"
    assert payload["resolved_drug"] == "metformin"
    assert payload["result"]["status"] == "answered"
    assert payload["result"]["citations"] == ["E1"]

    assert len(payload["evidence"]) == 1
    assert payload["evidence"][0]["evidence_id"] == "E1"

    assert payload["diagnostics"]["verification"]["verdict"] == "supported"
    assert payload["diagnostics"]["guardrail_triggered"] is False

    assert payload["diagnostics"]["drug_resolution"]["status"] == "resolved"
    assert (
        payload["diagnostics"]["drug_resolution"]["retrieval_drug"]
        == "metformin"
    )

    assert (
        payload["diagnostics"]["drug_mention_detection"]["status"]
        == "not_attempted_explicit_filter_present"
    )
    assert (
        payload["diagnostics"]["drug_mention_detection"].get("detected_mention")
        is None
    )


def test_answer_endpoint_returns_grounded_answer_for_auto_detected_query_drug(
    monkeypatch,
) -> None:
    fake_resolution = make_not_requested_filter_resolution()
    fake_detection = make_auto_detection_result()

    monkeypatch.setattr(
        api_app,
        "answer_query_with_drug_resolution",
        lambda **kwargs: QAWorkflowResult(
            generated=make_fake_generated_answer(),
            drug_resolution=fake_resolution,
            drug_mention_detection=fake_detection,
            retrieval_drug="metformin",
        ),
    )

    monkeypatch.setattr(
        api_app,
        "log_qa_interaction",
        lambda **kwargs: "auto-detection-request-log-id",
    )

    response = client.post(
        "/qa/answer",
        json={
            "query": "Can Glucophage cause lactic acidosis?",
            "family": "warnings_and_precautions",
            "include_evidence": True,
            "include_diagnostics": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["request_log_id"] == "auto-detection-request-log-id"
    assert "drug" not in payload or payload["drug"] is None
    assert payload["resolved_drug"] == "metformin"
    assert payload["result"]["status"] == "answered"

    assert (
        payload["diagnostics"]["drug_resolution"]["status"]
        == "not_requested"
    )

    assert (
        payload["diagnostics"]["drug_mention_detection"]["status"]
        == "rxnorm_resolved_query_mention"
    )
    assert (
        payload["diagnostics"]["drug_mention_detection"]["detected_mention"]
        == "Glucophage"
    )
    assert (
        payload["diagnostics"]["drug_mention_detection"]["retrieval_drug"]
        == "metformin"
    )


# =============================================================================
# Retrieval debug endpoint
# =============================================================================

def test_retrieval_debug_endpoint_returns_evidence_for_explicit_drug_filter(
    monkeypatch,
) -> None:
    fake_resolution = make_explicit_filter_resolution()
    fake_detection = make_explicit_filter_detection()

    monkeypatch.setattr(
        api_app,
        "build_debug_evidence_pack_with_drug_resolution",
        lambda **kwargs: RetrievalDebugWorkflowResult(
            evidence_pack=make_fake_evidence_pack(),
            drug_resolution=fake_resolution,
            drug_mention_detection=fake_detection,
            retrieval_drug="metformin",
        ),
    )

    response = client.post(
        "/retrieval/debug",
        json={
            "query": "lactic acidosis",
            "drug": "Glucophage",
            "family": "warnings_and_precautions",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["drug"] == "Glucophage"
    assert payload["resolved_drug"] == "metformin"
    assert payload["evidence_count"] == 1
    assert payload["evidence"][0]["evidence_id"] == "E1"
    assert payload["evidence"][0]["drug"] == "metformin"

    assert payload["drug_resolution"]["status"] == "resolved"
    assert (
        payload["drug_mention_detection"]["status"]
        == "not_attempted_explicit_filter_present"
    )


def test_retrieval_debug_endpoint_returns_evidence_for_auto_detected_drug(
    monkeypatch,
) -> None:
    fake_resolution = make_not_requested_filter_resolution()
    fake_detection = make_auto_detection_result()

    monkeypatch.setattr(
        api_app,
        "build_debug_evidence_pack_with_drug_resolution",
        lambda **kwargs: RetrievalDebugWorkflowResult(
            evidence_pack=make_fake_evidence_pack(),
            drug_resolution=fake_resolution,
            drug_mention_detection=fake_detection,
            retrieval_drug="metformin",
        ),
    )

    response = client.post(
        "/retrieval/debug",
        json={
            "query": "Can Glucophage cause lactic acidosis?",
            "family": "warnings_and_precautions",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert "drug" not in payload or payload["drug"] is None
    assert payload["resolved_drug"] == "metformin"
    assert payload["evidence_count"] == 1
    assert payload["evidence"][0]["evidence_id"] == "E1"

    assert payload["drug_resolution"]["status"] == "not_requested"
    assert (
        payload["drug_mention_detection"]["status"]
        == "rxnorm_resolved_query_mention"
    )
    assert (
        payload["drug_mention_detection"]["detected_mention"]
        == "Glucophage"
    )


# =============================================================================
# Corpus metadata endpoints
# =============================================================================

def test_drugs_endpoint_returns_dynamic_drug_list(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "list_drug_summaries",
        lambda: [
            {
                "concept_name": "metformin",
                "label_count": 1,
                "label_version_count": 1,
                "section_count": 72,
                "chunk_count": 71,
            }
        ],
    )

    response = client.get("/drugs")

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 1
    assert payload["drugs"][0]["concept_name"] == "metformin"
    assert payload["drugs"][0]["chunk_count"] == 71


def test_families_endpoint_returns_dynamic_family_list(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "list_retrieval_family_summaries",
        lambda: [
            {
                "retrieval_family": "warnings_and_precautions",
                "section_count": 101,
                "chunk_count": 105,
                "drug_count": 10,
            }
        ],
    )

    response = client.get("/families")

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 1
    assert (
        payload["families"][0]["retrieval_family"]
        == "warnings_and_precautions"
    )
    assert payload["families"][0]["chunk_count"] == 105


def test_corpus_stats_endpoint_returns_live_stats(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "collect_corpus_stats",
        lambda: {
            "drug_count": 12,
            "label_document_count": 12,
            "label_version_count": 12,
            "product_count": 37,
            "ingredient_count": 396,
            "section_count": 663,
            "retrievable_section_count": 520,
            "chunk_count": 867,
            "retrieval_family_count": 20,
            "qdrant_collection": "medlabeliq_chunks",
            "qdrant_point_count": 867,
            "embedding_model_name": "BAAI/bge-small-en-v1.5",
            "latest_build": {
                "build_id": "test-build-id",
                "built_at": "2026-05-15T00:00:00+00:00",
                "build_source": "bootstrap",
                "seed_file_path": "data/seeds/smoke_set.yaml",
                "drug_count": 12,
                "label_document_count": 12,
                "label_version_count": 12,
                "product_count": 37,
                "ingredient_count": 396,
                "section_count": 663,
                "retrievable_section_count": 520,
                "chunk_count": 867,
                "retrieval_family_count": 20,
                "qdrant_collection": "medlabeliq_chunks",
                "qdrant_point_count": 867,
                "embedding_model_name": "BAAI/bge-small-en-v1.5",
            },
        },
    )

    response = client.get("/corpus/stats")

    assert response.status_code == 200

    payload = response.json()

    assert payload["drug_count"] == 12
    assert payload["chunk_count"] == 867
    assert payload["qdrant_point_count"] == 867
    assert payload["latest_build"]["build_source"] == "bootstrap"


# =============================================================================
# RxNorm endpoints
# =============================================================================

def test_normalize_drug_endpoint_returns_corpus_concept(monkeypatch) -> None:
    fake_candidate = RxNormConcept(
        rxcui="6809",
        name="metformin",
        synonym="",
        tty="IN",
        match_method="exact_or_normalized",
    )

    fake_resolution = DrugTermResolution(
        input_term="metformin",
        status="resolved",
        corpus_concept="metformin",
        corpus_matches=["metformin"],
        selected_candidate=fake_candidate,
        candidates=[],
    )

    monkeypatch.setattr(
        api_app,
        "resolve_drug_term",
        lambda term: fake_resolution,
    )

    response = client.post(
        "/normalize/drug",
        json={
            "term": "metformin",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "resolved"
    assert payload["corpus_concept"] == "metformin"
    assert payload["selected_candidate"]["rxcui"] == "6809"


def test_rxnorm_version_endpoint(monkeypatch) -> None:
    class FakeVersionClient:
        def __enter__(self) -> "FakeVersionClient":
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def get_version(self) -> dict[str, str]:
            return {
                "version": "04-May-2026",
                "api_version": "3.1.351",
            }

    monkeypatch.setattr(
        api_app,
        "RxNormClient",
        FakeVersionClient,
    )

    response = client.get("/rxnorm/version")

    assert response.status_code == 200

    payload = response.json()

    assert payload["version"] == "04-May-2026"
    assert payload["api_version"] == "3.1.351"