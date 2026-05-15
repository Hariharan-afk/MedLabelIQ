from __future__ import annotations

from fastapi.testclient import TestClient

import medlabeliq.api.app as api_app
from medlabeliq.api.app import app
from medlabeliq.generation.answer_generator import GeneratedAnswer
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack
from medlabeliq.orchestration.drug_filter_resolution import (
    DrugFilterResolution,
)
from medlabeliq.orchestration.drug_mention_detection import (
    DrugMentionDetection,
)
from medlabeliq.orchestration.qa_workflow import QAWorkflowResult
from medlabeliq.orchestration.retrieval_family_planner import (
    RetrievalFamilyPlan,
)
from medlabeliq.orchestration.source_router import SourceRoutePlan
from medlabeliq.rxnorm.identity_models import RxNormIdentityEvidence


client = TestClient(app)


def test_answer_endpoint_serializes_identity_evidence(monkeypatch) -> None:
    answer = GroundedAnswer(
        status="answered",
        answer="Yes. Eliquis and apixaban map to the same ingredient.",
        citations=["R1", "R2"],
        evidence_summary="Shared RxNorm ingredient mapping.",
        safety_note="RxNorm identity safety note.",
    )

    generated = GeneratedAnswer(
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

    identity_evidence = [
        RxNormIdentityEvidence(
            evidence_id="R1",
            term="Eliquis",
            resolution_status="resolved",
            selected_candidate=None,
            related_ingredients=[],
            related_brands=[],
            summary="RxNorm evidence for Eliquis.",
        )
    ]

    monkeypatch.setattr(
        api_app,
        "answer_query_with_drug_resolution",
        lambda **kwargs: QAWorkflowResult(
            generated=generated,
            drug_resolution=drug_resolution,
            drug_mention_detection=detection,
            retrieval_drug="apixaban",
            family_plan=family_plan,
            retrieval_family=None,
            source_plan=source_plan,
            identity_evidence=identity_evidence,
        ),
    )

    monkeypatch.setattr(
        api_app,
        "log_qa_interaction",
        lambda **kwargs: "identity-request-log-id",
    )

    response = client.post(
        "/qa/answer",
        json={
            "query": "Is Eliquis the same as apixaban?",
            "include_evidence": True,
            "include_diagnostics": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["planned_source"] == "rxnorm_identity"
    assert payload["request_log_id"] == "identity-request-log-id"
    assert payload["result"]["status"] == "answered"
    assert payload["result"]["citations"] == ["R1", "R2"]
    assert payload["identity_evidence"][0]["evidence_id"] == "R1"
    assert payload["identity_evidence"][0]["term"] == "Eliquis"