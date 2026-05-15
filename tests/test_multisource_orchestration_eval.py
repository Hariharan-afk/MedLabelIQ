from __future__ import annotations

import medlabeliq.evaluation.evaluate_multisource_orchestration as module
from medlabeliq.generation.answer_generator import (
    APPLICATION_SAFETY_NOTE,
    GeneratedAnswer,
)
from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidenceItem, EvidencePack
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
from medlabeliq.rxnorm.identity_answer import (
    RXNORM_IDENTITY_SAFETY_NOTE,
)
from medlabeliq.rxnorm.identity_models import RxNormIdentityEvidence


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


def make_detected_drug(
    query: str,
    *,
    mention: str,
    retrieval_drug: str,
) -> DrugMentionDetection:
    return DrugMentionDetection(
        query=query,
        status="direct_corpus_mention",
        detected_mention=mention,
        retrieval_drug=retrieval_drug,
        corpus_matches=[retrieval_drug],
        selected_candidate=None,
        candidate_resolutions=[],
    )


def make_label_evidence_pack() -> EvidencePack:
    return EvidencePack(
        query="What is omeprazole used for?",
        concept_name="omeprazole",
        retrieval_family="indications_and_usage",
        evidence_items=[
            EvidenceItem(
                evidence_id="E1",
                chunk_id="22d7d2e6-2af5-59af-b930-0a10417ee027",
                section_id="1ada10ad-d795-50ea-a7b8-b53309194ce5",
                concept_name="omeprazole",
                retrieval_family="indications_and_usage",
                canonical_section_name="indications_and_usage",
                nearest_canonical_section_name="indications_and_usage",
                heading_path=[
                    "1 INDICATIONS AND USAGE",
                ],
                set_id="test-set-id",
                version_number=1,
                chunk_text="Omeprazole is used for GERD.",
                hybrid_score=0.01,
                lexical_rank=None,
                dense_rank=1,
            )
        ],
    )


def test_evaluate_case_passes_dailymed_label_route(monkeypatch) -> None:
    case = module.MultiSourceEvalCase(
        id="label_test",
        query="What is omeprazole used for?",
        drug=None,
        family=None,
        expected_status="answered",
        expected_source="dailymed_label",
        expected_source_route_status="routed_dailymed_label",
        expected_family_plan_status="routed_single_family",
        expected_retrieval_family="indications_and_usage",
        expected_response_any=["GERD"],
        minimum_citations=1,
        citation_prefix="E",
        allowed_citation_prefixes=None,
        minimum_label_evidence=1,
        maximum_label_evidence=None,
        minimum_identity_evidence=0,
        maximum_identity_evidence=0,
    )

    answer = GroundedAnswer(
        status="answered",
        answer="Omeprazole is used for GERD.",
        citations=["E1"],
        evidence_summary="The cited label evidence supports GERD use.",
        safety_note=APPLICATION_SAFETY_NOTE,
    )

    generated = GeneratedAnswer(
        evidence_pack=make_label_evidence_pack(),
        answer=answer,
        raw_model_output=None,
        proposed_answer=answer,
    )

    family_plan = RetrievalFamilyPlan(
        query=case.query,
        status="routed_single_family",
        intent="indication_or_use",
        planned_family="indications_and_usage",
        candidate_families=["indications_and_usage"],
        matches=[],
    )

    source_plan = SourceRoutePlan(
        query=case.query,
        status="routed_dailymed_label",
        selected_source="dailymed_label",
        intent="indication_or_use",
        candidate_sources=["dailymed_label"],
        matches=[],
    )

    workflow_result = QAWorkflowResult(
        generated=generated,
        drug_resolution=make_not_requested_drug_resolution(),
        drug_mention_detection=make_detected_drug(
            case.query,
            mention="omeprazole",
            retrieval_drug="omeprazole",
        ),
        retrieval_drug="omeprazole",
        family_plan=family_plan,
        retrieval_family="indications_and_usage",
        source_plan=source_plan,
        identity_evidence=None,
    )

    monkeypatch.setattr(
        module,
        "answer_query_with_drug_resolution",
        lambda **kwargs: workflow_result,
    )

    row = module.evaluate_case(case)

    assert row["overall_pass"] is True
    assert row["source_pass"] is True
    assert row["citation_reference_pass"] is True
    assert row["minimum_label_evidence_pass"] is True


def test_evaluate_case_passes_rxnorm_identity_route(monkeypatch) -> None:
    case = module.MultiSourceEvalCase(
        id="identity_test",
        query="Is Eliquis the same as apixaban?",
        drug=None,
        family=None,
        expected_status="answered",
        expected_source="rxnorm_identity",
        expected_source_route_status="routed_rxnorm_identity",
        expected_family_plan_status="no_route_detected",
        expected_retrieval_family=None,
        expected_response_any=["apixaban"],
        minimum_citations=1,
        citation_prefix="R",
        allowed_citation_prefixes=None,
        minimum_label_evidence=0,
        maximum_label_evidence=0,
        minimum_identity_evidence=1,
        maximum_identity_evidence=None,
    )

    answer = GroundedAnswer(
        status="answered",
        answer="Yes. Eliquis and apixaban share the apixaban ingredient.",
        citations=["R1"],
        evidence_summary="Shared RxNorm ingredient mapping.",
        safety_note=RXNORM_IDENTITY_SAFETY_NOTE,
    )

    generated = GeneratedAnswer(
        evidence_pack=EvidencePack(
            query=case.query,
            concept_name=None,
            retrieval_family=None,
            evidence_items=[],
        ),
        answer=answer,
        raw_model_output=None,
        proposed_answer=answer,
    )

    family_plan = RetrievalFamilyPlan(
        query=case.query,
        status="no_route_detected",
        intent=None,
        planned_family=None,
        candidate_families=[],
        matches=[],
    )

    source_plan = SourceRoutePlan(
        query=case.query,
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
            summary="RxNorm identity evidence.",
        )
    ]

    workflow_result = QAWorkflowResult(
        generated=generated,
        drug_resolution=make_not_requested_drug_resolution(),
        drug_mention_detection=make_detected_drug(
            case.query,
            mention="apixaban",
            retrieval_drug="apixaban",
        ),
        retrieval_drug="apixaban",
        family_plan=family_plan,
        retrieval_family=None,
        source_plan=source_plan,
        identity_evidence=identity_evidence,
    )

    monkeypatch.setattr(
        module,
        "answer_query_with_drug_resolution",
        lambda **kwargs: workflow_result,
    )

    row = module.evaluate_case(case)

    assert row["overall_pass"] is True
    assert row["source_pass"] is True
    assert row["citation_reference_pass"] is True
    assert row["minimum_identity_evidence_pass"] is True