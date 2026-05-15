from __future__ import annotations

from medlabeliq.orchestration.drug_filter_resolution import (
    DrugFilterResolution,
)
from medlabeliq.orchestration.drug_mention_detection import (
    DrugMentionDetection,
)
from medlabeliq.orchestration.retrieval_family_planner import (
    RetrievalFamilyPlan,
)
from medlabeliq.orchestration.source_router import plan_source_route


def make_not_requested_resolution() -> DrugFilterResolution:
    return DrugFilterResolution(
        requested_drug=None,
        status="not_requested",
        retrieval_drug=None,
        corpus_matches=[],
        selected_candidate=None,
        rxnorm_resolution=None,
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


def make_no_family_plan(query: str) -> RetrievalFamilyPlan:
    return RetrievalFamilyPlan(
        query=query,
        status="no_route_detected",
        intent=None,
        planned_family=None,
        candidate_families=[],
        matches=[],
    )


def make_label_family_plan(query: str) -> RetrievalFamilyPlan:
    return RetrievalFamilyPlan(
        query=query,
        status="routed_single_family",
        intent="indication_or_use",
        planned_family="indications_and_usage",
        candidate_families=["indications_and_usage"],
        matches=[],
    )


def test_identity_equivalence_routes_to_rxnorm() -> None:
    query = "Is Eliquis the same as apixaban?"

    plan = plan_source_route(
        query,
        requested_family=None,
        family_plan=make_no_family_plan(query),
        drug_resolution=make_not_requested_resolution(),
        drug_mention_detection=make_detected_drug(
            query,
            mention="apixaban",
            retrieval_drug="apixaban",
        ),
    )

    assert plan.status == "routed_rxnorm_identity"
    assert plan.selected_source == "rxnorm_identity"
    assert plan.intent == "brand_generic_equivalence"


def test_generic_name_query_routes_to_rxnorm() -> None:
    query = "What is the generic name of Glucophage?"

    plan = plan_source_route(
        query,
        requested_family=None,
        family_plan=make_no_family_plan(query),
        drug_resolution=make_not_requested_resolution(),
        drug_mention_detection=make_detected_drug(
            query,
            mention="Glucophage",
            retrieval_drug="metformin",
        ),
    )

    assert plan.status == "routed_rxnorm_identity"
    assert plan.selected_source == "rxnorm_identity"
    assert plan.intent == "generic_name_lookup"


def test_indications_query_routes_to_dailymed() -> None:
    query = "What is omeprazole used for?"

    plan = plan_source_route(
        query,
        requested_family=None,
        family_plan=make_label_family_plan(query),
        drug_resolution=make_not_requested_resolution(),
        drug_mention_detection=make_detected_drug(
            query,
            mention="omeprazole",
            retrieval_drug="omeprazole",
        ),
    )

    assert plan.status == "routed_dailymed_label"
    assert plan.selected_source == "dailymed_label"


def test_mixed_identity_and_label_query_is_ambiguous() -> None:
    query = "Is Glucophage the same as metformin and what is it used for?"

    plan = plan_source_route(
        query,
        requested_family=None,
        family_plan=make_label_family_plan(query),
        drug_resolution=make_not_requested_resolution(),
        drug_mention_detection=make_detected_drug(
            query,
            mention="metformin",
            retrieval_drug="metformin",
        ),
    )

    assert plan.status == "ambiguous_mixed_source"
    assert plan.selected_source == "dailymed_label"
    assert "rxnorm_identity" in plan.candidate_sources
    assert "dailymed_label" in plan.candidate_sources