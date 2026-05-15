from __future__ import annotations

from medlabeliq.orchestration.retrieval_family_planner import (
    plan_retrieval_family,
)


def test_plan_indications_family() -> None:
    plan = plan_retrieval_family(
        "What is omeprazole used for?"
    )

    assert plan.status == "routed_single_family"
    assert plan.intent == "indication_or_use"
    assert plan.planned_family == "indications_and_usage"
    assert plan.can_filter is True


def test_plan_interactions_family() -> None:
    plan = plan_retrieval_family(
        "Can apixaban be taken with aspirin?"
    )

    assert plan.status == "routed_single_family"
    assert plan.intent == "interaction"
    assert plan.planned_family == "drug_interactions"


def test_plan_adverse_reactions_family() -> None:
    plan = plan_retrieval_family(
        "What are the side effects of sertraline?"
    )

    assert plan.status == "routed_single_family"
    assert plan.intent == "adverse_reaction"
    assert plan.planned_family == "adverse_reactions"


def test_plan_dosage_family() -> None:
    plan = plan_retrieval_family(
        "What dose of omeprazole should I take?"
    )

    assert plan.status == "routed_single_family"
    assert plan.intent == "dosage"
    assert plan.planned_family == "dosage_and_administration"


def test_plan_safety_family_group_remains_unfiltered() -> None:
    plan = plan_retrieval_family(
        "Can metformin cause dangerous lactic acidosis?"
    )

    assert plan.status == "candidate_family_group_unfiltered"
    assert plan.intent == "safety_warning"
    assert plan.planned_family is None
    assert "warnings_and_precautions" in plan.candidate_families
    assert plan.can_filter is False


def test_explicit_family_disables_auto_planning() -> None:
    plan = plan_retrieval_family(
        "What is omeprazole used for?",
        requested_family="clinical_studies",
    )

    assert plan.status == "not_attempted_explicit_family_present"
    assert plan.planned_family is None