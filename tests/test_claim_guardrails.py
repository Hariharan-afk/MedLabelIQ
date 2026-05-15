from __future__ import annotations

from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.claim_guardrails import (
    assess_guarantee_claim_support,
    assess_negative_treatment_claim_support,
)
from medlabeliq.generation.evidence_pack import EvidenceItem, EvidencePack


def make_evidence_item(
    *,
    evidence_id: str = "E1",
    text: str,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id="22d7d2e6-2af5-59af-b930-0a10417ee027",
        section_id="1ada10ad-d795-50ea-a7b8-b53309194ce5",
        concept_name="metformin",
        retrieval_family="clinical_studies",
        canonical_section_name=None,
        nearest_canonical_section_name="clinical_studies",
        heading_path=["14 CLINICAL STUDIES", "Adult Clinical Studies"],
        set_id="test-set-id",
        version_number=1,
        chunk_text=text,
        hybrid_score=0.01,
        lexical_rank=None,
        dense_rank=1,
    )


def make_evidence_pack(
    *,
    text: str,
) -> EvidencePack:
    return EvidencePack(
        query="test query",
        concept_name="metformin",
        retrieval_family=None,
        evidence_items=[
            make_evidence_item(text=text),
        ],
    )


def make_answer(
    *,
    answer_text: str,
    citations: list[str] | None = None,
) -> GroundedAnswer:
    return GroundedAnswer(
        status="answered",
        answer=answer_text,
        citations=citations or ["E1"],
        evidence_summary="Test summary.",
        safety_note="Test safety note.",
    )


def test_guarantee_claim_abstains_without_explicit_guarantee_evidence() -> None:
    question = "Does metformin guarantee weight loss?"

    answer = make_answer(
        answer_text="No, metformin does not guarantee weight loss."
    )

    evidence_pack = make_evidence_pack(
        text=(
            "In one clinical study, patients receiving metformin had "
            "a modest average change in body weight."
        )
    )

    decision = assess_guarantee_claim_support(
        question=question,
        answer=answer,
        evidence_pack=evidence_pack,
    )

    assert decision.should_abstain is True
    assert decision.reason is not None


def test_guarantee_claim_does_not_abstain_when_evidence_explicitly_mentions_guarantee() -> None:
    question = "Does this label guarantee a clinical outcome?"

    answer = make_answer(
        answer_text="The label explicitly states that no outcome is guaranteed."
    )

    evidence_pack = make_evidence_pack(
        text="The label does not guarantee any clinical outcome."
    )

    decision = assess_guarantee_claim_support(
        question=question,
        answer=answer,
        evidence_pack=evidence_pack,
    )

    assert decision.should_abstain is False
    assert decision.reason is None


def test_negative_treatment_claim_abstains_when_target_condition_is_not_in_evidence() -> None:
    question = "Does apixaban treat bacterial infections?"

    answer = make_answer(
        answer_text=(
            "No, apixaban does not treat bacterial infections; "
            "it is used for clot-related conditions."
        )
    )

    evidence_pack = EvidencePack(
        query=question,
        concept_name="apixaban",
        retrieval_family=None,
        evidence_items=[
            EvidenceItem(
                evidence_id="E1",
                chunk_id="9b5510a5-14fe-5ee1-accb-ec4a6f27b946",
                section_id="6caa7b5b-e391-57ac-ad34-d7e1212e90ca",
                concept_name="apixaban",
                retrieval_family="medication_guide",
                canonical_section_name="medication_guide",
                nearest_canonical_section_name="medication_guide",
                heading_path=["MEDICATION GUIDE"],
                set_id="test-set-id",
                version_number=1,
                chunk_text=(
                    "Apixaban tablets are used to reduce stroke risk and "
                    "treat blood clots. They are not for use in people with "
                    "antiphospholipid syndrome."
                ),
                hybrid_score=0.01,
                lexical_rank=None,
                dense_rank=1,
            ),
        ],
    )

    decision = assess_negative_treatment_claim_support(
        question=question,
        answer=answer,
        evidence_pack=evidence_pack,
    )

    assert decision.should_abstain is True
    assert decision.reason is not None


def test_negative_treatment_claim_can_pass_when_target_and_negative_evidence_are_explicit() -> None:
    question = "Does this drug treat bacterial infections?"

    answer = make_answer(
        answer_text="No, this drug is not indicated for bacterial infections."
    )

    evidence_pack = make_evidence_pack(
        text="This drug is not indicated for bacterial infections."
    )

    decision = assess_negative_treatment_claim_support(
        question=question,
        answer=answer,
        evidence_pack=evidence_pack,
    )

    assert decision.should_abstain is False
    assert decision.reason is None


def test_non_negative_treatment_answer_does_not_trigger_negative_treatment_guardrail() -> None:
    question = "Can metformin cause lactic acidosis?"

    answer = make_answer(
        answer_text="Yes. Metformin can cause lactic acidosis."
    )

    evidence_pack = make_evidence_pack(
        text="Postmarketing cases of metformin-associated lactic acidosis have occurred."
    )

    decision = assess_negative_treatment_claim_support(
        question=question,
        answer=answer,
        evidence_pack=evidence_pack,
    )

    assert decision.should_abstain is False
    assert decision.reason is None