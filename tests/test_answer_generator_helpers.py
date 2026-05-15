from __future__ import annotations

from medlabeliq.generation.answer_generator import (
    APPLICATION_SAFETY_NOTE,
    INSUFFICIENT_EVIDENCE_ANSWER,
    INSUFFICIENT_EVIDENCE_SUMMARY,
    make_insufficient_answer,
)


def test_make_insufficient_answer_has_expected_contract() -> None:
    answer = make_insufficient_answer()

    assert answer.status == "insufficient_evidence"
    assert answer.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert answer.citations == []
    assert answer.evidence_summary == INSUFFICIENT_EVIDENCE_SUMMARY
    assert answer.safety_note == APPLICATION_SAFETY_NOTE