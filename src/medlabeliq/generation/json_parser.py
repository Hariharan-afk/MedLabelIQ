from __future__ import annotations

import json

from medlabeliq.generation.answer_schema import GroundedAnswer


def extract_json_object(raw_text: str) -> dict:
    """
    Parse the model response as JSON.
    If it contains surrounding text, extract the outermost JSON object.
    """
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level model output must be a JSON object.")
        return parsed
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not locate a JSON object in model output.")

        candidate = raw_text[start : end + 1]
        parsed = json.loads(candidate)

        if not isinstance(parsed, dict):
            raise ValueError("Extracted JSON was not an object.")

        return parsed


def parse_grounded_answer(
    raw_text: str,
    *,
    allowed_evidence_ids: set[str],
) -> GroundedAnswer:
    payload = extract_json_object(raw_text)
    answer = GroundedAnswer.model_validate(payload)

    invalid_ids = [
        citation
        for citation in answer.citations
        if citation not in allowed_evidence_ids
    ]

    if invalid_ids:
        raise ValueError(
            f"Model cited evidence IDs that were not provided: {invalid_ids}"
        )

    return answer