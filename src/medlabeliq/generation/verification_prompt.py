from __future__ import annotations

from medlabeliq.generation.answer_schema import GroundedAnswer
from medlabeliq.generation.evidence_pack import EvidencePack


VERIFIER_SYSTEM_PROMPT = """
You are a strict evidence-support verifier for a medication-label grounded QA system.

Your job is NOT to answer the user question.
Your job is ONLY to decide whether the generated answer is directly supported by the cited evidence.

Use these verdicts:

1. "supported"
   - The cited evidence directly establishes the generated answer.
   - This includes affirmative answers and negative/refuting answers, as long as the cited evidence directly supports them.
   - A paraphrase is supported if it preserves the meaning of the cited evidence.
   - Explanatory "why" answers are supported when the cited evidence directly states the causal or risk relationship used in the answer.
   - Example: if evidence states that renal impairment increases the risk of metformin accumulation and lactic acidosis, then an answer explaining that metformin can be more dangerous in serious kidney problems because that risk is increased should be marked "supported".

2. "insufficient"
   - The cited evidence is related, but does not directly establish the generated answer.
   - Use this when the answer requires an unsupported leap, stronger generalization, broader population claim, or stronger certainty than the evidence provides.
   - Absolute or universal claims such as "guarantees", "always", "never", "cures", or "completely prevents" require direct explicit support.
   - Example: if evidence reports average weight change in one metformin clinical study, that does NOT directly establish whether metformin "guarantees weight loss"; mark that "insufficient".

3. "refuted"
   - The cited evidence directly contradicts the generated answer.

Rules:
- Evaluate only the evidence snippets listed in the answer citations.
- Do not use uncited retrieved evidence.
- Do not use outside knowledge.
- Do not reward answers that merely sound plausible.
- Judge whether the core claim of the answer is directly supported, not whether every phrase is identical.
- For risk, warning, adverse-effect, contraindication, indication, and mechanism questions, mark "supported" when the answer faithfully paraphrases the cited label evidence.
- For broad, absolute, population-wide, or treatment-effect claims, require direct evidence that specifically establishes the claim.
- Return valid JSON only, with no markdown.

Return exactly:

{
  "verdict": "supported" or "insufficient" or "refuted",
  "rationale": "brief explanation",
  "cited_evidence_used": ["E1"]
}
""".strip()


def build_verifier_user_prompt(
    *,
    question: str,
    answer: GroundedAnswer,
    evidence_pack: EvidencePack,
) -> str:
    evidence_by_id = {
        item.evidence_id: item
        for item in evidence_pack.evidence_items
    }

    cited_blocks: list[str] = []

    for citation in answer.citations:
        item = evidence_by_id.get(citation)

        if item is None:
            continue

        cited_blocks.append(
            "\n".join(
                [
                    f"[{item.evidence_id}]",
                    f"Drug concept: {item.concept_name}",
                    f"Retrieval family: {item.retrieval_family}",
                    f"Heading path: {item.heading}",
                    f"Source: {item.source_label}",
                    "Evidence text:",
                    item.chunk_text,
                ]
            )
        )

    cited_evidence_block = (
        "\n\n".join(cited_blocks)
        if cited_blocks
        else "NO VALID CITED EVIDENCE PROVIDED."
    )

    return f"""
User question:
{question}

Generated answer:
{answer.answer}

Generated evidence summary:
{answer.evidence_summary}

Answer citations:
{answer.citations}

Cited evidence only:
{cited_evidence_block}

Now return the verification JSON.
""".strip()