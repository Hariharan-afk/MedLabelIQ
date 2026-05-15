from __future__ import annotations

import argparse
import json

from medlabeliq.generation.answer_generator import answer_query
from medlabeliq.generation.evidence_pack import build_evidence_pack
from medlabeliq.generation.prompt_builder import SYSTEM_PROMPT, build_user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a grounded medication-label answer."
    )

    parser.add_argument("--query", required=True)
    parser.add_argument("--drug", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print the retrieved evidence after the answer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do retrieval and prompt construction only; do not call the LLM.",
    )

    return parser.parse_args()


def print_evidence(evidence_pack) -> None:
    print("\nRETRIEVED EVIDENCE")
    print("=" * 80)

    if not evidence_pack.evidence_items:
        print("No evidence retrieved.")
        return

    for item in evidence_pack.evidence_items:
        print("\n" + "-" * 80)
        print(f"[{item.evidence_id}]")
        print(f"Drug: {item.concept_name}")
        print(f"Family: {item.retrieval_family}")
        print(f"Heading: {item.heading}")
        print(f"Source: {item.source_label}")
        print(f"Hybrid score: {item.hybrid_score:.6f}")
        print(f"Lexical rank: {item.lexical_rank}")
        print(f"Dense rank: {item.dense_rank}")
        print("\nText preview:")
        print(item.chunk_text[:700].replace("\n", " "))


def main() -> None:
    args = parse_args()

    if args.dry_run:
        evidence_pack = build_evidence_pack(
            query=args.query,
            concept_name=args.drug,
            retrieval_family=args.family,
            top_k=args.top_k,
        )

        user_prompt = build_user_prompt(
            query=args.query,
            evidence_pack=evidence_pack,
        )

        print("\nDRY RUN — NO LLM CALL")
        print("=" * 80)

        print("\nSYSTEM PROMPT")
        print("=" * 80)
        print(SYSTEM_PROMPT)

        print("\nUSER PROMPT")
        print("=" * 80)
        print(user_prompt)

        if args.show_evidence:
            print_evidence(evidence_pack)

        return

    generated = answer_query(
        query=args.query,
        concept_name=args.drug,
        retrieval_family=args.family,
        top_k=args.top_k,
    )

    print("\nGROUNDED ANSWER")
    print("=" * 80)
    print(json.dumps(generated.answer.model_dump(), indent=2))

    if generated.verification is not None:
        print("\nANSWER VERIFICATION")
        print("=" * 80)
        print(json.dumps(generated.verification.model_dump(), indent=2))

    if args.show_evidence:
        print_evidence(generated.evidence_pack)


if __name__ == "__main__":
    main()