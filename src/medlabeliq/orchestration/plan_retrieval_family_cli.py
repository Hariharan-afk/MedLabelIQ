from __future__ import annotations

import argparse
import json

from medlabeliq.orchestration.retrieval_family_planner import (
    plan_retrieval_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a retrieval family from medication-related query text."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Medication-related user query.",
    )

    parser.add_argument(
        "--family",
        default=None,
        help="Optional explicitly requested retrieval family.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plan = plan_retrieval_family(
        args.query,
        requested_family=args.family,
    )

    payload = {
        "query": plan.query,
        "status": plan.status,
        "intent": plan.intent,
        "planned_family": plan.planned_family,
        "candidate_families": plan.candidate_families,
        "matches": [
            {
                "family": match.family,
                "intent": match.intent,
                "score": match.score,
                "matched_signals": match.matched_signals,
            }
            for match in plan.matches
        ],
    }

    print("\nRETRIEVAL FAMILY PLAN")
    print("=" * 80)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()