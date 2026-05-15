from __future__ import annotations

import argparse
import json

from medlabeliq.orchestration.drug_filter_resolution import (
    resolve_optional_drug_filter,
)
from medlabeliq.orchestration.drug_mention_detection import (
    build_not_attempted_detection,
    detect_drug_mention_from_query,
)
from medlabeliq.orchestration.retrieval_family_planner import (
    plan_retrieval_family,
)
from medlabeliq.orchestration.source_router import (
    plan_source_route,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan the best knowledge source for a medication query."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Medication-related user query.",
    )

    parser.add_argument(
        "--drug",
        default=None,
        help="Optional explicit drug filter.",
    )

    parser.add_argument(
        "--family",
        default=None,
        help="Optional explicit retrieval family.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    drug_resolution = resolve_optional_drug_filter(args.drug)

    family_plan = plan_retrieval_family(
        args.query,
        requested_family=args.family,
    )

    if args.drug is not None and args.drug.strip():
        drug_detection = build_not_attempted_detection(args.query)
    else:
        drug_detection = detect_drug_mention_from_query(args.query)

    source_plan = plan_source_route(
        args.query,
        requested_family=args.family,
        family_plan=family_plan,
        drug_resolution=drug_resolution,
        drug_mention_detection=drug_detection,
    )

    payload = {
        "query": source_plan.query,
        "status": source_plan.status,
        "selected_source": source_plan.selected_source,
        "intent": source_plan.intent,
        "candidate_sources": source_plan.candidate_sources,
        "matches": [
            {
                "source": match.source,
                "intent": match.intent,
                "score": match.score,
                "matched_signals": match.matched_signals,
            }
            for match in source_plan.matches
        ],
    }

    print("\nSOURCE ROUTE PLAN")
    print("=" * 80)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()