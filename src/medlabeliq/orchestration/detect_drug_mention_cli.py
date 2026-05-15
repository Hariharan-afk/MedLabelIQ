from __future__ import annotations

import argparse
import json

from medlabeliq.orchestration.drug_mention_detection import (
    detect_drug_mention_from_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect a drug mention from query text and map it to an indexed "
            "MedLabelIQ corpus concept when possible."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Medication-related user query.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = detect_drug_mention_from_query(args.query)

    payload = {
        "query": result.query,
        "status": result.status,
        "detected_mention": result.detected_mention,
        "retrieval_drug": result.retrieval_drug,
        "corpus_matches": result.corpus_matches,
        "selected_candidate": (
            result.selected_candidate.to_dict()
            if result.selected_candidate is not None
            else None
        ),
        "candidate_resolutions": [
            {
                "mention_text": candidate.mention_text,
                "resolution": candidate.resolution.to_dict(),
            }
            for candidate in result.candidate_resolutions
        ],
    }

    print("\nQUERY DRUG MENTION DETECTION")
    print("=" * 80)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()