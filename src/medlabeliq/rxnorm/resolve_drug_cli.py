from __future__ import annotations

import argparse
import json

from medlabeliq.rxnorm.resolver import resolve_drug_term


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a drug term using RxNorm and map it to the indexed MedLabelIQ corpus."
    )

    parser.add_argument(
        "--term",
        required=True,
        help="Drug term, brand name, or noisy user input to normalize.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resolution = resolve_drug_term(args.term)

    print("\nRXNORM DRUG NORMALIZATION")
    print("=" * 80)
    print(json.dumps(resolution.to_dict(), indent=2))


if __name__ == "__main__":
    main()