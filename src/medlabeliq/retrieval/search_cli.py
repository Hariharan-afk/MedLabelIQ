from __future__ import annotations

import argparse
import textwrap

from medlabeliq.retrieval.lexical_search import search_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search section-aware DailyMed chunks using PostgreSQL full-text search."
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--drug", default=None, help="Optional concept_name filter")
    parser.add_argument(
        "--family",
        default=None,
        help="Optional retrieval_family filter, e.g. warnings_and_precautions",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = search_chunks(
        query=args.query,
        concept_name=args.drug,
        retrieval_family=args.family,
        limit=args.limit,
    )

    print("\nLEXICAL SEARCH RESULTS")
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"Drug filter: {args.drug or 'None'}")
    print(f"Family filter: {args.family or 'None'}")
    print(f"Results: {len(results)}")

    for idx, result in enumerate(results, start=1):
        heading = " > ".join(result.heading_path)

        print("\n" + "-" * 80)
        print(f"#{idx} | rank={result.rank:.4f}")
        print(f"Drug: {result.concept_name}")
        print(f"Retrieval family: {result.retrieval_family}")
        print(f"Canonical section: {result.canonical_section_name}")
        print(f"Nearest canonical: {result.nearest_canonical_section_name}")
        print(f"Heading: {heading}")
        print(f"SET ID: {result.set_id}")
        print(f"SPL version: {result.version_number}")
        print(f"Tokens: {result.token_count}")
        print("\nChunk preview:")
        print(textwrap.shorten(result.chunk_text, width=700, placeholder=" ..."))


if __name__ == "__main__":
    main()