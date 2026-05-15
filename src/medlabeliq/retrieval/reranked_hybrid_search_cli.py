from __future__ import annotations

import argparse
import textwrap

from medlabeliq.retrieval.reranked_hybrid_search import (
    reranked_hybrid_search_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval followed by reranking."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--drug", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = reranked_hybrid_search_chunks(
        query=args.query,
        concept_name=args.drug,
        retrieval_family=args.family,
        limit=args.limit,
        candidate_pool=args.candidate_pool,
    )

    print("\nRERANKED HYBRID SEARCH RESULTS")
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"Drug filter: {args.drug or 'None'}")
    print(f"Family filter: {args.family or 'None'}")
    print(f"Results: {len(results)}")

    for idx, result in enumerate(results, start=1):
        heading = " > ".join(result.heading_path)

        print("\n" + "-" * 80)
        print(f"#{idx} | reranker_score={result.reranker_score:.6f}")
        print(f"Original hybrid rank: {result.original_hybrid_rank}")
        print(f"Hybrid score: {result.hybrid_score:.6f}")
        print(f"Lexical rank: {result.lexical_rank}")
        print(f"Dense rank: {result.dense_rank}")
        print(f"Drug: {result.concept_name}")
        print(f"Retrieval family: {result.retrieval_family}")
        print(f"Heading: {heading}")
        print(f"SET ID: {result.set_id}")
        print(f"SPL version: {result.version_number}")
        print(f"Tokens: {result.token_count}")

        print("\nChunk preview:")
        print(textwrap.shorten(result.chunk_text, width=700, placeholder=" ..."))


if __name__ == "__main__":
    main()