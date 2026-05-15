from __future__ import annotations

from dataclasses import dataclass

from medlabeliq.reranking.bge_reranker import rerank_pairs
from medlabeliq.retrieval.hybrid_search import (
    HybridSearchResult,
    hybrid_search_chunks,
)


@dataclass(frozen=True)
class RerankedHybridSearchResult:
    chunk_id: str
    concept_name: str
    set_id: str
    version_number: int
    document_title: str | None
    retrieval_family: str
    canonical_section_name: str | None
    nearest_canonical_section_name: str | None
    heading_path: list[str]
    token_count: int
    chunk_text: str

    hybrid_score: float
    lexical_rank: int | None
    dense_rank: int | None
    lexical_score: float | None
    dense_score: float | None

    reranker_score: float
    original_hybrid_rank: int


def build_reranker_passage(result: HybridSearchResult) -> str:
    """
    Build the passage sent to the reranker.

    We include heading and chunk text, but avoid overloading the reranker with
    IDs and low-value metadata.
    """
    heading = " > ".join(result.heading_path)

    return (
        f"Drug: {result.concept_name}\n"
        f"Section family: {result.retrieval_family}\n"
        f"Heading: {heading}\n\n"
        f"{result.chunk_text}"
    )


def reranked_hybrid_search_chunks(
    query: str,
    *,
    concept_name: str | None = None,
    retrieval_family: str | None = None,
    limit: int = 10,
    candidate_pool: int = 30,
) -> list[RerankedHybridSearchResult]:
    """
    Hybrid retrieval followed by BGE reranking.
    """
    hybrid_candidates = hybrid_search_chunks(
        query=query,
        concept_name=concept_name,
        retrieval_family=retrieval_family,
        limit=candidate_pool,
        candidate_pool=candidate_pool,
    )

    if not hybrid_candidates:
        return []

    reranker_passages = [
        build_reranker_passage(candidate)
        for candidate in hybrid_candidates
    ]

    scores = rerank_pairs(query, reranker_passages)

    reranked: list[RerankedHybridSearchResult] = []

    for original_rank, (candidate, score) in enumerate(
        zip(hybrid_candidates, scores, strict=True),
        start=1,
    ):
        reranked.append(
            RerankedHybridSearchResult(
                chunk_id=candidate.chunk_id,
                concept_name=candidate.concept_name,
                set_id=candidate.set_id,
                version_number=candidate.version_number,
                document_title=candidate.document_title,
                retrieval_family=candidate.retrieval_family,
                canonical_section_name=candidate.canonical_section_name,
                nearest_canonical_section_name=candidate.nearest_canonical_section_name,
                heading_path=candidate.heading_path,
                token_count=candidate.token_count,
                chunk_text=candidate.chunk_text,
                hybrid_score=candidate.hybrid_score,
                lexical_rank=candidate.lexical_rank,
                dense_rank=candidate.dense_rank,
                lexical_score=candidate.lexical_score,
                dense_score=candidate.dense_score,
                reranker_score=score,
                original_hybrid_rank=original_rank,
            )
        )

    reranked.sort(
        key=lambda result: result.reranker_score,
        reverse=True,
    )

    return reranked[:limit]