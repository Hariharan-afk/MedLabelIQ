from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medlabeliq.retrieval.dense_search import DenseSearchResult, dense_search_chunks
from medlabeliq.retrieval.lexical_search import SearchResult, search_chunks


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    section_id: str
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


def rrf_score(rank: int | None, k: int = 60) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def hybrid_search_chunks(
    query: str,
    *,
    concept_name: str | None = None,
    retrieval_family: str | None = None,
    limit: int = 10,
    candidate_pool: int = 30,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    """
    Hybrid retrieval using Reciprocal Rank Fusion over:
      1. PostgreSQL lexical full-text search
      2. Qdrant dense vector search
    """

    lexical_results = search_chunks(
        query=query,
        concept_name=concept_name,
        retrieval_family=retrieval_family,
        limit=candidate_pool,
    )

    dense_results = dense_search_chunks(
        query=query,
        concept_name=concept_name,
        retrieval_family=retrieval_family,
        limit=candidate_pool,
    )

    fused: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(lexical_results, start=1):
        fused[result.chunk_id] = {
            "base": result,
            "lexical_rank": rank,
            "dense_rank": None,
            "lexical_score": result.rank,
            "dense_score": None,
        }

    for rank, result in enumerate(dense_results, start=1):
        if result.chunk_id not in fused:
            fused[result.chunk_id] = {
                "base": result,
                "lexical_rank": None,
                "dense_rank": rank,
                "lexical_score": None,
                "dense_score": result.score,
            }
        else:
            fused[result.chunk_id]["dense_rank"] = rank
            fused[result.chunk_id]["dense_score"] = result.score

    hybrid_results: list[HybridSearchResult] = []

    for payload in fused.values():
        base = payload["base"]

        lexical_rank = payload["lexical_rank"]
        dense_rank = payload["dense_rank"]

        score = rrf_score(lexical_rank, k=rrf_k) + rrf_score(dense_rank, k=rrf_k)

        hybrid_results.append(
            HybridSearchResult(
                chunk_id=base.chunk_id,
                section_id=base.section_id,
                concept_name=base.concept_name,
                set_id=base.set_id,
                version_number=base.version_number,
                document_title=base.document_title,
                retrieval_family=base.retrieval_family,
                canonical_section_name=base.canonical_section_name,
                nearest_canonical_section_name=base.nearest_canonical_section_name,
                heading_path=base.heading_path,
                token_count=base.token_count,
                chunk_text=base.chunk_text,
                hybrid_score=score,
                lexical_rank=lexical_rank,
                dense_rank=dense_rank,
                lexical_score=payload["lexical_score"],
                dense_score=payload["dense_score"],
            )
        )

    hybrid_results.sort(
        key=lambda result: (
            result.hybrid_score,
            -(result.lexical_rank or 10_000),
            -(result.dense_rank or 10_000),
        ),
        reverse=True,
    )

    return hybrid_results[:limit]