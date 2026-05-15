from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from medlabeliq.config.settings import settings
from medlabeliq.embeddings.embedding_model import embed_query
from medlabeliq.qdrant_store.client import get_qdrant_client


@dataclass(frozen=True)
class DenseSearchResult:
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
    score: float
    chunk_text: str


def build_filter(
    concept_name: str | None = None,
    retrieval_family: str | None = None,
) -> models.Filter | None:
    must_conditions = []

    if concept_name:
        must_conditions.append(
            models.FieldCondition(
                key="concept_name",
                match=models.MatchValue(value=concept_name),
            )
        )

    if retrieval_family:
        must_conditions.append(
            models.FieldCondition(
                key="retrieval_family",
                match=models.MatchValue(value=retrieval_family),
            )
        )

    if not must_conditions:
        return None

    return models.Filter(must=must_conditions)


def dense_search_chunks(
    query: str,
    *,
    concept_name: str | None = None,
    retrieval_family: str | None = None,
    limit: int = 10,
) -> list[DenseSearchResult]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = embed_query(query)
    client = get_qdrant_client()

    query_filter = build_filter(
        concept_name=concept_name,
        retrieval_family=retrieval_family,
    )

    query_response = client.query_points(
    collection_name=settings.qdrant_collection,
    query=query_vector,
    query_filter=query_filter,
    limit=limit,
    with_payload=True,
    with_vectors=False,
)

    hits = query_response.points

    results: list[DenseSearchResult] = []

    for hit in hits:
        payload: dict[str, Any] = hit.payload or {}

        results.append(
            DenseSearchResult(
                chunk_id=str(payload["chunk_id"]),
                section_id=str(payload["section_id"]),
                concept_name=str(payload["concept_name"]),
                set_id=str(payload["set_id"]),
                version_number=int(payload["version_number"]),
                document_title=payload.get("document_title"),
                retrieval_family=str(payload["retrieval_family"]),
                canonical_section_name=payload.get("canonical_section_name"),
                nearest_canonical_section_name=payload.get(
                    "nearest_canonical_section_name"
                ),
                heading_path=list(payload.get("heading_path") or []),
                token_count=int(payload["token_count"]),
                score=float(hit.score),
                chunk_text=str(payload["chunk_text"]),
            )
        )

    return results