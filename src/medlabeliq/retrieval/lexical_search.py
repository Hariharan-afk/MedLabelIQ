from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medlabeliq.db.connection import get_connection


@dataclass(frozen=True)
class SearchResult:
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
    rank: float
    chunk_text: str


def search_chunks(
    query: str,
    *,
    concept_name: str | None = None,
    retrieval_family: str | None = None,
    limit: int = 10,
) -> list[SearchResult]:
    """
    PostgreSQL full-text retrieval baseline over section-aware chunks.

    This searches `embedding_text`, not only raw chunk text, because embedding_text
    includes drug concept, label title, SET ID, SPL version, retrieval family,
    canonical section, heading path, and the actual chunk text.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    where_clauses = [
        "q.search_query @@ to_tsvector('english', c.embedding_text)"
    ]
    params: list[Any] = [query]

    if concept_name:
        where_clauses.append("LOWER(d.concept_name) = LOWER(%s)")
        params.append(concept_name)

    if retrieval_family:
        where_clauses.append("c.retrieval_family = %s")
        params.append(retrieval_family)

    params.append(limit)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        WITH q AS (
            SELECT websearch_to_tsquery('english', %s) AS search_query
        )
        SELECT
            c.chunk_id::text,
            c.section_id::text,
            d.concept_name,
            d.set_id,
            v.version_number,
            d.document_title,
            c.retrieval_family,
            c.canonical_section_name,
            c.nearest_canonical_section_name,
            c.heading_path,
            c.token_count,
            ts_rank_cd(
                to_tsvector('english', c.embedding_text),
                q.search_query
            ) AS rank,
            c.chunk_text
        FROM section_chunk c
        JOIN label_version v ON v.label_version_id = c.label_version_id
        JOIN label_document d ON d.label_id = v.label_id
        CROSS JOIN q
        WHERE {where_sql}
        ORDER BY rank DESC, d.concept_name, c.retrieval_family, c.chunk_index
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            section_id=row["section_id"],
            concept_name=row["concept_name"],
            set_id=row["set_id"],
            version_number=row["version_number"],
            document_title=row["document_title"],
            retrieval_family=row["retrieval_family"],
            canonical_section_name=row["canonical_section_name"],
            nearest_canonical_section_name=row["nearest_canonical_section_name"],
            heading_path=row["heading_path"],
            token_count=row["token_count"],
            rank=float(row["rank"]),
            chunk_text=row["chunk_text"],
        )
        for row in rows
    ]