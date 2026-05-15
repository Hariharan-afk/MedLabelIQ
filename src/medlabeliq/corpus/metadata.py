from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection
from medlabeliq.qdrant_store.client import get_qdrant_client


# =============================================================================
# Generic helpers
# =============================================================================

def project_relative_path(path: Path) -> str:
    """
    Convert a project-local path into a portable POSIX-style relative path.
    """
    try:
        return path.relative_to(settings.project_root).as_posix()
    except ValueError:
        return str(path)


def fetch_scalar_count(cur, query: str) -> int:
    cur.execute(query)
    row = cur.fetchone()

    if row is None:
        return 0

    return int(row["count"])


def fetch_qdrant_point_count() -> int | None:
    """
    Return current Qdrant point count for the configured collection.

    Returns None if the collection or Qdrant service is unavailable.
    """
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(
            settings.qdrant_collection
        )

        if collection_info.points_count is None:
            return 0

        return int(collection_info.points_count)

    except Exception:
        return None


# =============================================================================
# Live corpus summaries
# =============================================================================

def collect_live_corpus_counts() -> dict[str, Any]:
    """
    Collect live corpus counts from PostgreSQL and Qdrant.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            drug_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(DISTINCT concept_name) AS count
                FROM label_document;
                """,
            )

            label_document_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(*) AS count
                FROM label_document;
                """,
            )

            label_version_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(*) AS count
                FROM label_version;
                """,
            )

            product_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(*) AS count
                FROM label_product;
                """,
            )

            ingredient_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(*) AS count
                FROM product_ingredient;
                """,
            )

            section_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(*) AS count
                FROM label_section;
                """,
            )

            retrievable_section_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(*) AS count
                FROM label_section
                WHERE retrieval_family IS NOT NULL
                  AND direct_text IS NOT NULL
                  AND LENGTH(TRIM(direct_text)) > 0;
                """,
            )

            chunk_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(*) AS count
                FROM section_chunk;
                """,
            )

            retrieval_family_count = fetch_scalar_count(
                cur,
                """
                SELECT COUNT(DISTINCT retrieval_family) AS count
                FROM section_chunk
                WHERE retrieval_family IS NOT NULL;
                """,
            )

    return {
        "drug_count": drug_count,
        "label_document_count": label_document_count,
        "label_version_count": label_version_count,
        "product_count": product_count,
        "ingredient_count": ingredient_count,
        "section_count": section_count,
        "retrievable_section_count": retrievable_section_count,
        "chunk_count": chunk_count,
        "retrieval_family_count": retrieval_family_count,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_point_count": fetch_qdrant_point_count(),
        "embedding_model_name": settings.embedding_model_name,
    }


def list_drug_summaries() -> list[dict[str, Any]]:
    """
    Return indexed corpus counts grouped by normalized drug concept.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.concept_name,
                    COUNT(DISTINCT d.label_id)::int AS label_count,
                    COUNT(DISTINCT v.label_version_id)::int AS label_version_count,
                    COUNT(DISTINCT s.section_id)::int AS section_count,
                    COUNT(DISTINCT c.chunk_id)::int AS chunk_count
                FROM label_document d
                LEFT JOIN label_version v
                    ON v.label_id = d.label_id
                LEFT JOIN label_section s
                    ON s.label_version_id = v.label_version_id
                LEFT JOIN section_chunk c
                    ON c.label_version_id = v.label_version_id
                GROUP BY d.concept_name
                ORDER BY d.concept_name ASC;
                """
            )

            return list(cur.fetchall())


def list_retrieval_family_summaries() -> list[dict[str, Any]]:
    """
    Return indexed evidence coverage grouped by retrieval family.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.retrieval_family,
                    COUNT(DISTINCT c.section_id)::int AS section_count,
                    COUNT(DISTINCT c.chunk_id)::int AS chunk_count,
                    COUNT(DISTINCT d.concept_name)::int AS drug_count
                FROM section_chunk c
                JOIN label_version v
                    ON v.label_version_id = c.label_version_id
                JOIN label_document d
                    ON d.label_id = v.label_id
                WHERE c.retrieval_family IS NOT NULL
                GROUP BY c.retrieval_family
                ORDER BY chunk_count DESC, c.retrieval_family ASC;
                """
            )

            return list(cur.fetchall())


# =============================================================================
# Build-log metadata
# =============================================================================

def get_latest_corpus_build() -> dict[str, Any] | None:
    """
    Return the most recent recorded corpus build, if present.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    build_id,
                    built_at,
                    build_source,
                    seed_file_path,
                    drug_count,
                    label_document_count,
                    label_version_count,
                    product_count,
                    ingredient_count,
                    section_count,
                    retrievable_section_count,
                    chunk_count,
                    retrieval_family_count,
                    qdrant_collection,
                    qdrant_point_count,
                    embedding_model_name
                FROM corpus_build_log
                ORDER BY built_at DESC
                LIMIT 1;
                """
            )

            row = cur.fetchone()

    if row is None:
        return None

    payload = dict(row)
    payload["build_id"] = str(payload["build_id"])
    return payload


def collect_corpus_stats() -> dict[str, Any]:
    """
    Return live corpus stats plus the latest persisted build metadata.
    """
    stats = collect_live_corpus_counts()
    stats["latest_build"] = get_latest_corpus_build()
    return stats


def record_corpus_build(
    *,
    build_source: str = "bootstrap",
) -> tuple[str, dict[str, Any]]:
    """
    Persist a snapshot of the current corpus/index state.
    """
    live_stats = collect_live_corpus_counts()

    qdrant_point_count = live_stats["qdrant_point_count"]
    if qdrant_point_count is None:
        raise RuntimeError(
            "Cannot record corpus build because Qdrant point count is unavailable."
        )

    build_uuid = uuid4()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpus_build_log (
                    build_id,
                    build_source,
                    seed_file_path,
                    drug_count,
                    label_document_count,
                    label_version_count,
                    product_count,
                    ingredient_count,
                    section_count,
                    retrievable_section_count,
                    chunk_count,
                    retrieval_family_count,
                    qdrant_collection,
                    qdrant_point_count,
                    embedding_model_name
                )
                VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                );
                """,
                (
                    build_uuid,
                    build_source,
                    project_relative_path(settings.smoke_set_path),
                    live_stats["drug_count"],
                    live_stats["label_document_count"],
                    live_stats["label_version_count"],
                    live_stats["product_count"],
                    live_stats["ingredient_count"],
                    live_stats["section_count"],
                    live_stats["retrievable_section_count"],
                    live_stats["chunk_count"],
                    live_stats["retrieval_family_count"],
                    live_stats["qdrant_collection"],
                    qdrant_point_count,
                    live_stats["embedding_model_name"],
                ),
            )

        conn.commit()

    return str(build_uuid), live_stats