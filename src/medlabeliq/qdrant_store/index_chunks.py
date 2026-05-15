from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models

from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection
from medlabeliq.embeddings.embedding_model import embed_texts
from medlabeliq.qdrant_store.client import get_qdrant_client


def fetch_chunks() -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.chunk_id::text,
                    c.section_id::text,
                    c.label_version_id::text,
                    d.concept_name,
                    d.set_id,
                    v.version_number,
                    d.document_title,
                    c.retrieval_family,
                    c.canonical_section_name,
                    c.nearest_canonical_section_name,
                    c.heading_path,
                    c.chunk_text,
                    c.embedding_text,
                    c.token_count,
                    c.char_count,
                    c.metadata
                FROM section_chunk c
                JOIN label_version v ON v.label_version_id = c.label_version_id
                JOIN label_document d ON d.label_id = v.label_id
                ORDER BY d.concept_name, c.retrieval_family, c.chunk_index;
                """
            )
            return list(cur.fetchall())


def ensure_collection(vector_size: int) -> None:
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection

    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def chunk_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],
        "section_id": row["section_id"],
        "label_version_id": row["label_version_id"],
        "concept_name": row["concept_name"],
        "set_id": row["set_id"],
        "version_number": row["version_number"],
        "document_title": row["document_title"],
        "retrieval_family": row["retrieval_family"],
        "canonical_section_name": row["canonical_section_name"],
        "nearest_canonical_section_name": row["nearest_canonical_section_name"],
        "heading_path": row["heading_path"],
        "chunk_text": row["chunk_text"],
        "embedding_text": row["embedding_text"],
        "token_count": row["token_count"],
        "char_count": row["char_count"],
        "metadata": row["metadata"],
    }


def to_point_id(chunk_id: str) -> str:
    """
    Qdrant accepts UUID strings as point IDs.
    Our chunk_id is already a deterministic UUID from PostgreSQL.
    """
    return str(uuid.UUID(chunk_id))


def index_chunks(batch_size: int = 64) -> None:
    rows = fetch_chunks()
    if not rows:
        raise RuntimeError("No section_chunk rows found. Run Step 6 first.")

    print("\nQDRANT INDEXING")
    print("=" * 80)
    print(f"Chunks to index: {len(rows)}")
    print(f"Embedding model: {settings.embedding_model_name}")
    print(f"Collection: {settings.qdrant_collection}")

    # Create collection after seeing actual embedding dimension.
    first_vector = embed_texts([rows[0]["embedding_text"]])[0]
    ensure_collection(vector_size=len(first_vector))

    client = get_qdrant_client()

    total_indexed = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]

        # Reuse first vector for first row of first batch to avoid recomputing it.
        if start == 0:
            vectors = [first_vector]
            if len(batch) > 1:
                vectors.extend(embed_texts([row["embedding_text"] for row in batch[1:]]))
        else:
            vectors = embed_texts([row["embedding_text"] for row in batch])

        points = [
            models.PointStruct(
                id=to_point_id(row["chunk_id"]),
                vector=vector,
                payload=chunk_payload(row),
            )
            for row, vector in zip(batch, vectors, strict=True)
        ]

        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )

        total_indexed += len(points)
        print(f"Indexed {total_indexed}/{len(rows)} chunks")

    print("\nQDRANT INDEXING COMPLETE")
    print("=" * 80)
    print(f"Indexed chunks: {total_indexed}")


def main() -> None:
    index_chunks()


if __name__ == "__main__":
    main()