from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg.types.json import Jsonb

from medlabeliq.chunking.section_chunker import chunk_section_text
from medlabeliq.db.connection import get_connection


MEDLABELIQ_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/medlabeliq/phase1",
)


def deterministic_uuid(name: str) -> str:
    return str(uuid.uuid5(MEDLABELIQ_NAMESPACE, name))


def normalize_heading_path(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [value]

    return [str(value)]


def fetch_retrievable_sections(cur) -> list[dict[str, Any]]:
    """
    Fetch sections that should become retrieval chunks.

    We intentionally require retrieval_family IS NOT NULL so packaging/provenance
    sections such as principal display panels are not embedded as answer evidence.
    """
    cur.execute(
        """
        SELECT
            d.concept_name,
            d.document_title,
            d.set_id,
            v.version_number,
            v.label_version_id,
            s.section_id,
            s.source_section_uid,
            s.order_index,
            s.retrieval_family,
            s.canonical_section_name,
            s.nearest_canonical_section_name,
            s.heading_path,
            s.direct_text
        FROM label_section s
        JOIN label_version v ON v.label_version_id = s.label_version_id
        JOIN label_document d ON d.label_id = v.label_id
        WHERE s.retrieval_family IS NOT NULL
          AND s.direct_text IS NOT NULL
          AND LENGTH(TRIM(s.direct_text)) > 0
        ORDER BY d.concept_name, s.order_index;
        """
    )
    return list(cur.fetchall())


def clear_existing_chunks(cur) -> None:
    cur.execute("DELETE FROM section_chunk;")


def insert_chunk(cur, section: dict[str, Any], chunk) -> None:
    chunk_id = deterministic_uuid(
        f"chunk:{section['section_id']}:{chunk.chunk_index}:section_text"
    )

    cur.execute(
        """
        INSERT INTO section_chunk (
            chunk_id,
            section_id,
            label_version_id,
            chunk_index,
            chunk_type,
            retrieval_family,
            canonical_section_name,
            nearest_canonical_section_name,
            heading_path,
            chunk_text,
            embedding_text,
            token_count,
            char_count,
            start_word_index,
            end_word_index,
            metadata
        )
        VALUES (
            %s, %s, %s, %s, 'section_text', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (section_id, chunk_index, chunk_type) DO UPDATE SET
            retrieval_family = EXCLUDED.retrieval_family,
            canonical_section_name = EXCLUDED.canonical_section_name,
            nearest_canonical_section_name = EXCLUDED.nearest_canonical_section_name,
            heading_path = EXCLUDED.heading_path,
            chunk_text = EXCLUDED.chunk_text,
            embedding_text = EXCLUDED.embedding_text,
            token_count = EXCLUDED.token_count,
            char_count = EXCLUDED.char_count,
            start_word_index = EXCLUDED.start_word_index,
            end_word_index = EXCLUDED.end_word_index,
            metadata = EXCLUDED.metadata
        """,
        (
            chunk_id,
            section["section_id"],
            section["label_version_id"],
            chunk.chunk_index,
            section["retrieval_family"],
            section["canonical_section_name"],
            section["nearest_canonical_section_name"],
            Jsonb(normalize_heading_path(section["heading_path"])),
            chunk.chunk_text,
            chunk.embedding_text,
            chunk.token_count,
            chunk.char_count,
            chunk.start_word_index,
            chunk.end_word_index,
            Jsonb(chunk.metadata),
        ),
    )


def main() -> None:
    max_words = 220
    overlap_words = 40

    with get_connection() as conn:
        with conn.cursor() as cur:
            sections = fetch_retrievable_sections(cur)
            clear_existing_chunks(cur)

            total_chunks = 0

            for section in sections:
                heading_path = normalize_heading_path(section["heading_path"])

                chunks = chunk_section_text(
                    concept_name=section["concept_name"],
                    document_title=section["document_title"],
                    set_id=section["set_id"],
                    version_number=section["version_number"],
                    section_id=str(section["section_id"]),
                    source_section_uid=section["source_section_uid"],
                    order_index=section["order_index"],
                    retrieval_family=section["retrieval_family"],
                    canonical_section_name=section["canonical_section_name"],
                    nearest_canonical_section_name=section[
                        "nearest_canonical_section_name"
                    ],
                    heading_path=heading_path,
                    direct_text=section["direct_text"],
                    max_words=max_words,
                    overlap_words=overlap_words,
                )

                for chunk in chunks:
                    insert_chunk(cur, section, chunk)

                total_chunks += len(chunks)

    print("\nSECTION CHUNK BUILD COMPLETE")
    print("=" * 40)
    print(f"Retrievable sections processed: {len(sections)}")
    print(f"Chunks created: {total_chunks}")
    print(f"Chunk max words: {max_words}")
    print(f"Chunk overlap words: {overlap_words}")


if __name__ == "__main__":
    main()