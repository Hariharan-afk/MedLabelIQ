from __future__ import annotations

from medlabeliq.db.connection import get_connection


CHUNK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS section_chunk (
    chunk_id UUID PRIMARY KEY,

    section_id UUID NOT NULL REFERENCES label_section(section_id) ON DELETE CASCADE,
    label_version_id UUID NOT NULL REFERENCES label_version(label_version_id) ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'section_text',

    retrieval_family TEXT,
    canonical_section_name TEXT,
    nearest_canonical_section_name TEXT,

    heading_path JSONB NOT NULL DEFAULT '[]'::jsonb,

    chunk_text TEXT NOT NULL,
    embedding_text TEXT NOT NULL,

    token_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,

    start_word_index INTEGER NOT NULL,
    end_word_index INTEGER NOT NULL,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(section_id, chunk_index, chunk_type)
);

CREATE INDEX IF NOT EXISTS idx_section_chunk_section_id
ON section_chunk(section_id);

CREATE INDEX IF NOT EXISTS idx_section_chunk_label_version_id
ON section_chunk(label_version_id);

CREATE INDEX IF NOT EXISTS idx_section_chunk_retrieval_family
ON section_chunk(retrieval_family);

CREATE INDEX IF NOT EXISTS idx_section_chunk_canonical_section_name
ON section_chunk(canonical_section_name);

CREATE INDEX IF NOT EXISTS idx_section_chunk_nearest_canonical_section_name
ON section_chunk(nearest_canonical_section_name);

CREATE INDEX IF NOT EXISTS idx_section_chunk_embedding_text_fts
ON section_chunk USING GIN (to_tsvector('english', COALESCE(embedding_text, '')));
"""


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CHUNK_SCHEMA_SQL)

    print("Section chunk schema created successfully.")


if __name__ == "__main__":
    main()