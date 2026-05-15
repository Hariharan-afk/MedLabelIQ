from __future__ import annotations

from medlabeliq.db.connection import get_connection


DDL = """
CREATE TABLE IF NOT EXISTS corpus_build_log (
    build_id UUID PRIMARY KEY,
    built_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    build_source TEXT NOT NULL,
    seed_file_path TEXT NOT NULL,

    drug_count INTEGER NOT NULL,
    label_document_count INTEGER NOT NULL,
    label_version_count INTEGER NOT NULL,
    product_count INTEGER NOT NULL,
    ingredient_count INTEGER NOT NULL,
    section_count INTEGER NOT NULL,
    retrievable_section_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    retrieval_family_count INTEGER NOT NULL,

    qdrant_collection TEXT NOT NULL,
    qdrant_point_count INTEGER NOT NULL,
    embedding_model_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corpus_build_log_built_at
    ON corpus_build_log (built_at DESC);
"""


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)

        conn.commit()

    print("Corpus metadata schema created successfully.")


if __name__ == "__main__":
    main()