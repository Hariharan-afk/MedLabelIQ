from __future__ import annotations

from medlabeliq.db.connection import get_connection


def fetch_one(cur, query: str):
    cur.execute(query)
    return cur.fetchone()


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM label_section
                WHERE retrieval_family IS NOT NULL
                  AND direct_text IS NOT NULL
                  AND LENGTH(TRIM(direct_text)) > 0;
                """
            )
            retrievable_sections = int(cur.fetchone()["count"])

            cur.execute(
                """
                SELECT COUNT(DISTINCT section_id) AS count
                FROM section_chunk;
                """
            )
            chunked_sections = int(cur.fetchone()["count"])

            cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM section_chunk;
                """
            )
            total_chunks = int(cur.fetchone()["count"])

            cur.execute(
                """
                SELECT MAX(token_count) AS max_token_count
                FROM section_chunk;
                """
            )
            max_token_count = cur.fetchone()["max_token_count"] or 0

            cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM section_chunk
                WHERE retrieval_family IS NULL;
                """
            )
            null_family_chunks = int(cur.fetchone()["count"])

            cur.execute(
                """
                SELECT
                    d.concept_name,
                    COUNT(c.chunk_id) AS chunk_count,
                    COUNT(DISTINCT c.section_id) AS chunked_section_count
                FROM section_chunk c
                JOIN label_version v ON v.label_version_id = c.label_version_id
                JOIN label_document d ON d.label_id = v.label_id
                GROUP BY d.concept_name
                ORDER BY d.concept_name;
                """
            )
            concept_rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                    retrieval_family,
                    COUNT(*) AS chunk_count
                FROM section_chunk
                GROUP BY retrieval_family
                ORDER BY chunk_count DESC, retrieval_family;
                """
            )
            family_rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                    c.chunk_id,
                    d.concept_name,
                    c.retrieval_family,
                    c.token_count,
                    c.heading_path
                FROM section_chunk c
                JOIN label_version v ON v.label_version_id = c.label_version_id
                JOIN label_document d ON d.label_id = v.label_id
                WHERE c.token_count > 220
                ORDER BY c.token_count DESC
                LIMIT 10;
                """
            )
            oversized_rows = cur.fetchall()

    print("\nSECTION CHUNK VALIDATION")
    print("=" * 40)
    print(f"Retrievable sections with text: {retrievable_sections}")
    print(f"Distinct chunked sections: {chunked_sections}")
    print(f"Total chunks: {total_chunks}")
    print(f"Max chunk token count: {max_token_count}")
    print(f"Chunks with null retrieval family: {null_family_chunks}")

    print("\nChunks by concept:")
    for row in concept_rows:
        print(
            f"  - {row['concept_name']}: "
            f"{row['chunk_count']} chunks from "
            f"{row['chunked_section_count']} sections"
        )

    print("\nTop chunk retrieval families:")
    for row in family_rows[:20]:
        print(f"  - {row['retrieval_family']}: {row['chunk_count']}")

    errors = []

    if total_chunks == 0:
        errors.append("No chunks were created.")

    if retrievable_sections != chunked_sections:
        errors.append(
            f"Expected chunks for {retrievable_sections} retrievable sections, "
            f"but found chunks for {chunked_sections} sections."
        )

    if null_family_chunks != 0:
        errors.append(f"Found {null_family_chunks} chunks with null retrieval_family.")

    if max_token_count > 220:
        errors.append(f"Found chunk token_count above 220: {max_token_count}")

    if oversized_rows:
        print("\nOversized chunk examples:")
        for row in oversized_rows:
            print(
                f"  - {row['concept_name']} | {row['retrieval_family']} | "
                f"{row['token_count']} tokens | {row['heading_path']}"
            )

    if errors:
        print("\nVALIDATION STATUS: FAIL")
        print("=" * 40)
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: PASS")
    print("=" * 40)
    print("Section-aware chunks were built successfully.")


if __name__ == "__main__":
    main()