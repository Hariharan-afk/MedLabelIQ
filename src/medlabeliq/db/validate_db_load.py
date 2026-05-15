from __future__ import annotations

from medlabeliq.db.connection import get_connection


VALIDATION_QUERIES = {
    "label_document": "SELECT COUNT(*) AS count FROM label_document;",
    "label_version": "SELECT COUNT(*) AS count FROM label_version;",
    "label_product": "SELECT COUNT(*) AS count FROM label_product;",
    "product_ingredient": "SELECT COUNT(*) AS count FROM product_ingredient;",
    "label_section": "SELECT COUNT(*) AS count FROM label_section;",
}


def fetch_count(cur, query: str) -> int:
    cur.execute(query)
    row = cur.fetchone()
    return int(row["count"])


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            counts = {
                name: fetch_count(cur, query)
                for name, query in VALIDATION_QUERIES.items()
            }

            cur.execute(
                """
                SELECT 
                    d.concept_name,
                    COUNT(s.section_id) AS section_count,
                    COUNT(*) FILTER (WHERE s.retrieval_family IS NOT NULL) AS retrieval_family_count,
                    COUNT(*) FILTER (WHERE s.retrieval_family IS NULL) AS retrieval_family_missing_count
                FROM label_document d
                JOIN label_version v ON v.label_id = d.label_id
                JOIN label_section s ON s.label_version_id = v.label_version_id
                GROUP BY d.concept_name
                ORDER BY d.concept_name;
                """
            )
            section_rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                    retrieval_family,
                    COUNT(*) AS count
                FROM label_section
                WHERE retrieval_family IS NOT NULL
                GROUP BY retrieval_family
                ORDER BY count DESC, retrieval_family;
                """
            )
            family_rows = cur.fetchall()

    print("\nDATABASE LOAD VALIDATION")
    print("=" * 40)

    print("\nTable counts:")
    for table_name, count in counts.items():
        print(f"  - {table_name}: {count}")

    print("\nSection coverage by concept:")
    for row in section_rows:
        print(
            f"  - {row['concept_name']}: "
            f"{row['section_count']} sections | "
            f"{row['retrieval_family_count']} retrieval-family covered | "
            f"{row['retrieval_family_missing_count']} missing"
        )

    print("\nTop retrieval families:")
    for row in family_rows[:20]:
        print(f"  - {row['retrieval_family']}: {row['count']}")

    expected = {
        "label_document": 12,
        "label_version": 12,
        "label_section": 663,
    }

    errors = []

    for table_name, expected_count in expected.items():
        actual = counts[table_name]
        if actual != expected_count:
            errors.append(
                f"{table_name}: expected {expected_count}, found {actual}"
            )

    if errors:
        print("\nVALIDATION STATUS: FAIL")
        print("=" * 40)
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: PASS")
    print("=" * 40)
    print("Parsed labels were loaded into PostgreSQL successfully.")


if __name__ == "__main__":
    main()