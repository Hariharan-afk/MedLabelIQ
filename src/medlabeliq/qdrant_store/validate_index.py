from __future__ import annotations

from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection
from medlabeliq.qdrant_store.client import get_qdrant_client


def main() -> None:
    client = get_qdrant_client()

    collection_info = client.get_collection(settings.qdrant_collection)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM section_chunk;")
            db_count = int(cur.fetchone()["count"])

    qdrant_count = collection_info.points_count

    print("\nQDRANT INDEX VALIDATION")
    print("=" * 80)
    print(f"Collection: {settings.qdrant_collection}")
    print(f"DB section_chunk count: {db_count}")
    print(f"Qdrant point count: {qdrant_count}")
    print(f"Vector size: {collection_info.config.params.vectors.size}")
    print(f"Distance: {collection_info.config.params.vectors.distance}")

    if qdrant_count != db_count:
        print("\nVALIDATION STATUS: FAIL")
        print("=" * 80)
        print(f"Expected {db_count} Qdrant points, found {qdrant_count}.")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: PASS")
    print("=" * 80)
    print("Qdrant index matches PostgreSQL section_chunk table.")


if __name__ == "__main__":
    main()