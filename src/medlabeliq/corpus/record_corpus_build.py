from __future__ import annotations

from medlabeliq.corpus.metadata import record_corpus_build


def main() -> None:
    build_id, stats = record_corpus_build(
        build_source="bootstrap",
    )

    print("\nCORPUS BUILD METADATA RECORDED")
    print("=" * 80)
    print(f"Build ID: {build_id}")
    print(f"Drugs: {stats['drug_count']}")
    print(f"Label documents: {stats['label_document_count']}")
    print(f"Label versions: {stats['label_version_count']}")
    print(f"Sections: {stats['section_count']}")
    print(f"Retrievable sections: {stats['retrievable_section_count']}")
    print(f"Chunks: {stats['chunk_count']}")
    print(f"Retrieval families: {stats['retrieval_family_count']}")
    print(f"Qdrant collection: {stats['qdrant_collection']}")
    print(f"Qdrant points: {stats['qdrant_point_count']}")
    print(f"Embedding model: {stats['embedding_model_name']}")


if __name__ == "__main__":
    main()