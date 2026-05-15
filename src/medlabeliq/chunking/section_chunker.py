from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    chunk_index: int
    chunk_text: str
    embedding_text: str
    token_count: int
    char_count: int
    start_word_index: int
    end_word_index: int
    metadata: dict[str, Any]


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split()).strip()


def word_count(value: str) -> int:
    return len(value.split())


def split_text_by_words(
    text: str,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[tuple[str, int, int]]:
    """
    Split text by approximate word count.

    This intentionally does not cross section boundaries. The caller passes
    one section's direct_text at a time.
    """
    normalized = normalize_text(text)
    words = normalized.split()

    if not words:
        return []

    if len(words) <= max_words:
        return [(" ".join(words), 0, len(words))]

    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words.")

    chunks: list[tuple[str, int, int]] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append((" ".join(chunk_words), start, end))

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


def build_embedding_text(
    concept_name: str,
    document_title: str | None,
    set_id: str,
    version_number: int,
    retrieval_family: str | None,
    canonical_section_name: str | None,
    nearest_canonical_section_name: str | None,
    heading_path: list[str],
    chunk_text: str,
) -> str:
    """
    Build retrieval-ready text with provenance and section context.

    This is what we will later embed into Qdrant.
    """
    heading = " > ".join(heading_path)

    parts = [
        f"Drug concept: {concept_name}",
        f"Label title: {document_title or 'Unknown'}",
        f"DailyMed SET ID: {set_id}",
        f"SPL version: {version_number}",
        f"Retrieval family: {retrieval_family or 'unknown'}",
        f"Canonical section: {canonical_section_name or 'unknown'}",
        f"Nearest canonical section: {nearest_canonical_section_name or 'unknown'}",
        f"Heading path: {heading}",
        "",
        chunk_text,
    ]

    return "\n".join(parts).strip()


def chunk_section_text(
    *,
    concept_name: str,
    document_title: str | None,
    set_id: str,
    version_number: int,
    section_id: str,
    source_section_uid: str,
    order_index: int,
    retrieval_family: str | None,
    canonical_section_name: str | None,
    nearest_canonical_section_name: str | None,
    heading_path: list[str],
    direct_text: str,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[ChunkRecord]:
    """
    Convert one section's direct_text into retrieval-ready chunks.
    """
    normalized_text = normalize_text(direct_text)
    if not normalized_text:
        return []

    split_chunks = split_text_by_words(
        normalized_text,
        max_words=max_words,
        overlap_words=overlap_words,
    )

    records: list[ChunkRecord] = []

    for idx, (chunk_text, start_word, end_word) in enumerate(split_chunks, start=1):
        embedding_text = build_embedding_text(
            concept_name=concept_name,
            document_title=document_title,
            set_id=set_id,
            version_number=version_number,
            retrieval_family=retrieval_family,
            canonical_section_name=canonical_section_name,
            nearest_canonical_section_name=nearest_canonical_section_name,
            heading_path=heading_path,
            chunk_text=chunk_text,
        )

        records.append(
            ChunkRecord(
                chunk_index=idx,
                chunk_text=chunk_text,
                embedding_text=embedding_text,
                token_count=word_count(chunk_text),
                char_count=len(chunk_text),
                start_word_index=start_word,
                end_word_index=end_word,
                metadata={
                    "concept_name": concept_name,
                    "set_id": set_id,
                    "version_number": version_number,
                    "section_id": section_id,
                    "source_section_uid": source_section_uid,
                    "section_order_index": order_index,
                    "retrieval_family": retrieval_family,
                    "canonical_section_name": canonical_section_name,
                    "nearest_canonical_section_name": nearest_canonical_section_name,
                    "heading_path": heading_path,
                    "chunking_strategy": "section_direct_text_word_window",
                    "max_words": max_words,
                    "overlap_words": overlap_words,
                },
            )
        )

    return records