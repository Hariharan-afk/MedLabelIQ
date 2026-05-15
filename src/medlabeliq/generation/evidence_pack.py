from __future__ import annotations

from dataclasses import dataclass

from medlabeliq.config.settings import settings
from medlabeliq.retrieval.hybrid_search import (
    HybridSearchResult,
    hybrid_search_chunks,
)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    chunk_id: str
    section_id: str
    concept_name: str
    retrieval_family: str
    canonical_section_name: str | None
    nearest_canonical_section_name: str | None
    heading_path: list[str]
    set_id: str
    version_number: int
    chunk_text: str
    hybrid_score: float
    lexical_rank: int | None
    dense_rank: int | None

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def source_label(self) -> str:
        return f"DailyMed SET ID {self.set_id}, SPL version {self.version_number}"


@dataclass(frozen=True)
class EvidencePack:
    query: str
    concept_name: str | None
    retrieval_family: str | None
    evidence_items: list[EvidenceItem]

    @property
    def evidence_ids(self) -> set[str]:
        return {item.evidence_id for item in self.evidence_items}

    def to_prompt_block(self) -> str:
        if not self.evidence_items:
            return "NO EVIDENCE RETRIEVED."

        blocks: list[str] = []

        for item in self.evidence_items:
            blocks.append(
                "\n".join(
                    [
                        f"[{item.evidence_id}]",
                        f"Drug concept: {item.concept_name}",
                        f"Retrieval family: {item.retrieval_family}",
                        f"Canonical section: {item.canonical_section_name or 'unknown'}",
                        (
                            "Nearest canonical section: "
                            f"{item.nearest_canonical_section_name or 'unknown'}"
                        ),
                        f"Heading path: {item.heading}",
                        f"Source: {item.source_label}",
                        "Evidence text:",
                        item.chunk_text,
                    ]
                )
            )

        return "\n\n".join(blocks)


def select_diverse_results(
    results: list[HybridSearchResult],
    *,
    top_k: int,
    max_per_section: int,
) -> list[HybridSearchResult]:
    """
    Select a compact, diverse evidence set.

    Current policy:
    - preserve retrieval ranking order,
    - prefer distinct label sections,
    - limit the number of chunks from the same section,
    - return up to top_k items, not necessarily exactly top_k.
    """
    if top_k <= 0:
        return []

    selected: list[HybridSearchResult] = []
    section_counts: dict[str, int] = {}

    for result in results:
        count = section_counts.get(result.section_id, 0)

        if count >= max_per_section:
            continue

        selected.append(result)
        section_counts[result.section_id] = count + 1

        if len(selected) >= top_k:
            break

    return selected


def build_evidence_pack(
    query: str,
    *,
    concept_name: str | None = None,
    retrieval_family: str | None = None,
    top_k: int | None = None,
) -> EvidencePack:
    """
    Retrieve hybrid candidates and reduce them to a compact, diverse evidence pack.
    """
    final_top_k = top_k or settings.answer_top_k

    candidate_results = hybrid_search_chunks(
        query=query,
        concept_name=concept_name,
        retrieval_family=retrieval_family,
        limit=settings.answer_candidate_pool,
        candidate_pool=30,
    )

    # Only diversify within the originally strongest top-k retrieval window.
    # This prevents lower-ranked, weakly related sections from being backfilled
    # just because duplicate chunks from the best section were removed.
    selection_window = candidate_results[:final_top_k]

    selected_results = select_diverse_results(
        selection_window,
        top_k=final_top_k,
        max_per_section=settings.evidence_max_per_section,
    )
    
    evidence_items = [
        EvidenceItem(
            evidence_id=f"E{idx}",
            chunk_id=result.chunk_id,
            section_id=result.section_id,
            concept_name=result.concept_name,
            retrieval_family=result.retrieval_family,
            canonical_section_name=result.canonical_section_name,
            nearest_canonical_section_name=result.nearest_canonical_section_name,
            heading_path=result.heading_path,
            set_id=result.set_id,
            version_number=result.version_number,
            chunk_text=result.chunk_text,
            hybrid_score=result.hybrid_score,
            lexical_rank=result.lexical_rank,
            dense_rank=result.dense_rank,
        )
        for idx, result in enumerate(selected_results, start=1)
    ]

    return EvidencePack(
        query=query,
        concept_name=concept_name,
        retrieval_family=retrieval_family,
        evidence_items=evidence_items,
    )