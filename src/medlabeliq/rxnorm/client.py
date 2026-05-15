from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from medlabeliq.config.settings import settings


class RxNormClient:
    """
    Thin client for the official RxNorm REST API.

    This client intentionally exposes only the subset needed for MedLabelIQ's
    drug-term normalization workflow.
    """

    def __init__(self) -> None:
        self._client = httpx.Client(
            base_url=settings.rxnorm_base_url,
            timeout=settings.rxnorm_timeout_seconds,
            headers={
                "User-Agent": settings.http_user_agent,
                "Accept": "application/json",
            },
        )

    def __enter__(self) -> "RxNormClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.get(
            path,
            params=params,
        )
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected RxNorm JSON payload type: {type(payload).__name__}"
            )

        return payload

    def get_version(self) -> dict[str, str | None]:
        payload = self._get_json("/version.json")

        return {
            "version": payload.get("version"),
            "api_version": payload.get("apiVersion"),
        }

    def find_rxcuis_by_string(
        self,
        term: str,
        *,
        search: int = 2,
    ) -> list[str]:
        """
        Resolve a string using RxNorm's exact-or-normalized lookup.

        search=2 means:
        - exact match first,
        - normalized match if exact fails.
        """
        payload = self._get_json(
            "/rxcui.json",
            params={
                "name": term,
                "search": search,
                "allsrc": 0,
            },
        )

        id_group = payload.get("idGroup") or {}
        raw_ids = id_group.get("rxnormId") or []

        return [
            str(rxcui)
            for rxcui in raw_ids
            if rxcui is not None
        ]

    def get_approximate_matches(
        self,
        term: str,
        *,
        max_entries: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Resolve a noisy or misspelled term using RxNorm approximate matching.

        option=1 restricts the search to Active RxNorm concepts.
        """
        payload = self._get_json(
            "/approximateTerm.json",
            params={
                "term": term,
                "maxEntries": (
                    max_entries
                    if max_entries is not None
                    else settings.rxnorm_approximate_max_entries
                ),
                "option": 1,
            },
        )

        approximate_group = payload.get("approximateGroup") or {}
        raw_candidates = approximate_group.get("candidate") or []

        if isinstance(raw_candidates, dict):
            raw_candidates = [raw_candidates]

        return [
            candidate
            for candidate in raw_candidates
            if isinstance(candidate, dict)
        ]

    def get_concept_properties(
        self,
        rxcui: str,
    ) -> dict[str, Any] | None:
        payload = self._get_json(
            f"/rxcui/{rxcui}/properties.json",
        )

        properties = payload.get("properties")
        if not isinstance(properties, dict):
            return None

        return properties

    def get_related_by_type(
        self,
        rxcui: str,
        *,
        term_types: Iterable[str],
    ) -> list[dict[str, Any]]:
        """
        Get related concepts of specific RxNorm term types.

        For normalization we primarily use:
        - IN  = Ingredient
        - PIN = Precise Ingredient
        """
        tty_value = " ".join(term_types)

        payload = self._get_json(
            f"/rxcui/{rxcui}/related.json",
            params={
                "tty": tty_value,
            },
        )

        related_group = payload.get("relatedGroup") or {}
        concept_groups = related_group.get("conceptGroup") or []

        if isinstance(concept_groups, dict):
            concept_groups = [concept_groups]

        related: list[dict[str, Any]] = []

        for group in concept_groups:
            if not isinstance(group, dict):
                continue

            concepts = group.get("conceptProperties") or []

            if isinstance(concepts, dict):
                concepts = [concepts]

            for concept in concepts:
                if isinstance(concept, dict):
                    related.append(concept)

        return related