from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from medlabeliq.config.settings import settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrugSeed:
    concept_name: str
    query_name: str
    name_type: str
    target_profile: str
    prefer_title_contains: list[str]
    exclude_title_contains: list[str]
    set_id: str | None
    locked_title: str | None
    notes: str


@dataclass(frozen=True)
class LabelCandidate:
    concept_name: str
    query_name: str
    name_type: str
    setid: str
    spl_version: str
    title: str
    published_date: str
    score: int
    matched_preferred_tokens: list[str]
    matched_excluded_tokens: list[str]


def load_smoke_set(path: Path) -> list[DrugSeed]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    drugs = payload.get("drugs", [])
    return [DrugSeed(**drug) for drug in drugs]


def score_candidate(seed: DrugSeed, title: str) -> tuple[int, list[str], list[str]]:
    """
    Heuristic ranking only. This is *not* an automatic label selector.

    Positive tokens nudge likely formulation matches upward.
    Exclusion tokens heavily penalize likely out-of-scope labels.
    """
    normalized_title = title.upper()

    matched_preferred = [
        token for token in seed.prefer_title_contains if token.upper() in normalized_title
    ]
    matched_excluded = [
        token for token in seed.exclude_title_contains if token.upper() in normalized_title
    ]

    score = len(matched_preferred) * 10
    score -= len(matched_excluded) * 100

    # Small preference for titles that contain the concept name itself.
    if seed.concept_name.upper() in normalized_title:
        score += 5

    return score, matched_preferred, matched_excluded


def fetch_candidates(
    client: httpx.Client,
    seed: DrugSeed,
    pagesize: int = 100,
) -> list[LabelCandidate]:
    """
    Fetch all DailyMed SPL candidates for one seed across every available page.

    DailyMed caps pagesize at 100, so queries for common drugs such as
    acetaminophen or metformin can span multiple pages.
    """
    url = f"{settings.dailymed_base_url}/spls.json"

    all_items: list[dict[str, Any]] = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        params = {
            "drug_name": seed.query_name,
            "name_type": seed.name_type,
            "pagesize": pagesize,
            "page": page,
        }

        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

        metadata = payload.get("metadata", {})
        total_pages = int(metadata.get("total_pages", 1))
        all_items.extend(payload.get("data", []))

        page += 1

    candidates: list[LabelCandidate] = []

    for item in all_items:
        title = str(item.get("title", "")).strip()
        score, preferred, excluded = score_candidate(seed, title)

        candidates.append(
            LabelCandidate(
                concept_name=seed.concept_name,
                query_name=seed.query_name,
                name_type=seed.name_type,
                setid=str(item.get("setid", "")).strip(),
                spl_version=str(item.get("spl_version", "")).strip(),
                title=title,
                published_date=str(item.get("published_date", "")).strip(),
                score=score,
                matched_preferred_tokens=preferred,
                matched_excluded_tokens=excluded,
            )
        )

    # Defensive deduplication in case an API response ever repeats records.
    unique_candidates = {
        (candidate.setid, candidate.spl_version): candidate
        for candidate in candidates
    }

    return sorted(
        unique_candidates.values(),
        key=lambda c: (
            c.score,
            _safe_parse_date(c.published_date),
            c.spl_version,
        ),
        reverse=True,
    )


def _safe_parse_date(value: str) -> datetime:
    for fmt in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.min


def write_csv(candidates: list[LabelCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(candidates[0]).keys()) if candidates else [
        "concept_name",
        "query_name",
        "name_type",
        "setid",
        "spl_version",
        "title",
        "published_date",
        "score",
        "matched_preferred_tokens",
        "matched_excluded_tokens",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            row = asdict(candidate)
            row["matched_preferred_tokens"] = "|".join(row["matched_preferred_tokens"])
            row["matched_excluded_tokens"] = "|".join(row["matched_excluded_tokens"])
            writer.writerow(row)


def write_json(candidates: list[LabelCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(candidate) for candidate in candidates], f, indent=2)


def build_manifest(only_unlocked: bool = False) -> list[LabelCandidate]:
    seeds = load_smoke_set(settings.smoke_set_path)
    if only_unlocked:
        seeds = [seed for seed in seeds if seed.set_id is None]

    headers = {"User-Agent": settings.http_user_agent}
    timeout = httpx.Timeout(settings.http_timeout_seconds)

    all_candidates: list[LabelCandidate] = []
    with httpx.Client(headers=headers, timeout=timeout) as client:
        for seed in seeds:
            try:
                candidates = fetch_candidates(client, seed)
                LOGGER.info("Fetched %s candidates for %s", len(candidates), seed.concept_name)
                all_candidates.extend(candidates)
            except httpx.HTTPError as exc:
                LOGGER.exception("Failed to fetch candidates for %s: %s", seed.concept_name, exc)

    return all_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and rank DailyMed label candidates for the Phase 1 smoke set."
    )
    parser.add_argument(
        "--only-unlocked",
        action="store_true",
        help="Skip drugs that already have a locked SET ID in smoke_set.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    candidates = build_manifest(only_unlocked=args.only_unlocked)
    write_csv(candidates, settings.label_candidates_csv_path)
    write_json(candidates, settings.label_candidates_json_path)

    LOGGER.info("Wrote %s candidates to %s", len(candidates), settings.label_candidates_csv_path)
    LOGGER.info("Wrote JSON manifest to %s", settings.label_candidates_json_path)


if __name__ == "__main__":
    main()
