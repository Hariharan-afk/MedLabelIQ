from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from medlabeliq.config.settings import settings
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelHistoryRow:
    concept_name: str
    set_id: str
    locked_spl_version: int
    history_spl_version: str
    published_date: str
    title: str
    is_locked_version: bool


def fetch_history(client: httpx.Client, set_id: str) -> dict[str, Any]:
    """Fetch raw DailyMed version history for one SET ID."""
    url = f"{settings.dailymed_base_url}/spls/{set_id}/history.json"
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def save_raw_history(set_id: str, payload: dict[str, Any]) -> Path:
    """Persist the raw source response exactly as received."""
    output_dir = settings.raw_history_dir / set_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "history.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path


def normalize_history_rows(
    concept_name: str,
    set_id: str,
    locked_spl_version: int,
    payload: dict[str, Any],
) -> list[LabelHistoryRow]:
    """
    Convert the raw DailyMed history payload into normalized rows.

    DailyMed history response shape:
    {
        "data": {
            "spl": {
                "title": "...",
                "setid": "..."
            },
            "history": [
                {
                    "spl_version": 8,
                    "published_date": "Apr 30, 2026"
                }
            ]
        }
    }
    """
    data = payload.get("data", {})
    spl_metadata = data.get("spl", {})
    history_items = data.get("history", [])

    title = str(spl_metadata.get("title", "")).strip()

    rows: list[LabelHistoryRow] = []

    for item in history_items:
        spl_version = str(item.get("spl_version", "")).strip()
        published_date = str(item.get("published_date", "")).strip()

        rows.append(
            LabelHistoryRow(
                concept_name=concept_name,
                set_id=set_id,
                locked_spl_version=locked_spl_version,
                history_spl_version=spl_version,
                published_date=published_date,
                title=title,
                is_locked_version=spl_version == str(locked_spl_version),
            )
        )

    return rows


def write_history_csv(rows: list[LabelHistoryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "concept_name",
        "set_id",
        "locked_spl_version",
        "history_spl_version",
        "published_date",
        "title",
        "is_locked_version",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    seeds = load_locked_smoke_set(settings.smoke_set_path)
    if not seeds:
        raise RuntimeError("No locked smoke-set labels found in smoke_set.yaml.")

    headers = {"User-Agent": settings.http_user_agent}
    timeout = httpx.Timeout(settings.http_timeout_seconds)

    all_rows: list[LabelHistoryRow] = []

    with httpx.Client(headers=headers, timeout=timeout) as client:
        for seed in seeds:
            assert seed.set_id is not None
            assert seed.locked_spl_version is not None

            payload = fetch_history(client, seed.set_id)
            raw_path = save_raw_history(seed.set_id, payload)

            rows = normalize_history_rows(
                concept_name=seed.concept_name,
                set_id=seed.set_id,
                locked_spl_version=seed.locked_spl_version,
                payload=payload,
            )
            all_rows.extend(rows)

            LOGGER.info(
                "Fetched %s history entries for %s -> %s",
                len(rows),
                seed.concept_name,
                raw_path,
            )

    write_history_csv(all_rows, settings.label_history_csv_path)

    LOGGER.info(
        "Wrote %s normalized history rows to %s",
        len(all_rows),
        settings.label_history_csv_path,
    )
    LOGGER.info(
        "Completed label-history fetch at %s",
        datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    main()