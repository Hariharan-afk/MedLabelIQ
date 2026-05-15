from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DrugSeed:
    concept_name: str
    query_name: str
    name_type: str
    target_profile: str
    prefer_title_contains: list[str]
    exclude_title_contains: list[str]
    set_id: str | None
    locked_spl_version: int | None
    locked_published_date: str | None
    locked_title: str | None
    lock_reason: str | None
    notes: str


def load_smoke_set(path: Path) -> list[DrugSeed]:
    """Load the locked smoke-set YAML file into typed DrugSeed objects."""
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    drugs = payload.get("drugs", [])
    return [DrugSeed(**drug) for drug in drugs]


def load_locked_smoke_set(path: Path) -> list[DrugSeed]:
    """
    Return only seeds that have both a locked SET ID and a locked SPL version.

    Step 3 should operate only on locked labels because we want deterministic
    downloads rather than 'whatever is latest today'.
    """
    seeds = load_smoke_set(path)
    return [
        seed
        for seed in seeds
        if seed.set_id is not None and seed.locked_spl_version is not None
    ]