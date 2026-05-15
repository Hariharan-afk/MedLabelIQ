# MedLabelIQ — Phase 1 Starter

This starter package sets up the first two steps of the project:

1. Repository bootstrap
2. A reproducible smoke-set selection workflow for DailyMed labels

## What is included

- `data/seeds/smoke_set.yaml` — the initial 12-drug smoke set
- `src/medlabeliq/ingestion/fetch_label_manifest.py` — queries DailyMed and produces a ranked candidate manifest
- `src/medlabeliq/config/settings.py` — central path and API settings
- `docker-compose.yml` — local PostgreSQL service for later Phase 1 work
- `.env.example` — environment template
- `pyproject.toml` — project metadata and dependencies

## Setup

```bash
uv sync
cp .env.example .env
docker compose up -d postgres
```

## Generate candidate labels for the smoke set

```bash
uv run python -m medlabeliq.ingestion.fetch_label_manifest
```

Outputs:
- `data/interim/label_candidates.csv`
- `data/interim/label_candidates.json`

## Next manual step

Review `label_candidates.csv`, choose one current human-drug label per concept, and then fill the selected `set_id` and `locked_title` fields in `data/seeds/smoke_set.yaml`.

The project intentionally does **not** auto-lock labels yet. The first lock should be reviewed manually so we know exactly what enters the corpus.
