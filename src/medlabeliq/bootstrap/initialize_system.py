from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection
from medlabeliq.qdrant_store.client import get_qdrant_client


# =============================================================================
# Step model
# =============================================================================

@dataclass(frozen=True)
class BootstrapStep:
    phase: str
    name: str
    module: str
    validation_step: bool = False
    ingestion_step: bool = False


BOOTSTRAP_STEPS: list[BootstrapStep] = [
    # -------------------------------------------------------------------------
    # Corpus reconstruction
    # -------------------------------------------------------------------------
    BootstrapStep(
        phase="Corpus ingestion",
        name="Fetch locked DailyMed label history",
        module="medlabeliq.ingestion.fetch_label_history",
        ingestion_step=True,
    ),
    BootstrapStep(
        phase="Corpus ingestion",
        name="Download locked SPL XML files",
        module="medlabeliq.ingestion.download_spl_xml",
        ingestion_step=True,
    ),
    BootstrapStep(
        phase="Corpus ingestion",
        name="Validate downloaded smoke-set artifacts",
        module="medlabeliq.validation.validate_step3_artifacts",
        validation_step=True,
    ),

    # -------------------------------------------------------------------------
    # XML parsing and section validation
    # -------------------------------------------------------------------------
    BootstrapStep(
        phase="Structured parsing",
        name="Parse smoke-set SPL XML labels",
        module="medlabeliq.parsing.parse_smoke_set",
    ),
    BootstrapStep(
        phase="Structured parsing",
        name="Validate section hierarchy and retrieval-family inheritance",
        module="medlabeliq.validation.validate_section_hierarchy",
        validation_step=True,
    ),

    # -------------------------------------------------------------------------
    # PostgreSQL structured store
    # -------------------------------------------------------------------------
    BootstrapStep(
        phase="PostgreSQL knowledge base",
        name="Create structured label schema",
        module="medlabeliq.db.create_schema",
    ),
    BootstrapStep(
        phase="PostgreSQL knowledge base",
        name="Load parsed labels into PostgreSQL",
        module="medlabeliq.db.load_parsed_labels",
    ),
    BootstrapStep(
        phase="PostgreSQL knowledge base",
        name="Validate PostgreSQL label load",
        module="medlabeliq.db.validate_db_load",
        validation_step=True,
    ),

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------
    BootstrapStep(
        phase="Section-aware chunking",
        name="Create chunk schema",
        module="medlabeliq.db.create_chunk_schema",
    ),
    BootstrapStep(
        phase="Section-aware chunking",
        name="Build section-aware retrieval chunks",
        module="medlabeliq.chunking.build_section_chunks",
    ),
    BootstrapStep(
        phase="Section-aware chunking",
        name="Validate section chunks",
        module="medlabeliq.chunking.validate_section_chunks",
        validation_step=True,
    ),

    # -------------------------------------------------------------------------
    # Qdrant vector index
    # -------------------------------------------------------------------------
    BootstrapStep(
        phase="Qdrant vector index",
        name="Embed and index chunks into Qdrant",
        module="medlabeliq.qdrant_store.index_chunks",
    ),
    BootstrapStep(
        phase="Qdrant vector index",
        name="Validate Qdrant collection",
        module="medlabeliq.qdrant_store.validate_index",
        validation_step=True,
    ),

    BootstrapStep(
        phase="Corpus metadata",
        name="Create corpus metadata schema",
        module="medlabeliq.db.create_corpus_metadata_schema",
    ),
    BootstrapStep(
        phase="Corpus metadata",
        name="Record current corpus build metadata",
        module="medlabeliq.corpus.record_corpus_build",
    ),

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------
    BootstrapStep(
        phase="Observability",
        name="Create QA observability schema",
        module="medlabeliq.db.create_observability_schema",
    ),
]


# =============================================================================
# Preflight checks
# =============================================================================

def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def require_seed_file() -> None:
    if not settings.smoke_set_path.exists():
        raise FileNotFoundError(
            f"Smoke-set seed file not found: {settings.smoke_set_path}"
        )

    print(f"[OK] Smoke-set seed found: {settings.smoke_set_path}")


def require_postgres_connection() -> None:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok;")
                row = cur.fetchone()

        if row is None:
            raise RuntimeError("PostgreSQL query returned no row.")

        print(
            "[OK] PostgreSQL reachable at "
            f"{settings.postgres_host}:{settings.postgres_port}/"
            f"{settings.postgres_db}"
        )

    except Exception as exc:
        raise RuntimeError(
            "PostgreSQL preflight failed. "
            "Start PostgreSQL and confirm your .env connection settings."
        ) from exc


def require_qdrant_connection() -> None:
    try:
        client = get_qdrant_client()
        client.get_collections()

        print(
            "[OK] Qdrant reachable at "
            f"{settings.qdrant_url}"
        )

    except Exception as exc:
        raise RuntimeError(
            "Qdrant preflight failed. "
            "Start Qdrant and confirm QDRANT_HOST/QDRANT_PORT settings."
        ) from exc


def run_preflight() -> None:
    print_header("BOOTSTRAP PREFLIGHT")
    require_seed_file()
    require_postgres_connection()
    require_qdrant_connection()


# =============================================================================
# Execution helpers
# =============================================================================

def build_selected_steps(
    *,
    skip_ingestion: bool,
    skip_validations: bool,
) -> list[BootstrapStep]:
    selected: list[BootstrapStep] = []

    for step in BOOTSTRAP_STEPS:
        if skip_ingestion and step.ingestion_step:
            continue

        if skip_validations and step.validation_step:
            continue

        selected.append(step)

    return selected


def print_execution_plan(steps: list[BootstrapStep]) -> None:
    print_header("BOOTSTRAP EXECUTION PLAN")

    current_phase: str | None = None

    for index, step in enumerate(steps, start=1):
        if step.phase != current_phase:
            current_phase = step.phase
            print(f"\n{current_phase}")

        print(f"  {index:02d}. {step.name}")
        print(f"      python -m {step.module}")


def run_step(
    *,
    step: BootstrapStep,
    index: int,
    total: int,
) -> float:
    print()
    print("-" * 88)
    print(f"[{index}/{total}] {step.phase} — {step.name}")
    print(f"Module: {step.module}")
    print("-" * 88)

    started = time.perf_counter()

    command = [
        sys.executable,
        "-m",
        step.module,
    ]

    completed = subprocess.run(
        command,
        cwd=str(settings.project_root),
        check=False,
    )

    elapsed_seconds = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            f"Bootstrap step failed: {step.name} "
            f"(module={step.module}, exit_code={completed.returncode})"
        )

    print(
        f"[PASS] {step.name} "
        f"completed in {elapsed_seconds:.2f} second(s)."
    )

    return elapsed_seconds


def execute_steps(steps: list[BootstrapStep]) -> None:
    print_header("MEDLABELIQ SYSTEM BOOTSTRAP")

    total_started = time.perf_counter()
    timings: list[tuple[BootstrapStep, float]] = []

    for index, step in enumerate(steps, start=1):
        elapsed = run_step(
            step=step,
            index=index,
            total=len(steps),
        )
        timings.append((step, elapsed))

    total_elapsed = time.perf_counter() - total_started

    print_header("BOOTSTRAP COMPLETE")
    print(f"Steps completed: {len(steps)}")
    print(f"Total time: {total_elapsed:.2f} second(s)")

    print("\nStep timings:")
    for step, elapsed in timings:
        print(f"  - {step.name}: {elapsed:.2f} second(s)")

    print("\nThe MedLabelIQ corpus, PostgreSQL store, Qdrant index, and")
    print("observability schema are now initialized.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the full MedLabelIQ system from the locked smoke-set "
            "seed file: ingestion, parsing, database load, chunking, indexing, "
            "and observability schema creation."
        )
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the selected bootstrap plan without executing it.",
    )

    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help=(
            "Skip DailyMed history fetch and SPL XML download. "
            "Use only when raw artifacts already exist locally."
        ),
    )

    parser.add_argument(
        "--skip-validations",
        action="store_true",
        help="Skip validation/reporting steps and run only build/load/index steps.",
    )

    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip seed/PostgreSQL/Qdrant connectivity checks.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_steps = build_selected_steps(
        skip_ingestion=args.skip_ingestion,
        skip_validations=args.skip_validations,
    )

    if not selected_steps:
        raise RuntimeError("No bootstrap steps selected.")

    print_execution_plan(selected_steps)

    if args.plan:
        print("\nPlan-only mode: no bootstrap steps were executed.")
        return

    if not args.skip_preflight:
        run_preflight()

    execute_steps(selected_steps)


if __name__ == "__main__":
    main()