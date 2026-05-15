from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from medlabeliq.config.settings import settings
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_download_manifest(
    seeds,
    manifest_rows: list[dict[str, str]],
) -> list[str]:
    errors: list[str] = []

    if len(manifest_rows) != len(seeds):
        errors.append(
            f"Expected {len(seeds)} download-manifest rows, found {len(manifest_rows)}."
        )

    rows_by_concept = {row["concept_name"]: row for row in manifest_rows}

    for seed in seeds:
        row = rows_by_concept.get(seed.concept_name)

        if row is None:
            errors.append(f"Missing download-manifest row for {seed.concept_name}.")
            continue

        if row["set_id"] != seed.set_id:
            errors.append(
                f"{seed.concept_name}: manifest SET ID {row['set_id']} "
                f"does not match locked SET ID {seed.set_id}."
            )

        if row["locked_spl_version"] != str(seed.locked_spl_version):
            errors.append(
                f"{seed.concept_name}: manifest version {row['locked_spl_version']} "
                f"does not match locked version {seed.locked_spl_version}."
            )

        zip_path = Path(row["zip_path"])
        xml_path = Path(row["xml_path"])

        if not zip_path.exists():
            errors.append(f"{seed.concept_name}: missing ZIP file at {zip_path}.")
        else:
            actual_zip_sha = sha256_file(zip_path)
            if actual_zip_sha != row["zip_sha256"]:
                errors.append(f"{seed.concept_name}: ZIP checksum mismatch.")

        if not xml_path.exists():
            errors.append(f"{seed.concept_name}: missing XML file at {xml_path}.")
        else:
            actual_xml_sha = sha256_file(xml_path)
            if actual_xml_sha != row["xml_sha256"]:
                errors.append(f"{seed.concept_name}: XML checksum mismatch.")

    return errors


def validate_history(
    seeds,
    history_rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    history_counts: dict[str, int] = defaultdict(int)
    locked_version_found: dict[str, bool] = defaultdict(bool)

    for row in history_rows:
        concept_name = row["concept_name"]
        history_counts[concept_name] += 1

        if row["is_locked_version"].lower() == "true":
            locked_version_found[concept_name] = True

    for seed in seeds:
        if history_counts[seed.concept_name] == 0:
            errors.append(f"{seed.concept_name}: no history rows found.")

        if not locked_version_found[seed.concept_name]:
            errors.append(
                f"{seed.concept_name}: locked version {seed.locked_spl_version} "
                f"was not found in DailyMed history."
            )

    return errors, dict(history_counts)


def main() -> None:
    seeds = load_locked_smoke_set(settings.smoke_set_path)

    manifest_rows = read_csv_rows(settings.download_manifest_csv_path)
    history_rows = read_csv_rows(settings.label_history_csv_path)

    errors: list[str] = []
    errors.extend(validate_download_manifest(seeds, manifest_rows))

    history_errors, history_counts = validate_history(seeds, history_rows)
    errors.extend(history_errors)

    print("\nSTEP 3 VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Locked smoke-set labels: {len(seeds)}")
    print(f"Download-manifest rows: {len(manifest_rows)}")
    print(f"History rows: {len(history_rows)}")
    print("\nHistory entries by concept:")
    for concept_name in sorted(history_counts):
        print(f"  - {concept_name}: {history_counts[concept_name]}")

    if errors:
        print("\nVALIDATION STATUS: FAIL")
        print("=" * 40)
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: PASS")
    print("=" * 40)
    print("All locked labels, files, checksums, and history entries are consistent.")


if __name__ == "__main__":
    main()