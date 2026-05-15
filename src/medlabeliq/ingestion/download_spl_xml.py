from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from medlabeliq.config.settings import settings
from medlabeliq.ingestion.seed_loader import load_locked_smoke_set

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadManifestRow:
    concept_name: str
    set_id: str
    locked_spl_version: int
    locked_published_date: str | None
    locked_title: str | None
    zip_path: str
    xml_path: str
    zip_sha256: str
    xml_sha256: str
    xml_member_name: str
    dailymed_label_last_updated: str | None
    downloaded_at_utc: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_versioned_download_url(set_id: str, version: int) -> str:
    """
    Use the version-specific DailyMed ZIP endpoint.

    This is preferred over the 'latest' endpoint because the smoke set is
    intentionally locked to exact SPL versions.
    """
    return f"{settings.dailymed_download_base_url}/getFile.cfm"


def download_versioned_zip(
    client: httpx.Client,
    set_id: str,
    version: int,
) -> httpx.Response:
    url = build_versioned_download_url(set_id, version)
    params = {
        "type": "zip",
        "setid": set_id,
        "version": version,
    }

    response = client.get(url, params=params)
    response.raise_for_status()
    return response


def choose_main_xml_member(zf: zipfile.ZipFile) -> str:
    """
    Choose the main XML file from a DailyMed ZIP.

    In the normal case there is one SPL XML file. If more than one exists,
    choose the largest XML file and log a warning, because the main SPL document
    should generally be larger than auxiliary XML artifacts.
    """
    xml_members = [
        info for info in zf.infolist()
        if info.filename.lower().endswith(".xml")
    ]

    if not xml_members:
        raise RuntimeError("ZIP archive contained no XML files.")

    if len(xml_members) > 1:
        LOGGER.warning(
            "ZIP archive contained %s XML files; choosing the largest one.",
            len(xml_members),
        )

    selected = max(xml_members, key=lambda info: info.file_size)
    return selected.filename


def save_label_artifacts(
    concept_name: str,
    set_id: str,
    version: int,
    response: httpx.Response,
) -> tuple[Path, Path, str, str, str]:
    """
    Save raw ZIP and extracted SPL XML under a deterministic folder structure.

    data/raw/spl/{set_id}/v{version}/label.zip
    data/raw/spl/{set_id}/v{version}/label.xml
    """
    version_dir = settings.raw_spl_dir / set_id / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    zip_bytes = response.content
    zip_path = version_dir / "label.zip"
    zip_path.write_bytes(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_member_name = choose_main_xml_member(zf)
        xml_bytes = zf.read(xml_member_name)

    xml_path = version_dir / "label.xml"
    xml_path.write_bytes(xml_bytes)

    metadata = {
        "concept_name": concept_name,
        "set_id": set_id,
        "spl_version": version,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_url": str(response.request.url),
        "response_headers": dict(response.headers),
        "zip_sha256": sha256_bytes(zip_bytes),
        "xml_sha256": sha256_bytes(xml_bytes),
        "xml_member_name": xml_member_name,
    }

    metadata_path = version_dir / "download_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return (
        zip_path,
        xml_path,
        metadata["zip_sha256"],
        metadata["xml_sha256"],
        xml_member_name,
    )


def write_manifest_csv(rows: list[DownloadManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "concept_name",
        "set_id",
        "locked_spl_version",
        "locked_published_date",
        "locked_title",
        "zip_path",
        "xml_path",
        "zip_sha256",
        "xml_sha256",
        "xml_member_name",
        "dailymed_label_last_updated",
        "downloaded_at_utc",
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

    manifest_rows: list[DownloadManifestRow] = []

    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        for seed in seeds:
            assert seed.set_id is not None
            assert seed.locked_spl_version is not None

            response = download_versioned_zip(
                client=client,
                set_id=seed.set_id,
                version=seed.locked_spl_version,
            )

            (
                zip_path,
                xml_path,
                zip_sha256,
                xml_sha256,
                xml_member_name,
            ) = save_label_artifacts(
                concept_name=seed.concept_name,
                set_id=seed.set_id,
                version=seed.locked_spl_version,
                response=response,
            )

            manifest_rows.append(
                DownloadManifestRow(
                    concept_name=seed.concept_name,
                    set_id=seed.set_id,
                    locked_spl_version=seed.locked_spl_version,
                    locked_published_date=seed.locked_published_date,
                    locked_title=seed.locked_title,
                    zip_path=str(zip_path),
                    xml_path=str(xml_path),
                    zip_sha256=zip_sha256,
                    xml_sha256=xml_sha256,
                    xml_member_name=xml_member_name,
                    dailymed_label_last_updated=response.headers.get(
                        "X-DAILYMED-LABEL-LAST-UPDATED"
                    ),
                    downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

            LOGGER.info(
                "Downloaded %s v%s -> %s",
                seed.concept_name,
                seed.locked_spl_version,
                xml_path,
            )

    write_manifest_csv(manifest_rows, settings.download_manifest_csv_path)

    LOGGER.info(
        "Wrote %s download-manifest rows to %s",
        len(manifest_rows),
        settings.download_manifest_csv_path,
    )


if __name__ == "__main__":
    main()