from __future__ import annotations

from qdrant_client import QdrantClient

from medlabeliq.config.settings import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)