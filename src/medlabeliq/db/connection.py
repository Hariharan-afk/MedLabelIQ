from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from medlabeliq.config.settings import settings


def get_connection():
    """
    Return a PostgreSQL connection using the project .env settings.
    """
    return psycopg.connect(settings.postgres_dsn, row_factory=dict_row)