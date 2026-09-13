from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from sqlalchemy.engine import Connection, Engine
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from src.config.settings import SQLITE_DB_PATH, SQL_QUERY_TIMEOUT_SECONDS


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    path = Path(db_path) if db_path is not None else SQLITE_DB_PATH
    if not path.is_absolute():
        from src.config.settings import PROJECT_ROOT

        path = PROJECT_ROOT / path
    return path


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_readonly_engine(db_path: Path | str | None = None) -> Engine:
    """SQLAlchemy engine that opens SQLite in read-only / query-only mode."""
    path = resolve_db_path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    uri = f"file:{path.resolve().as_posix()}?mode=ro"

    def _creator() -> sqlite3.Connection:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    return create_engine("sqlite+pysqlite://", creator=_creator, poolclass=NullPool)


def apply_query_timeout(connection: Connection, timeout_seconds: float | None) -> None:
    """Abort long-running SQLite work via a progress handler.

    ``timeout_seconds`` ``None`` uses the configured default. Values ``<= 0``
    disable the handler.
    """
    seconds = SQL_QUERY_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    if seconds <= 0:
        return
    dbapi = _dbapi_connection(connection)
    started = time.monotonic()

    def _on_progress() -> int:
        return 1 if (time.monotonic() - started) >= seconds else 0

    dbapi.set_progress_handler(_on_progress, 1000)


def _dbapi_connection(connection: Connection) -> sqlite3.Connection:
    pooled = connection.connection
    dbapi = getattr(pooled, "dbapi_connection", None) or getattr(
        pooled, "driver_connection", None
    )
    if dbapi is None:
        dbapi = pooled
    return dbapi
