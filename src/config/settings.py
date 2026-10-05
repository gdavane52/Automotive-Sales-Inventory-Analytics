from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SQLITE_DB_PATH = Path(os.environ.get("SQLITE_DB_PATH", PROJECT_ROOT / "data" / "db" / "automotive.db"))
if not SQLITE_DB_PATH.is_absolute():
    SQLITE_DB_PATH = PROJECT_ROOT / SQLITE_DB_PATH

SQL_QUERY_TIMEOUT_SECONDS = float(os.environ.get("SQL_QUERY_TIMEOUT_SECONDS", "30"))
SQL_MAX_RESULT_ROWS = int(os.environ.get("SQL_MAX_RESULT_ROWS", "500"))
LANGGRAPH_RECURSION_LIMIT = int(os.environ.get("LANGGRAPH_RECURSION_LIMIT", "16"))
MAX_SQL_GENERATION_ATTEMPTS = int(os.environ.get("MAX_SQL_GENERATION_ATTEMPTS", "3"))
MAX_QUESTIONS_PER_SESSION = int(os.environ.get("MAX_QUESTIONS_PER_SESSION", "10"))
REQUEST_COOLDOWN_SECONDS = float(os.environ.get("REQUEST_COOLDOWN_SECONDS", "5"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def observability_log_question() -> bool:
    """Include the user question in request logs only when explicitly enabled."""
    raw = (os.environ.get("OBSERVABILITY_LOG_QUESTION") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def observability_log_sql() -> bool:
    """Include generated SQL in SQL-generation logs only when explicitly enabled."""
    raw = (os.environ.get("OBSERVABILITY_LOG_SQL") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
