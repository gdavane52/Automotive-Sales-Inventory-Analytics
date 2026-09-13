"""Database tools for the LangGraph agent.

These functions are the contract the graph will bind as tools. They
delegate to the database service so SQL and connection details stay in
``src.db``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.db.pipeline import question_to_dataframe as run_question
from src.db.pipeline import question_to_result as run_question_result
from src.db.pipeline import validate_and_execute as run_validated
from src.db.service import execute_sql as run_sql
from src.db.service import get_database_schema as inspect_schema
from src.db.service import validate_sql as inspect_query


def get_database_schema(
    db_path: Path | str | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Inspect the live SQLite database for the agent.

    Returns a JSON-serializable dict with:

    - ``table_names``
    - ``tables`` (columns, data types, nullability, per-table keys)
    - ``primary_keys``
    - ``foreign_keys``
    - ``relationships`` (including one-to-one vs many-to-one)
    """
    return inspect_schema(db_path=db_path, conn=conn)


def execute_sql(
    query: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Run a read-only SELECT against the analytics database.

    Accepts a single ``SELECT`` (including ``WITH`` CTEs). INSERT, UPDATE,
    DELETE, DROP, ALTER, TRUNCATE, and other destructive statements are
    rejected before execution. Results are returned as a pandas DataFrame.
    """
    return run_sql(query, db_path=db_path, timeout_seconds=timeout_seconds)


def validate_sql(
    query: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate a SQL query without returning result rows.

    Checks that the statement is a single read-only SELECT, that it is
    syntactically reasonable, and that referenced tables/columns exist
    where they can be resolved. Uses the same query timeout as execution.
    """
    return inspect_query(
        query, db_path=db_path, timeout_seconds=timeout_seconds
    )


def validate_and_execute(
    query: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Validate generated SQL, then execute it. Returns a pandas DataFrame."""
    return run_validated(
        query, db_path=db_path, timeout_seconds=timeout_seconds
    )


def question_to_dataframe(
    user_question: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Natural language → SQL → validate → execute → DataFrame."""
    return run_question(
        user_question, db_path=db_path, timeout_seconds=timeout_seconds
    )


def question_to_result(
    user_question: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Natural language pipeline including sql, explanation, and DataFrame."""
    return run_question_result(
        user_question, db_path=db_path, timeout_seconds=timeout_seconds
    )
