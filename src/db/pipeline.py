"""SQL generation → validation → SQLite execution → DataFrame.

Reuses get_database_schema, generate_sql, validate_sql, and execute_sql.
Does not open its own connections or execute SQL directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.db.errors import ReadOnlyQueryError, SQLValidationError
from src.db.service import execute_sql, get_database_schema, validate_sql


def validate_and_execute(
    query: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Validate read-only SQL, then execute it on the existing SQLite database.

    Rejects INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE,
    and other writes. Checks that referenced tables (and qualified columns)
    exist where they can be resolved. Returns a pandas DataFrame, including
    an empty frame when the query is valid but matches no rows.
    """
    report = validate_sql(
        query, db_path=db_path, timeout_seconds=timeout_seconds
    )
    if not report["read_only"]:
        raise ReadOnlyQueryError(
            "; ".join(report["errors"]) or "Only read-only SELECT queries are allowed."
        )
    if not report["valid"]:
        raise SQLValidationError(_format_validation_errors(report))
    return execute_sql(
        report["query"] or query,
        db_path=db_path,
        timeout_seconds=timeout_seconds,
    )


def question_to_dataframe(
    user_question: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Generate SQL from a question, validate it, execute it, return a DataFrame.

    Uses the live schema from get_database_schema and the Step 4 generator.
    Does not execute until validation succeeds.
    """
    result = question_to_result(
        user_question, db_path=db_path, timeout_seconds=timeout_seconds
    )
    return result["dataframe"]


def question_to_result(
    user_question: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Full NL → SQL → validate → execute pipeline with metadata.

    Returns sql, explanation, tables_used, and dataframe. The DataFrame is
    empty when the query is valid but has no matching rows.
    """
    schema = get_database_schema(db_path=db_path)
    from src.agents.sql_agent import generate_sql

    generated = generate_sql(user_question, schema)
    sql = (generated.get("sql") or "").strip()
    if not sql:
        raise SQLValidationError(
            generated.get("explanation")
            or "SQL generation produced no read-only SELECT."
        )
    frame = validate_and_execute(
        sql, db_path=db_path, timeout_seconds=timeout_seconds
    )
    return {
        "sql": sql,
        "explanation": generated.get("explanation") or "",
        "tables_used": generated.get("tables_used") or [],
        "dataframe": frame,
    }


def _format_validation_errors(report: dict[str, Any]) -> str:
    parts = list(report.get("errors") or [])
    unknown_tables = report.get("unknown_tables") or []
    unknown_columns = report.get("unknown_columns") or []
    if unknown_tables and not any("Unknown table" in p for p in parts):
        parts.append("Unknown table(s): " + ", ".join(unknown_tables))
    if unknown_columns and not any("Unknown column" in p for p in parts):
        parts.append("Unknown column(s): " + ", ".join(unknown_columns))
    return "; ".join(parts) or "SQL failed validation."
