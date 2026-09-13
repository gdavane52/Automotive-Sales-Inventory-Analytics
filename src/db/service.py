"""Database service used by LangGraph agent tools.

Introspects the live SQLite database rather than hard-coding DDL, so the
agent always sees the schema that is actually loaded. Query execution is
read-only SELECT via SQLAlchemy.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.config.logging import get_logger
from src.config.settings import SQL_MAX_RESULT_ROWS
from src.db.connection import apply_query_timeout, connect, create_readonly_engine, resolve_db_path
from src.db.errors import ReadOnlyQueryError, SQLExecutionError, SQLTimeoutError
from src.db.sql_guard import (
    assert_readonly_select,
    referenced_qualified_columns,
    referenced_tables,
)

_INTERNAL_TABLE_PREFIX = "sqlite_"
logger = get_logger(__name__)


def get_database_schema(
    db_path: Path | str | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Return table names, columns, types, keys, and relationships.

    Parameters
    ----------
    db_path:
        SQLite file. Ignored when ``conn`` is provided. Defaults to the
        configured analytics database.
    conn:
        Optional open connection (used by tests and callers that already
        hold a session). The service does not close a passed-in connection.
    """
    owns_conn = conn is None
    if owns_conn:
        path = resolve_db_path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        conn = connect(db_path)
    try:
        table_names = _user_tables(conn)
        tables: dict[str, dict[str, Any]] = {}
        relationships: list[dict[str, Any]] = []

        for table in table_names:
            columns, primary_keys = _columns_and_primary_keys(conn, table)
            foreign_keys = _foreign_keys(conn, table)
            unique_sets = _unique_column_sets(conn, table, primary_keys)
            tables[table] = {
                "name": table,
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            }
            relationships.extend(
                _relationships_for_table(table, foreign_keys, unique_sets)
            )

        return {
            "table_names": table_names,
            "tables": tables,
            "primary_keys": {
                name: tables[name]["primary_keys"] for name in table_names
            },
            "foreign_keys": [
                fk
                for name in table_names
                for fk in tables[name]["foreign_keys"]
            ],
            "relationships": relationships,
        }
    finally:
        if owns_conn:
            conn.close()


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE ?
        ORDER BY name
        """,
        (f"{_INTERNAL_TABLE_PREFIX}%",),
    ).fetchall()
    return [row["name"] for row in rows]


def _columns_and_primary_keys(
    conn: sqlite3.Connection,
    table: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    columns: list[dict[str, Any]] = []
    pk_ordered: list[tuple[int, str]] = []
    for row in info:
        name = row["name"]
        pk_index = int(row["pk"] or 0)
        columns.append(
            {
                "name": name,
                "data_type": row["type"] or "NUMERIC",
                "nullable": not bool(row["notnull"]) and pk_index == 0,
                "primary_key": pk_index > 0,
                "default": row["dflt_value"],
            }
        )
        if pk_index > 0:
            pk_ordered.append((pk_index, name))
    primary_keys = [name for _, name in sorted(pk_ordered)]
    return columns, primary_keys


def _foreign_keys(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        fk_id = int(row["id"])
        entry = grouped.setdefault(
            fk_id,
            {
                "constraint_id": fk_id,
                "from_table": table,
                "from_columns": [],
                "to_table": row["table"],
                "to_columns": [],
                "on_update": row["on_update"],
                "on_delete": row["on_delete"],
            },
        )
        entry["from_columns"].append(row["from"])
        entry["to_columns"].append(row["to"])
    return [grouped[key] for key in sorted(grouped)]


def _unique_column_sets(
    conn: sqlite3.Connection,
    table: str,
    primary_keys: list[str],
) -> set[frozenset[str]]:
    unique: set[frozenset[str]] = set()
    if primary_keys:
        unique.add(frozenset(primary_keys))
    indexes = conn.execute(f'PRAGMA index_list("{table}")').fetchall()
    for index in indexes:
        if not index["unique"]:
            continue
        columns = conn.execute(
            f'PRAGMA index_info("{index["name"]}")'
        ).fetchall()
        names = [col["name"] for col in columns if col["name"] is not None]
        if names:
            unique.add(frozenset(names))
    return unique


def _relationships_for_table(
    table: str,
    foreign_keys: list[dict[str, Any]],
    unique_sets: set[frozenset[str]],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    for fk in foreign_keys:
        from_cols = tuple(fk["from_columns"])
        to_cols = tuple(fk["to_columns"])
        relationship_type = (
            "one-to-one" if frozenset(from_cols) in unique_sets else "many-to-one"
        )
        if len(from_cols) == 1:
            from_ref = f"{table}.{from_cols[0]}"
            to_ref = f"{fk['to_table']}.{to_cols[0]}"
        else:
            from_ref = f"{table}.({', '.join(from_cols)})"
            to_ref = f"{fk['to_table']}.({', '.join(to_cols)})"
        relationships.append(
            {
                "from_table": table,
                "from_columns": list(from_cols),
                "to_table": fk["to_table"],
                "to_columns": list(to_cols),
                "relationship_type": relationship_type,
                "description": (
                    f"{from_ref} -> {to_ref} ({relationship_type})"
                ),
            }
        )
    return relationships


def execute_sql(
    query: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Run a single read-only SELECT and return a pandas DataFrame.

    Rejects INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, and other
    write or schema operations. The SQLite connection is opened read-only
    and aborted if the query exceeds ``timeout_seconds``.
    """
    safe_sql = assert_readonly_select(query)
    engine = create_readonly_engine(db_path)
    try:
        with engine.connect() as connection:
            apply_query_timeout(connection, timeout_seconds)
            frame = pd.read_sql_query(text(safe_sql), connection)
            return _limit_result_rows(frame)
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        if _is_timeout_error(exc):
            logger.warning("sql_timeout query=%s", safe_sql[:180])
            raise SQLTimeoutError(
                f"Query exceeded the timeout and was cancelled. Query: {safe_sql}"
            ) from exc
        logger.exception("sqlite_execution_failed")
        raise SQLExecutionError(_friendly_sql_error(exc, safe_sql)) from exc
    except Exception as exc:
        if isinstance(
            exc,
            (SQLExecutionError, FileNotFoundError, ReadOnlyQueryError, SQLTimeoutError),
        ):
            raise
        if _is_timeout_error(exc):
            raise SQLTimeoutError(
                f"Query exceeded the timeout and was cancelled. Query: {safe_sql}"
            ) from exc
        logger.exception("sqlite_execution_failed")
        raise SQLExecutionError("Failed to execute SQL.") from exc
    finally:
        engine.dispose()


def validate_sql(
    query: str,
    db_path: Path | str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Check that a query is a reasonable, read-only SELECT against this schema.

    Returns a JSON-serializable report. Does not raise for bad SQL; the
    agent can inspect ``valid`` and ``errors``. Destructive statements
    set ``read_only`` to False.
    """
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "valid": False,
        "read_only": False,
        "syntax_ok": False,
        "query": None,
        "tables": [],
        "unknown_tables": [],
        "unknown_columns": [],
        "errors": errors,
        "warnings": warnings,
    }

    try:
        safe_sql = assert_readonly_select(query)
    except ReadOnlyQueryError as exc:
        logger.warning("destructive_or_invalid_sql rejected")
        errors.append(str(exc))
        return report

    report["read_only"] = True
    report["query"] = safe_sql
    tables = referenced_tables(safe_sql)
    report["tables"] = tables

    try:
        schema = get_database_schema(db_path=db_path)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        return report

    known_tables = {name.lower(): name for name in schema["table_names"]}
    known_tables.update({"sqlite_master": "sqlite_master"})
    unknown_tables = [
        name for name in tables if name.lower() not in known_tables
    ]
    report["unknown_tables"] = unknown_tables
    if unknown_tables:
        errors.append(
            "Unknown table(s): " + ", ".join(unknown_tables) + ". "
            "Use get_database_schema() for valid table names."
        )

    unknown_columns: list[str] = []
    table_columns: dict[str, set[str]] = {
        name.lower(): {col["name"].lower() for col in spec["columns"]}
        for name, spec in schema["tables"].items()
    }
    for table, column in referenced_qualified_columns(safe_sql):
        real_table = known_tables.get(table.lower())
        if real_table is None:
            continue
        if column.lower() not in table_columns.get(table.lower(), set()):
            unknown_columns.append(f"{table}.{column}")
    report["unknown_columns"] = unknown_columns
    if unknown_columns:
        errors.append(
            "Unknown column(s): " + ", ".join(unknown_columns) + "."
        )

    engine = create_readonly_engine(db_path)
    try:
        with engine.connect() as connection:
            apply_query_timeout(connection, timeout_seconds)
            connection.execute(text(f"EXPLAIN QUERY PLAN {safe_sql}"))
        report["syntax_ok"] = True
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        if _is_timeout_error(exc):
            errors.append("Query timed out during validation.")
        else:
            message = _explain_error_message(exc)
            errors.append(message)
            lowered = message.lower()
            report["syntax_ok"] = "syntax error" not in lowered
            if "no such table" in lowered or "no such column" in lowered:
                report["syntax_ok"] = True
    except Exception as exc:
        if _is_timeout_error(exc):
            errors.append("Query timed out during validation.")
        else:
            errors.append(f"Could not validate SQL syntax: {exc}")
    finally:
        engine.dispose()

    report["valid"] = not errors and report["syntax_ok"]
    return report


def _limit_result_rows(frame: pd.DataFrame) -> pd.DataFrame:
    max_rows = SQL_MAX_RESULT_ROWS
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    logger.warning(
        "query_result_truncated original_rows=%s max_rows=%s",
        len(frame),
        max_rows,
    )
    return frame.iloc[:max_rows].copy()


def _is_timeout_error(exc: BaseException) -> bool:
    parts = [str(exc)]
    orig = getattr(exc, "orig", None)
    if orig is not None:
        parts.append(str(orig))
    blob = " ".join(parts).lower()
    return "interrupted" in blob or "cancelled" in blob


def _explain_error_message(exc: BaseException) -> str:
    orig = getattr(exc, "orig", None)
    original = str(orig) if orig else str(exc)
    return original.strip() or exc.__class__.__name__


def _friendly_sql_error(exc: BaseException, query: str) -> str:
    orig = getattr(exc, "orig", None)
    original = str(orig) if orig else str(exc)
    original = original.strip() or exc.__class__.__name__
    preview = " ".join(query.split())
    if len(preview) > 180:
        preview = preview[:177] + "..."
    return (
        f"SQL execution failed: {original}. "
        "Check table and column names (use get_database_schema), "
        f"syntax, and that the query is a SELECT. Query: {preview}"
    )
