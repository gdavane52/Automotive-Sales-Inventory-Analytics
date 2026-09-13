"""Manual check: question → SQL → validate → execute → DataFrame.

Run from the automotive-analytics directory:

    py -3 tests/manual_test_sql_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sql_agent import generate_sql
from src.db.service import execute_sql, get_database_schema, validate_sql

USER_QUESTION = "delete sales record from pune city and year 2026?"


def _print_validation(report: dict) -> None:
    summary = {
        "valid": report.get("valid"),
        "read_only": report.get("read_only"),
        "syntax_ok": report.get("syntax_ok"),
        "tables": report.get("tables"),
        "unknown_tables": report.get("unknown_tables"),
        "unknown_columns": report.get("unknown_columns"),
        "errors": report.get("errors"),
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    print("question")
    print(USER_QUESTION)
    print()

    schema = get_database_schema()
    generated = generate_sql(USER_QUESTION, schema)
    sql = (generated.get("sql") or "").strip()

    print("generated SQL")
    print(sql or "(none)")
    print()

    print("validation result")
    if not sql:
        print("(skipped: SQL generation produced no query)")
        print()
        print("DataFrame")
        print("(not executed)")
        return

    report = validate_sql(sql)
    _print_validation(report)
    print()

    print("DataFrame")
    if not report.get("valid"):
        print("(not executed: validation failed)")
        return

    frame = execute_sql(report.get("query") or sql)
    if frame.empty:
        print(frame)
        print("(empty result)")
    else:
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
