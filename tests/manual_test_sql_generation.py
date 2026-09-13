"""Manual check: natural language → SQL (does not execute the query).

Run from the automotive-analytics directory:

    py -3 tests/manual_test_sql_generation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sql_agent import generate_sql
from src.db.connection import connect
from src.db.service import get_database_schema

USER_QUESTION = "Which are the top 10 selling car models in Pune in 2026?"


def main() -> None:
    conn = connect()
    try:
        database_schema = get_database_schema(conn=conn)
    finally:
        conn.close()

    result = generate_sql(USER_QUESTION, database_schema)

    print("User Question")
    print(USER_QUESTION)
    print()
    print("Generated SQL")
    print(result.get("sql") or "(none)")
    print()
    print("Explanation")
    print(result.get("explanation") or "")
    print()
    print("Tables Used")
    tables = result.get("tables_used") or []
    print(", ".join(tables) if tables else "(none)")


if __name__ == "__main__":
    main()
