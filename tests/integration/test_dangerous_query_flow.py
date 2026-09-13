"""Destructive SQL and wipe requests must never change SQLite data."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.sql_agent import AutomotiveScope, SQLGeneration, _finalize_generation
from src.db.errors import ReadOnlyQueryError
from src.db.schema import DDL
from src.db.service import execute_sql, get_database_schema, validate_sql
from src.db.sql_guard import assert_readonly_select
from src.langgraph.graph import run_analytics_question
from src.langgraph import nodes as graph_nodes

DESTRUCTIVE_SQL = [
    "DELETE FROM sales;",
    "DROP TABLE sales;",
    "UPDATE sales SET selling_price = 0;",
]

DELETE_REQUEST = "Delete all sales data."


def _temp_sales_db() -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    path = Path(handle.name)
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    conn.execute(
        """
        INSERT INTO vehicle_stock (
            vehicle_id, brand, model, variant, fuel_type, location, dealer_id,
            stock_date, stock_status, listed_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "VEH-00001",
            "Honda",
            "City",
            "ZX",
            "Petrol",
            "Pune",
            "DLR-PUN-01",
            "2026-01-15",
            "In Stock",
            1450000,
        ),
    )
    conn.execute(
        """
        INSERT INTO sales (
            sale_id, customer_id, vehicle_id, sale_date, brand, model,
            location, dealer_id, selling_price, trade_in_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "SALE-00001",
            "CUST-00001",
            "VEH-00001",
            "2026-03-01",
            "Honda",
            "City",
            "Pune",
            "DLR-PUN-01",
            1400000,
            "false",
        ),
    )
    conn.commit()
    conn.close()
    return path


def _sales_snapshot(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "sales" in tables
        count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        price = conn.execute(
            "SELECT selling_price FROM sales WHERE sale_id = 'SALE-00001'"
        ).fetchone()[0]
        return count, price
    finally:
        conn.close()


@pytest.fixture()
def temp_db() -> Path:
    path = _temp_sales_db()
    yield path
    path.unlink(missing_ok=True)


@pytest.mark.parametrize("sql", DESTRUCTIVE_SQL)
def test_destructive_sql_is_rejected_by_guard_and_validator(sql: str, temp_db: Path) -> None:
    before_count, before_price = _sales_snapshot(temp_db)

    with pytest.raises(ReadOnlyQueryError):
        assert_readonly_select(sql)

    report = validate_sql(sql, db_path=temp_db)
    assert report["valid"] is False
    assert report["read_only"] is False
    assert report["errors"]

    with pytest.raises(ReadOnlyQueryError):
        execute_sql(sql, db_path=temp_db)

    after_count, after_price = _sales_snapshot(temp_db)
    assert after_count == before_count == 1
    assert after_price == before_price == 1400000


def test_sql_generator_finalize_strips_destructive_sql(temp_db: Path) -> None:
    schema = get_database_schema(db_path=temp_db)
    for sql in DESTRUCTIVE_SQL:
        result = _finalize_generation(
            SQLGeneration(
                sql=sql,
                explanation="Wipe sales.",
                tables_used=["sales"],
            ),
            schema,
        )
        assert result.sql == ""
        assert "read-only" in result.explanation.lower()


def test_delete_all_sales_request_does_not_execute_or_modify_sqlite(temp_db: Path) -> None:
    """Even if generated SQL is destructive, the graph must not run it."""
    before_count, before_price = _sales_snapshot(temp_db)

    def fake_sql(*_args, **_kwargs):
        return {
            "sql": "DELETE FROM sales",
            "explanation": "Delete all sales.",
            "tables_used": ["sales"],
        }

    schema = get_database_schema(db_path=temp_db)
    with (
        patch.object(
            graph_nodes,
            "classify_automotive_scope",
            return_value=AutomotiveScope(
                in_scope=True,
                reason="Request mentions sales data.",
            ),
        ),
        patch.object(graph_nodes, "get_database_schema", return_value=schema),
        patch.object(graph_nodes, "generate_sql_from_agent", side_effect=fake_sql),
        patch.object(
            graph_nodes,
            "validate_sql_query",
            wraps=graph_nodes.validate_sql_query,
        ) as validate_spy,
        patch.object(graph_nodes, "execute_sql_query") as execute_spy,
    ):
        result = run_analytics_question(DELETE_REQUEST)

    execute_spy.assert_not_called()
    assert validate_spy.called
    assert result.get("validation_result") is False
    assert result.get("query_result") is None
    assert (result.get("final_answer") or "").strip()

    after_count, after_price = _sales_snapshot(temp_db)
    assert after_count == before_count == 1
    assert after_price == before_price == 1400000
    conn = sqlite3.connect(temp_db)
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "sales" in names
    finally:
        conn.close()
