from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.errors import ReadOnlyQueryError, SQLExecutionError
from src.db.schema import DDL
from src.db.sql_guard import assert_readonly_select
from src.db.service import execute_sql
from src.tools.database import execute_sql as tool_execute_sql


def _temp_db() -> Path:
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
    conn.commit()
    conn.close()
    return path


class SqlGuardTests(unittest.TestCase):
    def test_allows_select_and_with(self) -> None:
        self.assertTrue(assert_readonly_select("SELECT * FROM sales").startswith("SELECT"))
        cte = """
        WITH recent AS (SELECT sale_id FROM sales)
        SELECT COUNT(*) AS n FROM recent
        """
        self.assertTrue(assert_readonly_select(cte).upper().startswith("WITH"))

    def test_allows_select_that_mentions_replace_function(self) -> None:
        sql = "SELECT replace(model, ' ', '-') AS slug FROM vehicle_stock"
        self.assertEqual(assert_readonly_select(sql), sql)

    def test_rejects_empty_and_multiple_statements(self) -> None:
        with self.assertRaises(ReadOnlyQueryError):
            assert_readonly_select("   ")
        with self.assertRaises(ReadOnlyQueryError):
            assert_readonly_select("SELECT 1; SELECT 2")

    def test_rejects_destructive_statements(self) -> None:
        blocked = [
            "INSERT INTO sales (sale_id) VALUES ('x')",
            "UPDATE vehicle_stock SET stock_status = 'Sold'",
            "DELETE FROM sales",
            "DROP TABLE sales",
            "ALTER TABLE sales ADD COLUMN x TEXT",
            "TRUNCATE TABLE sales",
            "CREATE TABLE x (id INTEGER)",
            "SELECT 1; DROP TABLE sales",
            "SELECT * FROM vehicle_stock; DELETE FROM vehicle_stock",
        ]
        for query in blocked:
            with self.subTest(query=query):
                with self.assertRaises(ReadOnlyQueryError):
                    assert_readonly_select(query)

    def test_rejects_comment_hidden_writes(self) -> None:
        with self.assertRaises(ReadOnlyQueryError):
            assert_readonly_select("SELECT 1; /* comment */ DELETE FROM sales")


class ExecuteSqlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = _temp_db()

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_normal_result(self) -> None:
        frame = execute_sql(
            "SELECT brand, model, listed_price FROM vehicle_stock WHERE brand = 'Honda'",
            db_path=self.db_path,
        )
        self.assertGreater(len(frame), 0)
        self.assertEqual(frame.iloc[0]["model"], "City")

    def test_empty_result(self) -> None:
        frame = execute_sql(
            "SELECT brand, model FROM vehicle_stock WHERE 1 = 0",
            db_path=self.db_path,
        )
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(len(frame), 0)
        self.assertEqual(list(frame.columns), ["brand", "model"])

    def test_tool_wrapper_matches_service(self) -> None:
        query = "SELECT COUNT(*) AS n FROM vehicle_stock"
        left = execute_sql(query, db_path=self.db_path)
        right = tool_execute_sql(query, db_path=self.db_path)
        pd.testing.assert_frame_equal(left, right)

    def test_rejects_write_before_execution(self) -> None:
        with self.assertRaises(ReadOnlyQueryError):
            execute_sql("DELETE FROM vehicle_stock", db_path=self.db_path)
        remaining = execute_sql(
            "SELECT COUNT(*) AS n FROM vehicle_stock",
            db_path=self.db_path,
        )
        self.assertEqual(int(remaining.iloc[0]["n"]), 1)

    def test_invalid_sql_raises_execution_error(self) -> None:
        with self.assertRaises(SQLExecutionError) as ctx:
            execute_sql("SELECT nope FROM vehicle_stock", db_path=self.db_path)
        self.assertIn("SQL execution failed", str(ctx.exception))

    def test_missing_database_raises(self) -> None:
        missing = self.db_path.parent / "execute_sql_missing.db"
        if missing.exists():
            missing.unlink()
        with self.assertRaises(FileNotFoundError):
            execute_sql("SELECT 1 AS n", db_path=missing)


if __name__ == "__main__":
    unittest.main()
