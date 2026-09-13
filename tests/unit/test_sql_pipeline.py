from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.errors import ReadOnlyQueryError, SQLExecutionError, SQLValidationError
from src.db.pipeline import question_to_dataframe, validate_and_execute
from tests.unit.test_execute_sql import _temp_db


class ValidateAndExecuteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = _temp_db()

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_valid_select_returns_dataframe(self) -> None:
        frame = validate_and_execute(
            "SELECT brand, model FROM vehicle_stock WHERE brand = 'Honda'",
            db_path=self.db_path,
        )
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["model"], "City")

    def test_empty_result(self) -> None:
        frame = validate_and_execute(
            "SELECT brand FROM vehicle_stock WHERE brand = 'NoSuchBrand'",
            db_path=self.db_path,
        )
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(len(frame), 0)
        self.assertEqual(list(frame.columns), ["brand"])

    def test_rejects_destructive_sql(self) -> None:
        for query in (
            "INSERT INTO vehicle_stock (vehicle_id) VALUES ('x')",
            "UPDATE vehicle_stock SET stock_status = 'Sold'",
            "DELETE FROM vehicle_stock",
            "DROP TABLE vehicle_stock",
            "ALTER TABLE vehicle_stock ADD COLUMN x TEXT",
            "TRUNCATE TABLE vehicle_stock",
            "CREATE TABLE x (id INTEGER)",
            "REPLACE INTO vehicle_stock (vehicle_id) VALUES ('x')",
        ):
            with self.subTest(query=query):
                with self.assertRaises(ReadOnlyQueryError):
                    validate_and_execute(query, db_path=self.db_path)

    def test_invalid_sql(self) -> None:
        with self.assertRaises(SQLValidationError):
            validate_and_execute("SELECT FROM vehicle_stock", db_path=self.db_path)

    def test_nonexistent_table(self) -> None:
        with self.assertRaises(SQLValidationError) as ctx:
            validate_and_execute("SELECT * FROM not_a_table", db_path=self.db_path)
        self.assertIn("not_a_table", str(ctx.exception).lower())

    def test_nonexistent_column(self) -> None:
        with self.assertRaises((SQLValidationError, SQLExecutionError)):
            validate_and_execute(
                "SELECT nope FROM vehicle_stock",
                db_path=self.db_path,
            )

    def test_database_error_missing_file(self) -> None:
        missing = self.db_path.parent / "sql_pipeline_missing.db"
        if missing.exists():
            missing.unlink()
        with self.assertRaises((FileNotFoundError, SQLValidationError)):
            validate_and_execute("SELECT 1 AS n", db_path=missing)


class QuestionToDataFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = _temp_db()

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_uses_generated_sql_then_validates_and_executes(self) -> None:
        generated = {
            "sql": "SELECT brand, model FROM vehicle_stock",
            "explanation": "List stock.",
            "tables_used": ["vehicle_stock"],
        }
        with patch(
            "src.agents.sql_agent.generate_sql", return_value=generated
        ) as mocked:
            frame = question_to_dataframe("Show stock", db_path=self.db_path)
        mocked.assert_called_once()
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["brand"], "Honda")

    def test_refuses_when_generator_returns_no_sql(self) -> None:
        generated = {
            "sql": "",
            "explanation": "Only read-only analytics queries are allowed.",
            "tables_used": [],
        }
        with patch("src.agents.sql_agent.generate_sql", return_value=generated):
            with self.assertRaises(SQLValidationError):
                question_to_dataframe("Delete all sales", db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
