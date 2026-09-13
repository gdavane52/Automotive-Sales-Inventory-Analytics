from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.service import execute_sql, validate_sql
from src.tools.database import validate_sql as tool_validate_sql
from tests.unit.test_execute_sql import _temp_db


class ValidateSqlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = _temp_db()

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_valid_select(self) -> None:
        report = validate_sql(
            "SELECT brand, model FROM vehicle_stock WHERE brand = 'Honda'",
            db_path=self.db_path,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(report["read_only"])
        self.assertTrue(report["syntax_ok"])
        self.assertEqual(report["errors"], [])
        self.assertIn("vehicle_stock", [name.lower() for name in report["tables"]])
        self.assertEqual(tool_validate_sql(
            "SELECT brand, model FROM vehicle_stock WHERE brand = 'Honda'",
            db_path=self.db_path,
        )["valid"], True)

    def test_invalid_sql(self) -> None:
        report = validate_sql("SELECT FROM vehicle_stock", db_path=self.db_path)
        self.assertFalse(report["valid"])
        self.assertTrue(report["read_only"])
        self.assertFalse(report["syntax_ok"])
        self.assertTrue(report["errors"])

        missing = validate_sql(
            "SELECT nope FROM vehicle_stock",
            db_path=self.db_path,
        )
        self.assertFalse(missing["valid"])
        self.assertTrue(
            any("nope" in err.lower() or "column" in err.lower() for err in missing["errors"])
        )

        unknown_table = validate_sql(
            "SELECT * FROM not_a_table",
            db_path=self.db_path,
        )
        self.assertFalse(unknown_table["valid"])
        self.assertIn("not_a_table", unknown_table["unknown_tables"])

    def test_destructive_sql(self) -> None:
        for query in (
            "DELETE FROM vehicle_stock",
            "DROP TABLE vehicle_stock",
            "UPDATE vehicle_stock SET stock_status = 'Sold'",
            "INSERT INTO vehicle_stock (vehicle_id) VALUES ('x')",
        ):
            with self.subTest(query=query):
                report = validate_sql(query, db_path=self.db_path)
                self.assertFalse(report["valid"])
                self.assertFalse(report["read_only"])
                self.assertTrue(report["errors"])

    def test_empty_result(self) -> None:
        frame = execute_sql(
            "SELECT brand FROM vehicle_stock WHERE brand = 'NoSuchMake'",
            db_path=self.db_path,
        )
        self.assertEqual(len(frame), 0)
        self.assertEqual(list(frame.columns), ["brand"])
        report = validate_sql(
            "SELECT brand FROM vehicle_stock WHERE brand = 'NoSuchMake'",
            db_path=self.db_path,
        )
        self.assertTrue(report["valid"], report["errors"])

    def test_normal_result(self) -> None:
        report = validate_sql(
            "SELECT brand, model FROM vehicle_stock",
            db_path=self.db_path,
        )
        self.assertTrue(report["valid"], report["errors"])
        frame = execute_sql("SELECT brand, model FROM vehicle_stock", db_path=self.db_path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["brand"], "Honda")


if __name__ == "__main__":
    unittest.main()
