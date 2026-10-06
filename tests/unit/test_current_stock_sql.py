"""Current-stock questions must filter stock_status = 'In Stock'."""

from __future__ import annotations

import unittest

from src.db.service import validate_sql
from src.db.sql_guard import asks_current_stock, current_stock_filter_error
from src.prompts.sql_prompt import SQL_SYSTEM_PROMPT
from tests.unit.test_execute_sql import _temp_db


class CurrentStockPromptTests(unittest.TestCase):
    def test_prompt_uses_stock_status_column(self) -> None:
        self.assertIn("stock_status = 'In Stock'", SQL_SYSTEM_PROMPT)
        self.assertIn("stock_status = 'Reserved'", SQL_SYSTEM_PROMPT)
        self.assertIn("stock_status = 'Sold'", SQL_SYSTEM_PROMPT)
        self.assertIn("There is no column named status.", SQL_SYSTEM_PROMPT)
        self.assertIn("model = 'Baleno' AND location = 'Pune' AND stock_status = 'In Stock'", SQL_SYSTEM_PROMPT)


class CurrentStockRuleTests(unittest.TestCase):
    def test_baleno_in_pune_requires_in_stock(self) -> None:
        question = "What is the current stock of Baleno in Pune?"
        missing = (
            "SELECT COUNT(*) AS current_stock FROM vehicle_stock "
            "WHERE model = 'Baleno' AND location = 'Pune'"
        )
        correct = missing + " AND stock_status = 'In Stock'"
        self.assertTrue(asks_current_stock(question))
        self.assertIsNotNone(current_stock_filter_error(question, missing))
        self.assertIsNone(current_stock_filter_error(question, correct))
        self.assertIsNone(
            current_stock_filter_error(
                question,
                "SELECT COUNT(*) AS current_stock FROM vehicle_stock "
                "WHERE model = 'Baleno' AND location = 'Pune' "
                "AND vehicle_stock.stock_status='In Stock'",
            )
        )

    def test_baleno_current_stock_without_location(self) -> None:
        question = "What is the current stock of Baleno?"
        correct = (
            "SELECT COUNT(*) AS current_stock FROM vehicle_stock "
            "WHERE model = 'Baleno' AND stock_status = 'In Stock'"
        )
        self.assertIsNone(current_stock_filter_error(question, correct))
        self.assertIsNotNone(
            current_stock_filter_error(
                question,
                "SELECT COUNT(*) AS current_stock FROM vehicle_stock WHERE model = 'Baleno'",
            )
        )

    def test_reserved_question_does_not_force_in_stock(self) -> None:
        question = "How many Baleno cars are reserved in Pune?"
        sql = (
            "SELECT COUNT(*) AS reserved_count FROM vehicle_stock "
            "WHERE model = 'Baleno' AND location = 'Pune' AND stock_status = 'Reserved'"
        )
        self.assertFalse(asks_current_stock(question))
        self.assertIsNone(current_stock_filter_error(question, sql))

    def test_sold_question_does_not_force_in_stock(self) -> None:
        question = "How many Baleno cars were sold in Pune?"
        sql = (
            "SELECT COUNT(*) AS sold_count FROM vehicle_stock "
            "WHERE model = 'Baleno' AND location = 'Pune' AND stock_status = 'Sold'"
        )
        self.assertFalse(asks_current_stock(question))
        self.assertIsNone(current_stock_filter_error(question, sql))

    def test_current_stock_by_model_requires_in_stock(self) -> None:
        question = "Show current stock by model in Pune."
        correct = (
            "SELECT model, COUNT(*) AS current_stock FROM vehicle_stock "
            "WHERE location = 'Pune' AND stock_status = 'In Stock' GROUP BY model"
        )
        missing = (
            "SELECT model, COUNT(*) AS current_stock FROM vehicle_stock "
            "WHERE location = 'Pune' GROUP BY model"
        )
        self.assertIsNone(current_stock_filter_error(question, correct))
        self.assertIsNotNone(current_stock_filter_error(question, missing))

    def test_total_table_count_is_not_current_stock(self) -> None:
        question = "How many Baleno vehicles are present in the vehicle_stock table?"
        sql = (
            "SELECT COUNT(*) AS vehicle_count FROM vehicle_stock WHERE model = 'Baleno'"
        )
        self.assertFalse(asks_current_stock(question))
        self.assertIsNone(current_stock_filter_error(question, sql))

    def test_sales_question_is_unchanged(self) -> None:
        question = "How many cars were sold in 2025?"
        sql = (
            "SELECT COUNT(*) AS sold_count FROM sales "
            "WHERE sale_date >= '2025-01-01' AND sale_date < '2026-01-01'"
        )
        self.assertFalse(asks_current_stock(question))
        self.assertIsNone(current_stock_filter_error(question, sql))


class CurrentStockValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = _temp_db()

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_validator_rejects_current_stock_without_filter(self) -> None:
        question = "What is the current stock of Baleno in Pune?"
        sql = (
            "SELECT COUNT(*) AS current_stock FROM vehicle_stock "
            "WHERE model = 'Baleno' AND location = 'Pune'"
        )
        report = validate_sql(sql, db_path=self.db_path, user_question=question)
        self.assertFalse(report["valid"])
        self.assertTrue(any("stock_status = 'In Stock'" in err for err in report["errors"]))

    def test_validator_accepts_current_stock_filter(self) -> None:
        question = "What is the current stock of Baleno in Pune?"
        sql = (
            "SELECT COUNT(*) AS current_stock FROM vehicle_stock "
            "WHERE model = 'Baleno' AND location = 'Pune' AND stock_status = 'In Stock'"
        )
        report = validate_sql(sql, db_path=self.db_path, user_question=question)
        self.assertTrue(report["valid"], report["errors"])

    def test_same_sql_stays_valid_without_a_current_stock_question(self) -> None:
        sql = (
            "SELECT COUNT(*) AS current_stock FROM vehicle_stock "
            "WHERE model = 'Baleno' AND location = 'Pune'"
        )
        report = validate_sql(sql, db_path=self.db_path)
        self.assertTrue(report["valid"], report["errors"])
