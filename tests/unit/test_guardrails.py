from __future__ import annotations

import logging
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sql_agent import AutomotiveScope
from src.config.logging import redact_secrets
from src.config.safety import (
    LLM_UNAVAILABLE,
    LOOP_STOPPED,
    OUT_OF_SCOPE,
    READ_ONLY_ONLY,
    RETRY_LIMIT,
    public_error_message,
)
from src.db.errors import ReadOnlyQueryError, SQLExecutionError
from src.db.schema import DDL
from src.db.service import execute_sql, validate_sql
from src.db.sql_guard import assert_readonly_select
from src.langgraph.nodes import (
    MAX_SQL_GENERATION_ATTEMPTS,
    generate_answer,
    validate_question,
    validation_failed,
)
from src.langgraph.routing import route_after_validation


def _temp_db() -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    path = Path(handle.name)
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    for index in range(3):
        conn.execute(
            """
            INSERT INTO vehicle_stock (
                vehicle_id, brand, model, variant, fuel_type, location, dealer_id,
                stock_date, stock_status, listed_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"VEH-{index:05d}",
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


class GuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = _temp_db()

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_valid_automotive_select(self) -> None:
        sql = (
            "SELECT brand, model FROM vehicle_stock "
            "WHERE location = 'Pune' ORDER BY brand LIMIT 10"
        )
        report = validate_sql(sql, db_path=self.db_path)
        self.assertTrue(report["valid"], report["errors"])
        frame = execute_sql(sql, db_path=self.db_path)
        self.assertGreater(len(frame), 0)
        self.assertIn("model", list(frame.columns))

    def test_non_automotive_question_rejected(self) -> None:
        with patch(
            "src.langgraph.nodes.classify_automotive_scope",
            return_value=AutomotiveScope(
                in_scope=False,
                reason=OUT_OF_SCOPE,
            ),
        ):
            result = validate_question(
                {
                    "user_question": (
                        "Which are the top 10 selling mobile models in Pune in 2026?"
                    )
                }
            )
        self.assertFalse(result["in_scope"])
        self.assertIn("automotive", result["final_answer"].lower())
        self.assertNotIn("Traceback", result["final_answer"])

    def test_delete_query_rejected(self) -> None:
        with self.assertRaises(ReadOnlyQueryError):
            assert_readonly_select("DELETE FROM vehicle_stock")
        report = validate_sql("DELETE FROM vehicle_stock", db_path=self.db_path)
        self.assertFalse(report["valid"])
        self.assertFalse(report["read_only"])
        with self.assertRaises(ReadOnlyQueryError):
            execute_sql("DELETE FROM vehicle_stock", db_path=self.db_path)

    def test_insert_update_alter_create_truncate_rejected(self) -> None:
        blocked = [
            "INSERT INTO vehicle_stock (vehicle_id) VALUES ('x')",
            "UPDATE vehicle_stock SET stock_status = 'Sold'",
            "DROP TABLE vehicle_stock",
            "ALTER TABLE vehicle_stock ADD COLUMN extra TEXT",
            "CREATE TABLE hacked (id INTEGER)",
            "TRUNCATE TABLE vehicle_stock",
        ]
        for query in blocked:
            with self.subTest(query=query):
                with self.assertRaises(ReadOnlyQueryError):
                    assert_readonly_select(query)
                report = validate_sql(query, db_path=self.db_path)
                self.assertFalse(report["read_only"])
                with self.assertRaises(ReadOnlyQueryError):
                    execute_sql(query, db_path=self.db_path)

    def test_invalid_sql_and_unknown_objects(self) -> None:
        syntax = validate_sql("SELECT FROM vehicle_stock", db_path=self.db_path)
        self.assertFalse(syntax["valid"])
        self.assertTrue(syntax["errors"])

        missing_col = validate_sql(
            "SELECT not_a_column FROM vehicle_stock", db_path=self.db_path
        )
        self.assertFalse(missing_col["valid"])

        missing_table = validate_sql(
            "SELECT * FROM not_a_table", db_path=self.db_path
        )
        self.assertFalse(missing_table["valid"])
        self.assertIn("not_a_table", missing_table["unknown_tables"])

        with self.assertRaises(SQLExecutionError):
            execute_sql("SELECT nope FROM vehicle_stock", db_path=self.db_path)

    def test_empty_result(self) -> None:
        frame = execute_sql(
            "SELECT brand FROM vehicle_stock WHERE brand = 'NoSuchBrand'",
            db_path=self.db_path,
        )
        self.assertEqual(len(frame), 0)
        answer = generate_answer(
            {
                "user_question": "Which Honda models are in stock in Pune?",
                "query_result": frame,
                "sql": "SELECT brand FROM vehicle_stock WHERE brand = 'NoSuchBrand'",
            }
        )
        self.assertIn("no matching data", answer["final_answer"].lower())

    def test_sql_retry_routes_back_to_generate(self) -> None:
        self.assertEqual(MAX_SQL_GENERATION_ATTEMPTS, 3)
        nxt = route_after_validation(
            {
                "validation_result": False,
                "retry_count": 1,
                "validation_error": "Unknown table(s): vehicle",
            }
        )
        self.assertEqual(nxt, "generate_sql")

    def test_maximum_retry_limit(self) -> None:
        nxt = route_after_validation(
            {
                "validation_result": False,
                "retry_count": 3,
                "validation_error": "SQL failed validation.",
            }
        )
        self.assertEqual(nxt, "validation_failed")
        failed = validation_failed(
            {
                "retry_count": 3,
                "validation_error": "Unknown column(s): sales.nope",
                "sql": "SELECT nope FROM sales",
            }
        )
        self.assertFalse(failed["validation_result"])
        self.assertIn("3", failed["final_answer"])
        self.assertTrue(
            RETRY_LIMIT in failed["final_answer"]
            or "attempt" in failed["final_answer"].lower()
            or "rephrase" in failed["final_answer"].lower()
        )

    def test_result_row_limit(self) -> None:
        with patch("src.db.service.SQL_MAX_RESULT_ROWS", 1):
            frame = execute_sql(
                "SELECT vehicle_id FROM vehicle_stock",
                db_path=self.db_path,
            )
        self.assertEqual(len(frame), 1)

    def test_public_errors_hide_stack_traces_and_llm_failures(self) -> None:
        class FakeAPIError(Exception):
            pass

        message = public_error_message(FakeAPIError("openai api key sk-secret failed"))
        self.assertEqual(message, LLM_UNAVAILABLE)
        self.assertNotIn("sk-secret", message)
        self.assertNotIn("Traceback", message)

        recursion = public_error_message(RuntimeError("GraphRecursionError limit"))
        self.assertEqual(recursion, LOOP_STOPPED)

        destructive = public_error_message(ReadOnlyQueryError("DELETE FROM sales"))
        self.assertEqual(destructive, READ_ONLY_ONLY)

        with patch(
            "src.langgraph.nodes.classify_automotive_scope",
            side_effect=RuntimeError("OPENAI_API_KEY missing traceback"),
        ):
            result = validate_question({"user_question": "top selling cars in Pune"})
        self.assertFalse(result["in_scope"])
        self.assertNotIn("Traceback", result["final_answer"])
        self.assertNotIn("OPENAI_API_KEY", result["final_answer"])

    def test_logging_redacts_secrets(self) -> None:
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-test-secret-value"
        try:
            text = redact_secrets("using OPENAI_API_KEY=sk-test-secret-value")
            self.assertNotIn("sk-test-secret-value", text)
            self.assertIn("[REDACTED]", text)
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous

    def test_recursion_limit_is_finite(self) -> None:
        from src.config.settings import LANGGRAPH_RECURSION_LIMIT
        from src.langgraph.graph import _RECURSION_LIMIT

        self.assertEqual(_RECURSION_LIMIT, LANGGRAPH_RECURSION_LIMIT)
        self.assertGreaterEqual(_RECURSION_LIMIT, 8)
        self.assertLessEqual(_RECURSION_LIMIT, 32)


if __name__ == "__main__":
    unittest.main()
