from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sql_agent import SQLGeneration, _finalize_generation
from src.db.schema import DDL
from src.db.service import get_database_schema
from src.prompts.sql_prompt import format_schema_for_prompt


def _schema() -> dict:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    try:
        return get_database_schema(conn=conn)
    finally:
        conn.close()


class SqlPromptTests(unittest.TestCase):
    def test_schema_lists_real_tables_and_columns(self) -> None:
        text = format_schema_for_prompt(_schema())
        self.assertIn("TABLE vehicle_stock", text)
        self.assertIn("vehicle_id", text)
        self.assertIn("sales.vehicle_id -> vehicle_stock.vehicle_id", text)
        self.assertNotIn("trips", text)


class SqlAgentFinalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = _schema()

    def test_accepts_readonly_select(self) -> None:
        result = _finalize_generation(
            SQLGeneration(
                sql="SELECT brand, COUNT(*) AS n FROM vehicle_stock GROUP BY brand ORDER BY n DESC LIMIT 10",
                explanation="Count vehicles by brand.",
                tables_used=["vehicle_stock"],
            ),
            self.schema,
        )
        self.assertIn("SELECT", result.sql.upper())
        self.assertEqual(result.tables_used, ["vehicle_stock"])

    def test_rejects_destructive_sql(self) -> None:
        result = _finalize_generation(
            SQLGeneration(
                sql="DELETE FROM vehicle_stock",
                explanation="Removing rows.",
                tables_used=["vehicle_stock"],
            ),
            self.schema,
        )
        self.assertEqual(result.sql, "")
        self.assertIn("read-only", result.explanation.lower())

    def test_rejects_unknown_table(self) -> None:
        result = _finalize_generation(
            SQLGeneration(
                sql="SELECT * FROM trips",
                explanation="Trip count.",
                tables_used=["trips"],
            ),
            self.schema,
        )
        self.assertEqual(result.sql, "")
        self.assertIn("not in the provided schema", result.explanation)

    def test_rejects_postgres_syntax(self) -> None:
        result = _finalize_generation(
            SQLGeneration(
                sql="SELECT EXTRACT(YEAR FROM sale_date) FROM sales",
                explanation="Year.",
                tables_used=["sales"],
            ),
            self.schema,
        )
        self.assertEqual(result.sql, "")
        self.assertIn("PostgreSQL", result.explanation)


class AutomotiveScopeTests(unittest.TestCase):
    def test_validate_question_refuses_when_out_of_scope(self) -> None:
        from unittest.mock import patch

        from src.agents.sql_agent import AutomotiveScope
        from src.langgraph.nodes import validate_question

        with patch(
            "src.langgraph.nodes.classify_automotive_scope",
            return_value=AutomotiveScope(
                in_scope=False,
                reason=(
                    "This app only answers automotive industry questions "
                    "(vehicle stock, checkout, sales, and trade-ins)."
                ),
            ),
        ) as classify:
            result = validate_question(
                {
                    "user_question": (
                        "Which are the top 10 selling mobile models in Pune in 2026?"
                    )
                }
            )

        classify.assert_called_once()
        self.assertFalse(result["in_scope"])
        self.assertIn("automotive", result["final_answer"].lower())

    def test_validate_question_accepts_automotive_question(self) -> None:
        from unittest.mock import patch

        from src.agents.sql_agent import AutomotiveScope
        from src.langgraph.nodes import validate_question

        with patch(
            "src.langgraph.nodes.classify_automotive_scope",
            return_value=AutomotiveScope(in_scope=True, reason=""),
        ):
            result = validate_question(
                {
                    "user_question": (
                        "Which are the top 10 selling car models in Pune in 2026?"
                    )
                }
            )

        self.assertTrue(result["in_scope"])
        self.assertNotIn("final_answer", result)


if __name__ == "__main__":
    unittest.main()
