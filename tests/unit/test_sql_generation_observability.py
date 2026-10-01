"""SQL generation observability: per-attempt timing on the Phase 1 request."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sql_agent import SQLGeneration, generate_sql
from src.db.schema import DDL
from src.db.service import get_database_schema
from src.langgraph.nodes import generate_sql as generate_sql_node
from src.langgraph.routing import route_after_validation
from src.observability.context import begin_request, current_request, reset_request
from src.observability.logger import complete_request

_VALID_SQL = (
    "SELECT brand, COUNT(*) AS n FROM vehicle_stock "
    "GROUP BY brand ORDER BY n DESC LIMIT 10"
)


def _schema() -> dict:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    try:
        return get_database_schema(conn=conn)
    finally:
        conn.close()


class _ScriptedLLM:
    def __init__(self, results: list, delay_seconds: float = 0.0) -> None:
        self._results = list(results)
        self._delay_seconds = delay_seconds

    def with_structured_output(self, _schema):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self._invoke)

    def _invoke(self, _payload):
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        item = self._results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class SqlGenerationObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = _schema()
        self._sql_flag = os.environ.get("OBSERVABILITY_LOG_SQL")
        os.environ.pop("OBSERVABILITY_LOG_SQL", None)
        self.ctx = begin_request(thread_id="T1", user_question="top brands")

    def tearDown(self) -> None:
        if current_request() is self.ctx:
            reset_request(self.ctx)
        if self._sql_flag is None:
            os.environ.pop("OBSERVABILITY_LOG_SQL", None)
        else:
            os.environ["OBSERVABILITY_LOG_SQL"] = self._sql_flag

    def test_successful_sql_generation_records_success(self) -> None:
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM(
                [SQLGeneration(sql=_VALID_SQL, explanation="Count brands.", tables_used=["vehicle_stock"])]
            ),
        ):
            payload = generate_sql("top brands", self.schema)

        attempt = self.ctx.sql_generation_attempts[0]
        self.assertEqual(attempt.status, "SUCCESS")
        self.assertIsNone(attempt.error_type)
        self.assertIn("vehicle_stock", payload["sql"])
        self.assertEqual(set(payload), {"sql", "explanation", "tables_used", "in_scope"})

    def test_sql_generation_latency_is_recorded(self) -> None:
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM(
                [SQLGeneration(sql=_VALID_SQL, explanation="Count brands.", tables_used=["vehicle_stock"])],
                delay_seconds=0.08,
            ),
        ):
            generate_sql("top brands", self.schema)

        attempt = self.ctx.sql_generation_attempts[0]
        self.assertGreaterEqual(attempt.latency_ms, 60)
        self.assertIsNotNone(attempt.start_time.tzinfo)
        self.assertIsNotNone(attempt.end_time.tzinfo)
        self.assertGreaterEqual(attempt.end_time, attempt.start_time)

    def test_sql_generation_failure_records_failed(self) -> None:
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM([RuntimeError("model unavailable")]),
        ):
            with self.assertRaises(RuntimeError):
                generate_sql("top brands", self.schema)

        attempt = self.ctx.sql_generation_attempts[0]
        self.assertEqual(attempt.status, "FAILED")
        self.assertEqual(attempt.error_type, "RuntimeError")
        self.assertTrue(attempt.error_message)
        self.assertNotIn("model unavailable", attempt.error_message or "")

    def test_request_id_is_associated_with_sql_generation(self) -> None:
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM(
                [SQLGeneration(sql=_VALID_SQL, explanation="Count brands.", tables_used=["vehicle_stock"])]
            ),
        ):
            with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
                generate_sql("top brands", self.schema)

        attempt = self.ctx.sql_generation_attempts[0]
        self.assertEqual(attempt.request_id, self.ctx.request_id)
        payload = _event("\n".join(logs.output), "SQL_GENERATION_COMPLETED")
        self.assertEqual(payload["request_id"], self.ctx.request_id)

    def test_thread_id_is_associated_with_sql_generation(self) -> None:
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM(
                [SQLGeneration(sql=_VALID_SQL, explanation="Count brands.", tables_used=["vehicle_stock"])]
            ),
        ):
            with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
                generate_sql("top brands", self.schema)

        attempt = self.ctx.sql_generation_attempts[0]
        self.assertEqual(attempt.thread_id, "T1")
        payload = _event("\n".join(logs.output), "SQL_GENERATION_COMPLETED")
        self.assertEqual(payload["thread_id"], "T1")

    def test_generated_sql_is_captured_when_available(self) -> None:
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM(
                [SQLGeneration(sql=_VALID_SQL, explanation="Count brands.", tables_used=["vehicle_stock"])]
            ),
        ):
            with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
                generate_sql("top brands", self.schema)

        attempt = self.ctx.sql_generation_attempts[0]
        self.assertIn("vehicle_stock", attempt.generated_sql or "")
        self.assertNotIn("vehicle_stock", "\n".join(logs.output))

    def test_multiple_sql_generation_attempts_are_tracked(self) -> None:
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM(
                [
                    SQLGeneration(sql=_VALID_SQL, explanation="First.", tables_used=["vehicle_stock"]),
                    SQLGeneration(
                        sql="SELECT model FROM vehicle_stock LIMIT 5",
                        explanation="Second.",
                        tables_used=["vehicle_stock"],
                    ),
                ]
            ),
        ):
            generate_sql("top brands", self.schema)
            generate_sql("top models", self.schema)

        attempts = self.ctx.sql_generation_attempts
        self.assertEqual([item.attempt for item in attempts], [1, 2])
        self.assertEqual([item.status for item in attempts], ["SUCCESS", "SUCCESS"])
        self.assertEqual({item.request_id for item in attempts}, {self.ctx.request_id})
        self.assertEqual({item.thread_id for item in attempts}, {"T1"})
        self.assertIn("COUNT", attempts[0].generated_sql or "")
        self.assertIn("model", attempts[1].generated_sql or "")
        self.assertNotEqual(attempts[0].latency_ms, None)
        self.assertNotEqual(attempts[1].latency_ms, None)

    def test_existing_retry_behavior_still_works(self) -> None:
        llm = _ScriptedLLM(
            [
                SQLGeneration(sql="SELECT * FROM trips", explanation="Missing table.", tables_used=["trips"]),
                SQLGeneration(sql=_VALID_SQL, explanation="Count brands.", tables_used=["vehicle_stock"]),
            ]
        )
        with (
            patch("src.agents.sql_agent.get_chat_llm", return_value=llm),
            patch("src.langgraph.nodes.get_database_schema", return_value=self.schema),
        ):
            first = generate_sql_node(
                {"user_question": "top brands", "retry_count": 0, "chat_history": []}
            )
            self.assertEqual(first["retry_count"], 1)
            self.assertEqual(first["sql"], "")
            routed = route_after_validation(
                {**first, "validation_result": False, "validation_error": "Unknown table(s): trips"}
            )
            self.assertEqual(routed, "generate_sql")
            second = generate_sql_node(
                {
                    "user_question": "top brands",
                    "retry_count": first["retry_count"],
                    "sql": first["sql"],
                    "validation_error": "Unknown table(s): trips",
                    "chat_history": [],
                }
            )

        self.assertEqual(second["retry_count"], 2)
        self.assertIn("vehicle_stock", second["sql"])
        self.assertNotIn("request_id", second)
        attempts = self.ctx.sql_generation_attempts
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].status, "SUCCESS")
        self.assertEqual(attempts[0].generated_sql, "")
        self.assertEqual(attempts[1].status, "SUCCESS")
        self.assertIn("vehicle_stock", attempts[1].generated_sql or "")

    def test_sql_generation_observability_does_not_replace_request_latency(self) -> None:
        time.sleep(0.06)
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            with patch(
                "src.agents.sql_agent.get_chat_llm",
                return_value=_ScriptedLLM(
                    [SQLGeneration(sql=_VALID_SQL, explanation="Count brands.", tables_used=["vehicle_stock"])],
                    delay_seconds=0.05,
                ),
            ):
                generate_sql("top brands", self.schema)
            time.sleep(0.06)
            complete_request(self.ctx, "SUCCESS")

        attempt = self.ctx.sql_generation_attempts[0]
        self.assertEqual(self.ctx.status, "SUCCESS")
        self.assertGreater(self.ctx.total_latency_ms, attempt.latency_ms)
        self.assertGreaterEqual(attempt.latency_ms, 40)
        blob = "\n".join(logs.output)
        sql_event = _event(blob, "SQL_GENERATION_COMPLETED")
        request_event = _event(blob, "REQUEST_COMPLETED")
        self.assertIn("sql_generation_latency_ms", sql_event)
        self.assertNotIn("sql_generation_latency_ms", request_event)
        self.assertIn("total_latency_ms", request_event)
        self.assertNotEqual(
            request_event["total_latency_ms"],
            sql_event["sql_generation_latency_ms"],
        )

    def test_secrets_are_not_written_to_logs(self) -> None:
        secret = "sk-phase2-sql-observability-secret"
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = secret
        marker = "SQL_SHOULD_NOT_BE_LOGGED_zz"
        sql = f"SELECT brand FROM vehicle_stock WHERE brand = '{marker}'"
        try:
            with patch(
                "src.agents.sql_agent.get_chat_llm",
                return_value=_ScriptedLLM(
                    [
                        RuntimeError(f"failed key {secret}"),
                        SQLGeneration(sql=sql, explanation="Brand filter.", tables_used=["vehicle_stock"]),
                    ]
                ),
            ):
                with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
                    with self.assertRaises(RuntimeError):
                        generate_sql("top brands", self.schema)
                    generate_sql("top brands", self.schema)
            blob = "\n".join(logs.output)
            self.assertNotIn(secret, blob)
            self.assertNotIn(marker, blob)
            self.assertIn("SQL_GENERATION_FAILED", blob)
            self.assertIn(marker, self.ctx.sql_generation_attempts[1].generated_sql or "")
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous


def _event(blob: str, name: str) -> dict:
    for line in blob.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        payload = json.loads(line[start:])
        if payload.get("event") == name:
            return payload
    raise AssertionError(f"{name} not found in logs: {blob}")


if __name__ == "__main__":
    unittest.main()
