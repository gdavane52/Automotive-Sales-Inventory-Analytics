"""SQL execution observability, kept separate from generation and request timing."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sql_agent import SQLGeneration, generate_sql
from src.db.errors import ReadOnlyQueryError, SQLExecutionError
from src.db.service import execute_sql
from src.langgraph.nodes import execute_sql as execute_sql_node
from src.langgraph.nodes import validation_failed
from src.observability.context import begin_request, current_request, reset_request
from src.observability.logger import complete_request

_SELECT = "SELECT 1 AS n"
_EMPTY = "SELECT 1 AS n WHERE 0"


class _ScriptedLLM:
    def __init__(self, results: list) -> None:
        self._results = list(results)

    def with_structured_output(self, _schema):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self._invoke)

    def _invoke(self, _payload):
        item = self._results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class SqlExecutionObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        handle.close()
        self.db_path = Path(handle.name)
        sqlite3.connect(self.db_path).close()
        self._sql_flag = os.environ.get("OBSERVABILITY_LOG_SQL")
        os.environ.pop("OBSERVABILITY_LOG_SQL", None)
        self.ctx = begin_request(thread_id="T1", user_question="count rows")

    def tearDown(self) -> None:
        if current_request() is self.ctx:
            reset_request(self.ctx)
        self.db_path.unlink(missing_ok=True)
        if self._sql_flag is None:
            os.environ.pop("OBSERVABILITY_LOG_SQL", None)
        else:
            os.environ["OBSERVABILITY_LOG_SQL"] = self._sql_flag

    def test_successful_sql_execution(self) -> None:
        frame = execute_sql(_SELECT, db_path=self.db_path)

        attempt = self.ctx.sql_execution_attempts[0]
        self.assertEqual(len(frame), 1)
        self.assertEqual(attempt.status, "SUCCESS")
        self.assertEqual(attempt.row_count, 1)
        self.assertIsNone(attempt.error_type)
        self.assertIsNotNone(attempt.start_time.tzinfo)
        self.assertIsNotNone(attempt.end_time.tzinfo)
        self.assertEqual(self.ctx.sql_generation_attempts, [])

    def test_empty_result_set_is_success(self) -> None:
        frame = execute_sql(_EMPTY, db_path=self.db_path)

        attempt = self.ctx.sql_execution_attempts[0]
        self.assertTrue(frame.empty)
        self.assertEqual(attempt.status, "SUCCESS")
        self.assertEqual(attempt.row_count, 0)
        self.assertIsNotNone(attempt.latency_ms)

    def test_sql_execution_latency_and_row_count(self) -> None:
        def slow_read(*_args, **_kwargs):
            time.sleep(0.08)
            return pd.DataFrame({"model": ["Creta", "Swift", "Nexon"]})

        with patch("src.db.service.pd.read_sql_query", side_effect=slow_read):
            frame = execute_sql(_SELECT, db_path=self.db_path)

        attempt = self.ctx.sql_execution_attempts[0]
        self.assertEqual(len(frame), 3)
        self.assertEqual(attempt.row_count, 3)
        self.assertGreaterEqual(attempt.latency_ms, 60)
        self.assertEqual(attempt.status, "SUCCESS")

    def test_database_execution_failure(self) -> None:
        def explode(*_args, **_kwargs):
            time.sleep(0.04)
            raise SQLAlchemyError("no such column: nope")

        with patch("src.db.service.pd.read_sql_query", side_effect=explode):
            with self.assertRaises(SQLExecutionError):
                execute_sql(_SELECT, db_path=self.db_path)

        attempt = self.ctx.sql_execution_attempts[0]
        self.assertEqual(attempt.status, "FAILED")
        self.assertEqual(attempt.error_type, "SQLExecutionError")
        self.assertTrue(attempt.error_message)
        self.assertIsNone(attempt.row_count)
        self.assertGreaterEqual(attempt.latency_ms, 30)
        self.assertNotIn("no such column", attempt.error_message or "")

    def test_blocked_sql_does_not_execute(self) -> None:
        with patch("src.db.service.pd.read_sql_query") as read:
            with self.assertRaises(ReadOnlyQueryError):
                execute_sql("DELETE FROM vehicle_stock", db_path=self.db_path)
        read.assert_not_called()

        blocked = self.ctx.sql_execution_attempts[0]
        self.assertEqual(blocked.status, "BLOCKED")
        self.assertIsNone(blocked.latency_ms)
        self.assertIsNone(blocked.start_time)
        self.assertIsNone(blocked.row_count)
        self.assertTrue(blocked.reason)
        self.assertIsNone(blocked.error_type)

        validation_failed(
            {
                "retry_count": 3,
                "validation_error": "Unknown table(s): trips",
                "validation_result": False,
            }
        )
        terminal = self.ctx.sql_execution_attempts[1]
        self.assertEqual(terminal.status, "BLOCKED")
        self.assertIsNone(terminal.latency_ms)
        self.assertIn("Unknown table", terminal.reason or "")

    def test_skipped_execution_is_blocked_without_a_query(self) -> None:
        with patch("src.langgraph.nodes.execute_sql_query") as run:
            result = execute_sql_node(
                {
                    "validation_result": False,
                    "validation_error": "Unknown table(s): trips",
                    "sql": "SELECT * FROM trips",
                }
            )
        run.assert_not_called()
        self.assertEqual(result, {})
        attempt = self.ctx.sql_execution_attempts[0]
        self.assertEqual(attempt.status, "BLOCKED")
        self.assertIsNone(attempt.latency_ms)

    def test_request_id_and_thread_id_propagate(self) -> None:
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            execute_sql(_SELECT, db_path=self.db_path)

        attempt = self.ctx.sql_execution_attempts[0]
        self.assertEqual(attempt.request_id, self.ctx.request_id)
        self.assertEqual(attempt.thread_id, "T1")
        payload = _event("\n".join(logs.output), "SQL_EXECUTION_COMPLETED")
        self.assertEqual(payload["request_id"], self.ctx.request_id)
        self.assertEqual(payload["thread_id"], "T1")
        self.assertEqual(payload["sql_execution_status"], "SUCCESS")
        self.assertEqual(payload["row_count"], 1)

    def test_multiple_execution_attempts_are_appended(self) -> None:
        execute_sql(_SELECT, db_path=self.db_path)
        execute_sql(_EMPTY, db_path=self.db_path)

        attempts = self.ctx.sql_execution_attempts
        self.assertEqual([item.attempt for item in attempts], [1, 2])
        self.assertEqual([item.status for item in attempts], ["SUCCESS", "SUCCESS"])
        self.assertEqual([item.row_count for item in attempts], [1, 0])
        self.assertEqual({item.request_id for item in attempts}, {self.ctx.request_id})
        self.assertEqual({item.thread_id for item in attempts}, {"T1"})

    def test_sql_generation_metrics_remain_unchanged(self) -> None:
        schema = {"table_names": [], "tables": {}}
        with patch(
            "src.agents.sql_agent.get_chat_llm",
            return_value=_ScriptedLLM(
                [SQLGeneration(sql=_SELECT, explanation="Constant.", tables_used=[])]
            ),
        ):
            generated = generate_sql("how many", schema)
        execute_sql(generated["sql"], db_path=self.db_path)

        self.assertEqual(len(self.ctx.sql_generation_attempts), 1)
        self.assertEqual(self.ctx.sql_generation_attempts[0].status, "SUCCESS")
        self.assertEqual(len(self.ctx.sql_execution_attempts), 1)
        self.assertEqual(self.ctx.sql_execution_attempts[0].status, "SUCCESS")
        self.assertIsNotNone(self.ctx.sql_generation_attempts[0].latency_ms)
        self.assertIsNotNone(self.ctx.sql_execution_attempts[0].latency_ms)
        self.assertNotIn("sql_execution_latency_ms", repr(self.ctx.sql_generation_attempts[0]))

    def test_total_request_latency_remains_separate(self) -> None:
        time.sleep(0.06)
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            execute_sql(_SELECT, db_path=self.db_path)
            time.sleep(0.06)
            complete_request(self.ctx, "SUCCESS")

        execution = self.ctx.sql_execution_attempts[0]
        self.assertGreater(self.ctx.total_latency_ms, execution.latency_ms)
        blob = "\n".join(logs.output)
        execution_event = _event(blob, "SQL_EXECUTION_COMPLETED")
        request_event = _event(blob, "REQUEST_COMPLETED")
        self.assertIn("sql_execution_latency_ms", execution_event)
        self.assertNotIn("sql_execution_latency_ms", request_event)
        self.assertIn("total_latency_ms", request_event)
        self.assertNotIn("sql_generation_latency_ms", execution_event)
        self.assertNotEqual(
            request_event["total_latency_ms"],
            execution_event["sql_execution_latency_ms"],
        )

    def test_secrets_are_not_logged(self) -> None:
        secret = "sk-phase3-execution-observability-secret"
        marker = "SQL_EXEC_SHOULD_NOT_BE_LOGGED_zz"
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = secret
        try:
            with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
                execute_sql(
                    f"SELECT '{marker}' AS label",
                    db_path=self.db_path,
                )
                validation_failed(
                    {
                        "retry_count": 3,
                        "validation_error": f"blocked {secret}",
                    }
                )
            blob = "\n".join(logs.output)
            self.assertNotIn(secret, blob)
            self.assertNotIn(marker, blob)
            self.assertIn("[REDACTED]", blob)
            self.assertIn("SQL_EXECUTION_COMPLETED", blob)
            self.assertIn("SQL_EXECUTION_BLOCKED", blob)
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
