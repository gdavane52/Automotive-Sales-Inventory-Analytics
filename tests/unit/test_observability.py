"""Observability foundation: request ids, status, latency, and redacted logs."""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import LANGGRAPH_RECURSION_LIMIT
from src.langgraph.graph import run_analytics_question, stream_analytics_question
from src.observability.context import FAILED, begin_request, current_request, reset_request
from src.observability.logger import complete_request


def _success_state() -> dict:
    return {
        "in_scope": True,
        "validation_result": True,
        "query_result": pd.DataFrame({"model": ["Creta"], "sales_count": [3]}),
        "final_answer": "Creta leads.",
    }


class ObservabilityFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._question_flag = os.environ.get("OBSERVABILITY_LOG_QUESTION")
        os.environ.pop("OBSERVABILITY_LOG_QUESTION", None)

    def tearDown(self) -> None:
        if self._question_flag is None:
            os.environ.pop("OBSERVABILITY_LOG_QUESTION", None)
        else:
            os.environ["OBSERVABILITY_LOG_QUESTION"] = self._question_flag

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_request_id_is_generated(self, mock_get_graph) -> None:
        seen: dict = {}

        def invoke(state, config):
            seen["request_id"] = state["request_id"]
            seen["ctx"] = current_request()
            return _success_state()

        mock_get_graph.return_value.invoke.side_effect = invoke
        run_analytics_question("top models in Pune", thread_id="thread-1")

        parsed = uuid.UUID(seen["request_id"])
        self.assertEqual(str(parsed), seen["request_id"])
        self.assertEqual(seen["ctx"].request_id, seen["request_id"])
        self.assertIsNotNone(seen["ctx"].start_time.tzinfo)

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_request_id_is_unique(self, mock_get_graph) -> None:
        ids: list[str] = []

        def invoke(state, config):
            ids.append(state["request_id"])
            return _success_state()

        mock_get_graph.return_value.invoke.side_effect = invoke
        run_analytics_question("first", thread_id="thread-1")
        run_analytics_question("second", thread_id="thread-1")

        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])
        uuid.UUID(ids[0])
        uuid.UUID(ids[1])

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_thread_id_remains_unchanged(self, mock_get_graph) -> None:
        threads: list[str] = []
        request_ids: list[str] = []

        def invoke(state, config):
            threads.append(state["thread_id"])
            request_ids.append(state["request_id"])
            return _success_state()

        mock_get_graph.return_value.invoke.side_effect = invoke
        run_analytics_question("one", thread_id="abc123")
        run_analytics_question("two", thread_id="abc123")

        self.assertEqual(threads, ["abc123", "abc123"])
        self.assertNotEqual(request_ids[0], request_ids[1])

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_request_id_is_propagated_through_the_request(self, mock_get_graph) -> None:
        seen: dict = {}

        def invoke(state, config):
            ctx = current_request()
            seen["state_request_id"] = state["request_id"]
            seen["state_thread_id"] = state["thread_id"]
            seen["context_request_id"] = ctx.request_id if ctx else None
            seen["recursion_limit"] = config["recursion_limit"]
            return _success_state()

        mock_get_graph.return_value.invoke.side_effect = invoke
        run_analytics_question("inventory in Pune", thread_id="thread-keep")

        self.assertEqual(seen["state_request_id"], seen["context_request_id"])
        self.assertEqual(seen["state_thread_id"], "thread-keep")
        self.assertEqual(seen["recursion_limit"], LANGGRAPH_RECURSION_LIMIT)
        self.assertIsNone(current_request())

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_successful_request_produces_success_status(self, mock_get_graph) -> None:
        seen: dict = {}

        def invoke(state, config):
            seen["ctx"] = current_request()
            return _success_state()

        mock_get_graph.return_value.invoke.side_effect = invoke
        question = "question-should-stay-out-of-logs"
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            run_analytics_question(question, thread_id="thread-1")

        self.assertEqual(seen["ctx"].status, "SUCCESS")
        self.assertIsNone(seen["ctx"].error_type)
        blob = "\n".join(logs.output)
        self.assertIn("REQUEST_COMPLETED", blob)
        self.assertIn('"status": "SUCCESS"', blob)
        self.assertNotIn(question, blob)

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_failed_request_produces_failed_status(self, mock_get_graph) -> None:
        seen: dict = {}

        def invoke(state, config):
            seen["ctx"] = current_request()
            raise RuntimeError("database exploded")

        mock_get_graph.return_value.invoke.side_effect = invoke
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            result = run_analytics_question("sales", thread_id="thread-1")

        self.assertEqual(seen["ctx"].status, "FAILED")
        self.assertEqual(seen["ctx"].error_type, "RuntimeError")
        self.assertTrue(seen["ctx"].error_message)
        self.assertFalse(result["in_scope"])
        self.assertFalse(result["validation_result"])
        self.assertEqual(result["request_id"], seen["ctx"].request_id)
        self.assertEqual(result["thread_id"], "thread-1")
        blob = "\n".join(logs.output)
        self.assertIn("REQUEST_FAILED", blob)
        self.assertIn('"status": "FAILED"', blob)
        self.assertNotIn("database exploded", blob)
        payload = _event(blob, "REQUEST_FAILED")
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertIn("error_message", payload)

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_validation_failure_is_rejected_not_failed(self, mock_get_graph) -> None:
        seen: dict = {}

        def invoke(state, config):
            seen["ctx"] = current_request()
            return {
                "in_scope": True,
                "validation_result": False,
                "final_answer": "Could not produce a valid analytics query after 3 attempts.",
                "retry_count": 3,
            }

        mock_get_graph.return_value.invoke.side_effect = invoke
        run_analytics_question("delete everything", thread_id="thread-1")

        self.assertEqual(seen["ctx"].status, "REJECTED")
        self.assertEqual(seen["ctx"].error_type, "sql_validation_failed")

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_total_latency_ms_is_recorded(self, mock_get_graph) -> None:
        seen: dict = {}

        def invoke(state, config):
            seen["ctx"] = current_request()
            time.sleep(0.08)
            return _success_state()

        mock_get_graph.return_value.invoke.side_effect = invoke
        run_analytics_question("latency check", thread_id="thread-1")

        self.assertIsNotNone(seen["ctx"].total_latency_ms)
        self.assertGreaterEqual(seen["ctx"].total_latency_ms, 70)
        self.assertIsNotNone(seen["ctx"].end_time)
        self.assertIsNotNone(seen["ctx"].end_time.tzinfo)

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_streaming_request_measures_completion(self, mock_get_graph) -> None:
        seen: dict = {}

        def stream(state, config, stream_mode="updates"):
            seen["ctx"] = current_request()
            seen["request_id"] = state["request_id"]
            seen["thread_id"] = state["thread_id"]
            yield {"validate_question": {"in_scope": True}}
            seen["mid_latency"] = seen["ctx"].total_latency_ms
            seen["mid_status"] = seen["ctx"].status
            time.sleep(0.12)
            yield {
                "execute_sql": {
                    "validation_result": True,
                    "query_result": pd.DataFrame({"model": ["Creta"]}),
                }
            }

        mock_get_graph.return_value.stream.side_effect = stream
        updates = list(
            stream_analytics_question("streamed question", thread_id="thread-stream")
        )

        self.assertEqual(len(updates), 2)
        self.assertIsNone(seen["mid_latency"])
        self.assertIsNone(seen["mid_status"])
        self.assertGreaterEqual(seen["ctx"].total_latency_ms, 100)
        self.assertEqual(seen["ctx"].status, "SUCCESS")
        self.assertEqual(seen["request_id"], seen["ctx"].request_id)
        self.assertEqual(seen["thread_id"], "thread-stream")
        self.assertIsNotNone(seen["ctx"].end_time.tzinfo)
        self.assertIsNone(current_request())

    @patch("src.langgraph.graph.get_analytics_graph")
    def test_run_analytics_question_measures_latency(self, mock_get_graph) -> None:
        seen: dict = {}

        def invoke(state, config):
            seen["ctx"] = current_request()
            time.sleep(0.05)
            return _success_state()

        mock_get_graph.return_value.invoke.side_effect = invoke
        started = time.perf_counter()
        run_analytics_question("run latency", thread_id="thread-run")
        elapsed_ms = (time.perf_counter() - started) * 1000

        self.assertGreaterEqual(seen["ctx"].total_latency_ms, 40)
        self.assertLessEqual(seen["ctx"].total_latency_ms, elapsed_ms + 30)
        self.assertEqual(seen["ctx"].status, "SUCCESS")

    def test_no_secrets_are_written_to_logs(self) -> None:
        secret = "sk-phase1-observability-secret"
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = secret
        ctx = begin_request(thread_id="thread-1", user_question="hello")
        try:
            with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
                complete_request(
                    ctx,
                    FAILED,
                    error_type="RuntimeError",
                    error_message=f"failed with key {secret} and password=hunter2",
                )
            blob = "\n".join(logs.output)
            self.assertNotIn(secret, blob)
            self.assertNotIn("hunter2", blob)
            self.assertIn("[REDACTED]", blob)
            self.assertIn("REQUEST_FAILED", blob)
            self.assertNotIn("hello", blob)
        finally:
            reset_request(ctx)
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
