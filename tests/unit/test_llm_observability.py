"""Generic LLM-call observability, separate from SQL generation and execution."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.answer_agent import generate_business_answer
from src.agents.llm import get_chat_llm
from src.agents.sql_agent import AutomotiveScope, SQLGeneration, classify_automotive_scope, generate_sql
from src.db.service import execute_sql
from src.langgraph.routing import route_after_validation
from src.observability.context import begin_request, current_request, reset_request
from src.observability.llm import LlmObservabilityHandler, llm_component
from src.observability.logger import complete_request

_SELECT = "SELECT 1 AS n"


def _result(message: AIMessage) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _finish(
    handler: LlmObservabilityHandler,
    run_id: uuid.UUID,
    message: AIMessage,
) -> None:
    handler.on_llm_end(_result(message), run_id=run_id)


class _SlowAnswerModel(GenericFakeChatModel):
    """Sleeps inside the stream so latency includes consumption, not just setup."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(0.06)
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        time.sleep(0.06)


class LlmObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = LlmObservabilityHandler()
        self.ctx = begin_request(thread_id="conversation-123", user_question="top models")

    def tearDown(self) -> None:
        if current_request() is self.ctx:
            reset_request(self.ctx)

    def _start(self, model: str = "unit-test-model") -> uuid.UUID:
        run_id = uuid.uuid4()
        with llm_component("generate_sql"):
            self.handler.on_chat_model_start(
                {"kwargs": {"model_name": model}},
                [[]],
                run_id=run_id,
                invocation_params={"model": model},
            )
        return run_id

    def test_successful_llm_call_is_observed(self) -> None:
        run_id = self._start()
        _finish(
            self.handler,
            run_id,
            AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            ),
        )
        call = self.ctx.llm_calls[0]
        self.assertEqual(call.status, "SUCCESS")
        self.assertIsNone(call.error_type)
        self.assertEqual(len(self.ctx.sql_generation_attempts), 0)

    def test_failed_llm_call_is_observed(self) -> None:
        run_id = self._start()
        self.handler.on_llm_error(RuntimeError("model down"), run_id=run_id)
        call = self.ctx.llm_calls[0]
        self.assertEqual(call.status, "FAILED")
        self.assertEqual(call.error_type, "RuntimeError")
        self.assertTrue(call.error_message)
        self.assertNotIn("model down", call.error_message or "")
        self.assertFalse(call.token_usage_available)

    def test_llm_latency_is_recorded(self) -> None:
        run_id = self._start()
        time.sleep(0.06)
        _finish(self.handler, run_id, AIMessage(content="ok"))
        self.assertGreaterEqual(self.ctx.llm_calls[0].llm_latency_ms, 50)
        self.assertIsNotNone(self.ctx.llm_calls[0].start_time.tzinfo)
        self.assertIsNotNone(self.ctx.llm_calls[0].end_time.tzinfo)

    def test_model_name_comes_from_the_invocation(self) -> None:
        run_id = self._start("gpt-phase4-test")
        _finish(self.handler, run_id, AIMessage(content="ok"))
        self.assertEqual(self.ctx.llm_calls[0].model, "gpt-phase4-test")

    def test_request_and_thread_ids_are_reused(self) -> None:
        run_id = self._start()
        _finish(self.handler, run_id, AIMessage(content="ok"))
        call = self.ctx.llm_calls[0]
        self.assertEqual(call.request_id, self.ctx.request_id)
        self.assertEqual(call.thread_id, "conversation-123")

    def test_call_ids_are_unique(self) -> None:
        first = self._start()
        _finish(self.handler, first, AIMessage(content="one"))
        second = self._start()
        _finish(self.handler, second, AIMessage(content="two"))
        ids = [call.call_id for call in self.ctx.llm_calls]
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])
        uuid.UUID(ids[0])
        uuid.UUID(ids[1])

    def test_token_usage_is_captured_when_present(self) -> None:
        run_id = self._start()
        _finish(
            self.handler,
            run_id,
            AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            ),
        )
        call = self.ctx.llm_calls[0]
        self.assertTrue(call.token_usage_available)
        self.assertEqual(call.input_tokens, 11)
        self.assertEqual(call.output_tokens, 7)
        self.assertEqual(call.total_tokens, 18)

    def test_missing_token_usage_is_not_zero(self) -> None:
        run_id = self._start()
        _finish(self.handler, run_id, AIMessage(content="ok"))
        call = self.ctx.llm_calls[0]
        self.assertFalse(call.token_usage_available)
        self.assertIsNone(call.input_tokens)
        self.assertIsNone(call.output_tokens)
        self.assertIsNone(call.total_tokens)
        self.assertNotEqual(call.input_tokens, 0)

    def test_partial_token_usage_keeps_missing_fields_as_none(self) -> None:
        run_id = self._start()
        _finish(
            self.handler,
            run_id,
            AIMessage(
                content="ok",
                response_metadata={"token_usage": {"prompt_tokens": 4}},
            ),
        )
        call = self.ctx.llm_calls[0]
        self.assertTrue(call.token_usage_available)
        self.assertEqual(call.input_tokens, 4)
        self.assertIsNone(call.output_tokens)
        self.assertIsNone(call.total_tokens)

    def test_multiple_calls_append(self) -> None:
        first = self._start()
        _finish(
            self.handler,
            first,
            AIMessage(
                content="a",
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        )
        second = self._start()
        _finish(
            self.handler,
            second,
            AIMessage(
                content="b",
                usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            ),
        )
        self.assertEqual(
            [call.component for call in self.ctx.llm_calls],
            ["generate_sql", "generate_sql"],
        )
        self.assertEqual(len(self.ctx.llm_calls), 2)

    def test_request_latency_and_token_aggregates(self) -> None:
        first = self._start()
        time.sleep(0.02)
        _finish(
            self.handler,
            first,
            AIMessage(
                content="a",
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        )
        second = self._start()
        time.sleep(0.02)
        _finish(
            self.handler,
            second,
            AIMessage(
                content="b",
                usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            ),
        )
        self.assertAlmostEqual(
            self.ctx.total_llm_latency_ms,
            sum(call.llm_latency_ms for call in self.ctx.llm_calls),
        )
        self.assertEqual(self.ctx.total_input_tokens, 4)
        self.assertEqual(self.ctx.total_output_tokens, 5)
        self.assertEqual(self.ctx.total_tokens, 9)

    def test_aggregate_is_none_when_any_call_lacks_tokens(self) -> None:
        first = self._start()
        _finish(
            self.handler,
            first,
            AIMessage(
                content="a",
                usage_metadata={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            ),
        )
        second = self._start()
        _finish(self.handler, second, AIMessage(content="b"))
        self.assertIsNone(self.ctx.total_input_tokens)
        self.assertIsNone(self.ctx.total_output_tokens)
        self.assertIsNone(self.ctx.total_tokens)
        self.assertIsNotNone(self.ctx.total_llm_latency_ms)
        self.assertGreater(self.ctx.total_llm_latency_ms, 0)

    def test_streaming_latency_covers_stream_consumption(self) -> None:
        handler = LlmObservabilityHandler()
        model = _SlowAnswerModel(
            messages=iter([AIMessage(content="Creta leads.")]),
            callbacks=[handler],
        )
        frame = pd.DataFrame({"model": ["Creta"], "n": [3]})
        with patch("src.agents.answer_agent.get_chat_llm", return_value=model):
            answer = generate_business_answer("top models", frame, _SELECT)
        self.assertIn("Creta", answer)
        call = self.ctx.llm_calls[0]
        self.assertEqual(call.component, "generate_answer")
        self.assertEqual(call.status, "SUCCESS")
        self.assertGreaterEqual(call.llm_latency_ms, 100)

    def test_scope_rejection_still_records_llm_success(self) -> None:
        handler = self.handler

        def _invoke(_payload):
            run_id = uuid.uuid4()
            handler.on_chat_model_start(
                {"kwargs": {"model_name": "scope-model"}},
                [[]],
                run_id=run_id,
                invocation_params={"model": "scope-model"},
            )
            handler.on_llm_end(
                _result(AIMessage(content="{}")),
                run_id=run_id,
            )
            return AutomotiveScope(in_scope=False, reason="This is about the weather.")

        class _ScopeModel:
            def with_structured_output(self, _schema):
                return RunnableLambda(_invoke)

        with patch("src.agents.sql_agent.get_chat_llm", return_value=_ScopeModel()):
            scope = classify_automotive_scope("What is the weather today?")
        self.assertFalse(scope.in_scope)
        self.assertEqual(self.ctx.llm_calls[0].status, "SUCCESS")
        self.assertEqual(self.ctx.llm_calls[0].component, "classify_automotive_scope")

    def test_secrets_are_not_recorded(self) -> None:
        secret = "sk-phase4-llm-observability-secret"
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = secret
        try:
            run_id = uuid.uuid4()
            prompt = "prompt " + secret
            with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
                self.handler.on_chat_model_start(
                    {"kwargs": {"model_name": "unit-test-model"}},
                    [[{"content": prompt}]],
                    run_id=run_id,
                    invocation_params={"model": "unit-test-model"},
                )
                self.handler.on_llm_error(
                    RuntimeError(f"provider failed {secret}"),
                    run_id=run_id,
                )
            blob = "\n".join(logs.output)
            self.assertNotIn(secret, blob)
            self.assertNotIn(prompt, blob)
            self.assertIn("LLM_FAILED", blob)
            self.assertNotIn(secret, self.ctx.llm_calls[0].error_message or "")
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous

    def test_sql_generation_phase_remains_separate(self) -> None:
        handler = self.handler

        def _invoke(_payload):
            run_id = uuid.uuid4()
            handler.on_chat_model_start(
                {"kwargs": {"model_name": "sql-model"}},
                [[]],
                run_id=run_id,
                invocation_params={"model": "sql-model"},
            )
            handler.on_llm_end(_result(AIMessage(content="select")), run_id=run_id)
            return SQLGeneration(sql=_SELECT, explanation="Constant.", tables_used=[])

        class _SqlModel:
            def with_structured_output(self, _schema):
                return RunnableLambda(_invoke)

        schema = {"table_names": [], "tables": {}}
        with patch("src.agents.sql_agent.get_chat_llm", return_value=_SqlModel()):
            generated = generate_sql("how many", schema)
        self.assertEqual(generated["sql"], _SELECT)
        self.assertEqual(len(self.ctx.sql_generation_attempts), 1)
        self.assertEqual(self.ctx.sql_generation_attempts[0].status, "SUCCESS")
        self.assertEqual(len(self.ctx.llm_calls), 1)
        self.assertEqual(self.ctx.llm_calls[0].component, "generate_sql")
        self.assertEqual(self.ctx.llm_calls[0].status, "SUCCESS")

    def test_request_and_sql_execution_observations_stay_independent(self) -> None:
        run_id = self._start()
        _finish(
            self.handler,
            run_id,
            AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        )
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        handle.close()
        db_path = Path(handle.name)
        sqlite3.connect(db_path).close()
        try:
            execute_sql(_SELECT, db_path=db_path)
        finally:
            db_path.unlink(missing_ok=True)
        time.sleep(0.03)
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            complete_request(self.ctx, "SUCCESS")
        self.assertEqual(self.ctx.status, "SUCCESS")
        self.assertGreater(self.ctx.total_latency_ms, self.ctx.total_llm_latency_ms)
        self.assertEqual(len(self.ctx.sql_execution_attempts), 1)
        self.assertEqual(self.ctx.sql_execution_attempts[0].status, "SUCCESS")
        self.assertEqual(len(self.ctx.llm_calls), 1)
        request_event = _event("\n".join(logs.output), "REQUEST_COMPLETED")
        self.assertIn("total_latency_ms", request_event)
        self.assertNotIn("llm_latency_ms", request_event)
        self.assertEqual(
            route_after_validation({"validation_result": True, "retry_count": 1}),
            "execute_sql",
        )

    def test_chat_client_uses_the_observability_handler(self) -> None:
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-phase4-handler-check"
        try:
            client = get_chat_llm()
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous
        self.assertTrue(
            any(isinstance(callback, LlmObservabilityHandler) for callback in client.callbacks)
        )

    def test_log_contains_metadata_not_prompt_text(self) -> None:
        run_id = self._start()
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            _finish(
                self.handler,
                run_id,
                AIMessage(
                    content="full model response that must stay out of logs",
                    usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                ),
            )
        blob = "\n".join(logs.output)
        payload = _event(blob, "LLM_COMPLETED")
        self.assertEqual(payload["status"], "SUCCESS")
        self.assertEqual(payload["component"], "generate_sql")
        self.assertEqual(payload["input_tokens"], 1)
        self.assertNotIn("full model response", blob)


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
