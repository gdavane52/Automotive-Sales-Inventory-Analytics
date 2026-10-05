"""Session demo rate limit: quota, cooldown, and no workflow on block."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.config.rate_limit import (
    REASON_COOLDOWN,
    REASON_EMPTY,
    REASON_QUOTA,
    admit_question,
    ensure_rate_limit_state,
    guard_submission,
)
from src.langgraph.nodes import MAX_SQL_GENERATION_ATTEMPTS


def _state(**overrides) -> dict:
    state = {
        "question_count": 0,
        "last_request_time": None,
        "thread_id": "thread-1",
        "turns": [{"question": "earlier", "answer": "answer"}],
        "chats": [{"thread_id": "thread-1"}],
    }
    state.update(overrides)
    return state


class RateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._quota = patch("src.config.rate_limit.MAX_QUESTIONS_PER_SESSION", 10)
        self._cooldown = patch("src.config.rate_limit.REQUEST_COOLDOWN_SECONDS", 5.0)
        self._quota.start()
        self._cooldown.start()
        self.addCleanup(self._quota.stop)
        self.addCleanup(self._cooldown.stop)

    def test_first_request_is_accepted(self) -> None:
        state = _state()
        decision = admit_question(state, "top models in Pune", now=100.0)
        self.assertTrue(decision.allowed)
        self.assertIsNone(decision.reason)

    def test_accepted_request_increments_question_count_once(self) -> None:
        state = _state()
        admit_question(state, "top models in Pune", now=100.0)
        self.assertEqual(state["question_count"], 1)
        self.assertEqual(state["last_request_time"], 100.0)

    def test_immediate_second_request_is_blocked_by_cooldown(self) -> None:
        state = _state()
        admit_question(state, "first", now=100.0)
        decision = admit_question(state, "second", now=100.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, REASON_COOLDOWN)
        self.assertEqual(decision.retry_after_seconds, 5)
        self.assertEqual(
            decision.user_message(),
            "Please wait 5 seconds before submitting another question.",
        )

    def test_request_is_accepted_after_cooldown(self) -> None:
        state = _state()
        admit_question(state, "first", now=100.0)
        decision = admit_question(state, "second", now=105.0)
        self.assertTrue(decision.allowed)
        self.assertEqual(state["question_count"], 2)
        self.assertEqual(state["last_request_time"], 105.0)

    def test_session_quota_blocks_the_eleventh_question(self) -> None:
        state = _state()
        for index in range(10):
            decision = admit_question(state, f"question {index}", now=index * 5.0)
            self.assertTrue(decision.allowed)
        blocked = admit_question(state, "one more", now=50.0)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, REASON_QUOTA)
        self.assertEqual(
            blocked.user_message(),
            "Demo question limit reached for this session. Please try again later.",
        )

    def test_cooldown_block_does_not_increment_question_count(self) -> None:
        state = _state()
        admit_question(state, "first", now=10.0)
        admit_question(state, "second", now=12.0)
        self.assertEqual(state["question_count"], 1)
        self.assertEqual(state["last_request_time"], 10.0)

    def test_quota_block_does_not_increment_question_count(self) -> None:
        state = _state(question_count=10, last_request_time=0.0)
        admit_question(state, "another", now=100.0)
        self.assertEqual(state["question_count"], 10)
        self.assertEqual(state["last_request_time"], 0.0)

    def test_empty_question_does_not_consume_quota(self) -> None:
        state = _state(question_count=2, last_request_time=20.0)
        calls: list[str] = []
        decision = guard_submission(
            state,
            "   ",
            lambda: calls.append("run"),
            now=100.0,
            thread_id="thread-1",
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, REASON_EMPTY)
        self.assertEqual(calls, [])
        self.assertEqual(state["question_count"], 2)
        self.assertEqual(state["last_request_time"], 20.0)

    @patch("src.langgraph.graph.stream_analytics_question")
    @patch("src.langgraph.graph.run_analytics_question")
    @patch("src.observability.logger.log_request_started")
    def test_blocked_request_does_not_invoke_workflow(
        self,
        log_request_started,
        run_analytics_question,
        stream_analytics_question,
    ) -> None:
        state = _state()
        calls: list[str] = []
        first = guard_submission(
            state,
            "first",
            lambda: calls.append("run"),
            now=0.0,
            thread_id="thread-1",
        )
        second = guard_submission(
            state,
            "second",
            lambda: calls.append("run"),
            now=1.0,
            thread_id="thread-1",
        )
        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertEqual(calls, ["run"])
        stream_analytics_question.assert_not_called()
        run_analytics_question.assert_not_called()
        log_request_started.assert_not_called()

    def test_limiter_does_not_change_conversation_state(self) -> None:
        turns = [{"question": "earlier", "answer": "answer"}]
        state = _state(turns=turns)
        guard_submission(state, "follow up", lambda: None, now=30.0, thread_id=state["thread_id"])
        blocked = guard_submission(
            state,
            "too soon",
            lambda: (_ for _ in ()).throw(AssertionError("workflow ran")),
            now=31.0,
            thread_id=state["thread_id"],
        )
        self.assertFalse(blocked.allowed)
        self.assertEqual(state["thread_id"], "thread-1")
        self.assertIs(state["turns"], turns)
        self.assertEqual(state["chats"], [{"thread_id": "thread-1"}])
        self.assertEqual(state["question_count"], 1)

    def test_repeated_initialization_does_not_reset_limits(self) -> None:
        state = _state()
        admit_question(state, "first", now=5.0)
        ensure_rate_limit_state(state)
        ensure_rate_limit_state(state)
        self.assertEqual(state["question_count"], 1)
        self.assertEqual(state["last_request_time"], 5.0)

    def test_sql_retry_cap_is_unchanged(self) -> None:
        self.assertEqual(MAX_SQL_GENERATION_ATTEMPTS, 3)

    def test_rate_limit_logs_do_not_start_a_request(self) -> None:
        state = _state()
        with self.assertLogs("automotive_analytics.observability", level="INFO") as logs:
            guard_submission(state, "first", lambda: None, now=0.0, thread_id="thread-1")
            guard_submission(state, "second", lambda: None, now=1.0, thread_id="thread-1")
        events = [json.loads(entry[entry.index("{") :]) for entry in logs.output]
        self.assertEqual(
            [event["event"] for event in events],
            ["RATE_LIMIT_ACCEPTED", "RATE_LIMIT_BLOCKED"],
        )
        blocked = events[1]
        self.assertEqual(blocked["reason"], REASON_COOLDOWN)
        self.assertEqual(blocked["question_count"], 1)
        self.assertEqual(blocked["retry_after_seconds"], 4)
        self.assertEqual(blocked["thread_id"], "thread-1")
        self.assertNotIn("request_id", blocked)
        self.assertNotIn("user_question", blocked)
        self.assertTrue(all("REQUEST_STARTED" not in entry for entry in logs.output))
