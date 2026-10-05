"""Session-level demo rate limiting for analytics questions.

This is lightweight session-level demo rate limiting intended to control
excessive LLM usage. It is not a production distributed/IP-based rate limiter.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

from src.config.settings import MAX_QUESTIONS_PER_SESSION, REQUEST_COOLDOWN_SECONDS
from src.observability.logger import log_rate_limit

REASON_COOLDOWN = "cooldown"
REASON_QUOTA = "quota"
REASON_EMPTY = "empty"

QUESTION_COUNT_KEY = "question_count"
LAST_REQUEST_TIME_KEY = "last_request_time"

QUOTA_MESSAGE = "Demo question limit reached for this session. Please try again later."


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    reason: str | None = None
    retry_after_seconds: int | None = None

    def user_message(self) -> str | None:
        if self.reason == REASON_COOLDOWN:
            seconds = self.retry_after_seconds if self.retry_after_seconds is not None else 1
            return f"Please wait {seconds} seconds before submitting another question."
        if self.reason == REASON_QUOTA:
            return QUOTA_MESSAGE
        return None


def ensure_rate_limit_state(state: MutableMapping) -> None:
    """Initialize limiter keys once. Later calls leave stored values unchanged."""
    if QUESTION_COUNT_KEY not in state:
        state[QUESTION_COUNT_KEY] = 0
    if LAST_REQUEST_TIME_KEY not in state:
        state[LAST_REQUEST_TIME_KEY] = None


def admit_question(
    state: MutableMapping,
    question: str,
    *,
    now: float | None = None,
) -> RateLimitDecision:
    """Accept one new user question or explain why it is blocked.

    Increments ``question_count`` and stores ``last_request_time`` only after
    the question is accepted. Empty text, cooldown, and quota exhaustion do
    not consume a question. SQL retries are outside this helper: one accepted
    user question counts once.
    """
    ensure_rate_limit_state(state)
    if not (question or "").strip():
        return RateLimitDecision(allowed=False, reason=REASON_EMPTY)

    clock = time.monotonic() if now is None else now
    count = int(state.get(QUESTION_COUNT_KEY) or 0)
    if count >= MAX_QUESTIONS_PER_SESSION:
        return RateLimitDecision(allowed=False, reason=REASON_QUOTA)

    last = state.get(LAST_REQUEST_TIME_KEY)
    if last is not None:
        remaining = REQUEST_COOLDOWN_SECONDS - (clock - float(last))
        if remaining > 0:
            return RateLimitDecision(
                allowed=False,
                reason=REASON_COOLDOWN,
                retry_after_seconds=max(1, math.ceil(remaining)),
            )

    state[QUESTION_COUNT_KEY] = count + 1
    state[LAST_REQUEST_TIME_KEY] = clock
    return RateLimitDecision(allowed=True)


def guard_submission(
    state: MutableMapping,
    question: str,
    start_workflow: Callable[[], None],
    *,
    now: float | None = None,
    thread_id: str | None = None,
) -> RateLimitDecision:
    """Run ``start_workflow`` only when the question is accepted.

    Blocked questions are logged as ``RATE_LIMIT_BLOCKED`` and do not start
    an analytics request. Accepted questions are logged as
    ``RATE_LIMIT_ACCEPTED`` before the workflow runs.
    """
    decision = admit_question(state, question, now=now)
    if decision.reason == REASON_EMPTY:
        return decision
    log_rate_limit(
        allowed=decision.allowed,
        thread_id=thread_id,
        question_count=int(state.get(QUESTION_COUNT_KEY) or 0),
        reason=decision.reason,
        retry_after_seconds=decision.retry_after_seconds,
    )
    if decision.allowed:
        start_workflow()
    return decision
