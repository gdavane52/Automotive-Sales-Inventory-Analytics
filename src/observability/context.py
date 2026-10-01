"""In-process request context. Holds identifiers and timing, never secrets."""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SUCCESS = "SUCCESS"
FAILED = "FAILED"
REJECTED = "REJECTED"
BLOCKED = "BLOCKED"

_MISSING = object()

_CURRENT: ContextVar[RequestContext | None] = ContextVar(
    "analytics_request_context",
    default=None,
)


@dataclass
class SqlGenerationAttempt:
    """One SQL-generation LLM call. Validation and execution are not included.

    ``SUCCESS`` means the LLM call returned. Guardrails may still clear the
    SQL before validation. ``FAILED`` means the LLM call raised.
    """

    attempt: int
    request_id: str
    thread_id: str | None
    status: str
    start_time: datetime
    end_time: datetime
    latency_ms: float
    generated_sql: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __repr__(self) -> str:
        return (
            "SqlGenerationAttempt("
            f"attempt={self.attempt!r}, "
            f"request_id={self.request_id!r}, "
            f"thread_id={self.thread_id!r}, "
            f"status={self.status!r}, "
            f"latency_ms={self.latency_ms!r})"
        )


@dataclass
class SqlExecutionAttempt:
    """One SQL execution decision for the active request.

    ``SUCCESS`` and ``FAILED`` are real SQLite runs. ``BLOCKED`` means an
    existing safety or validation check stopped the query before execution,
    so ``latency_ms`` stays None.
    """

    attempt: int
    request_id: str
    thread_id: str | None
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    latency_ms: float | None = None
    row_count: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    reason: str | None = None

    def __repr__(self) -> str:
        return (
            "SqlExecutionAttempt("
            f"attempt={self.attempt!r}, "
            f"request_id={self.request_id!r}, "
            f"thread_id={self.thread_id!r}, "
            f"status={self.status!r}, "
            f"latency_ms={self.latency_ms!r}, "
            f"row_count={self.row_count!r})"
        )


@dataclass
class RequestContext:
    """One analytics invocation. ``request_id`` is unique per call.

    ``thread_id`` is the conversation id supplied by the caller (Streamlit
    session). This object does not mint a new thread id.
    """

    request_id: str
    thread_id: str | None
    user_question: str
    start_time: datetime
    perf_start: float
    end_time: datetime | None = None
    total_latency_ms: float | None = None
    status: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    finished: bool = False
    reset_token: Token[RequestContext | None] | None = None
    sql_generation_attempts: list[SqlGenerationAttempt] = field(default_factory=list)
    sql_execution_attempts: list[SqlExecutionAttempt] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            "RequestContext("
            f"request_id={self.request_id!r}, "
            f"thread_id={self.thread_id!r}, "
            f"status={self.status!r})"
        )


def begin_request(
    *,
    thread_id: str | None,
    user_question: str,
) -> RequestContext:
    """Start a request scope and generate a new ``request_id``."""
    cleaned_thread = (thread_id or "").strip() or None
    ctx = RequestContext(
        request_id=str(uuid.uuid4()),
        thread_id=cleaned_thread,
        user_question=user_question or "",
        start_time=datetime.now(timezone.utc),
        perf_start=time.perf_counter(),
    )
    ctx.reset_token = _CURRENT.set(ctx)
    return ctx


def current_request() -> RequestContext | None:
    """Return the active request, if a run or stream is in progress."""
    return _CURRENT.get()


def finish_request(
    ctx: RequestContext,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Stamp end time, latency, and status. Does not log or clear the scope."""
    if ctx.finished:
        return
    ctx.finished = True
    ctx.end_time = datetime.now(timezone.utc)
    ctx.total_latency_ms = (time.perf_counter() - ctx.perf_start) * 1000.0
    ctx.status = status
    ctx.error_type = error_type or None
    ctx.error_message = (error_message or "").strip() or None


def reset_request(ctx: RequestContext) -> None:
    """Pop this request off the context var. Safe to call once."""
    token = ctx.reset_token
    if token is None:
        return
    ctx.reset_token = None
    try:
        _CURRENT.reset(token)
    except ValueError:
        _CURRENT.set(None)


def classify_completed_request(state: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Map a finished graph state to SUCCESS, REJECTED, or FAILED.

    Domain and SQL-validation rejections are REJECTED, not crashes.
    A ``query_result`` of None after valid SQL is FAILED (execution).
    A missing ``query_result`` key is not treated as an execution failure.
    """
    if state.get("in_scope") is False:
        message = (state.get("final_answer") or "").strip() or None
        return REJECTED, "out_of_scope", message

    if state.get("validation_result") is False:
        message = (
            (state.get("final_answer") or "").strip()
            or (state.get("validation_error") or "").strip()
            or None
        )
        return REJECTED, "sql_validation_failed", message

    query_result = state.get("query_result", _MISSING)
    if state.get("validation_result") is True and query_result is None:
        message = (
            (state.get("final_answer") or "").strip()
            or (state.get("validation_error") or "").strip()
            or None
        )
        return FAILED, "sql_execution_failed", message

    return SUCCESS, None, None
