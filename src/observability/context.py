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
class LlmCall:
    """One underlying model invocation, separate from SQL task metrics.

    ``SUCCESS`` means the model call itself finished. A later domain rejection
    or SQL validation failure does not change this status.

    Token fields stay None when the provider does not report them. They are
    never estimated and never stored as zero in place of a missing value.
    ``token_usage_available`` is True only when a usage payload was present,
    even if one of the counts inside it was omitted.
    """

    call_id: str
    request_id: str
    thread_id: str | None
    component: str
    model: str | None
    start_time: datetime
    end_time: datetime
    llm_latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    token_usage_available: bool
    status: str
    error_type: str | None = None
    error_message: str | None = None

    def __repr__(self) -> str:
        return (
            "LlmCall("
            f"call_id={self.call_id!r}, "
            f"request_id={self.request_id!r}, "
            f"thread_id={self.thread_id!r}, "
            f"component={self.component!r}, "
            f"model={self.model!r}, "
            f"status={self.status!r}, "
            f"llm_latency_ms={self.llm_latency_ms!r})"
        )


def _sum_present(calls: list[LlmCall], field_name: str) -> float | int | None:
    if not calls:
        return None
    values: list[float | int] = []
    for call in calls:
        value = getattr(call, field_name)
        if value is None:
            return None
        values.append(value)
    return sum(values)


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
    llm_calls: list[LlmCall] = field(default_factory=list)

    @property
    def total_llm_latency_ms(self) -> float | None:
        """Sum of per-call latencies. None when no call was timed.

        A missing latency is not treated as zero. If any call lacks a latency,
        the exact total is unknown and this returns None.
        """
        return _sum_present(self.llm_calls, "llm_latency_ms")

    @property
    def total_input_tokens(self) -> int | None:
        """Sum of input tokens, or None if any call did not report that count."""
        return _sum_present(self.llm_calls, "input_tokens")

    @property
    def total_output_tokens(self) -> int | None:
        """Sum of output tokens, or None if any call did not report that count."""
        return _sum_present(self.llm_calls, "output_tokens")

    @property
    def total_tokens(self) -> int | None:
        """Sum of total tokens, or None if any call did not report that count."""
        return _sum_present(self.llm_calls, "total_tokens")

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
