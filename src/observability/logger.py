"""Structured request logs. Payloads are passed through secret redaction."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from src.config.logging import get_logger, redact_secrets
from src.config.safety import public_error_message
from src.config.settings import observability_log_question, observability_log_sql
from src.observability.context import (
    BLOCKED,
    FAILED,
    RequestContext,
    SqlExecutionAttempt,
    SqlGenerationAttempt,
    current_request,
    finish_request,
)

logger = get_logger("observability")

_MAX_ERROR_CHARS = 500


def log_request_started(ctx: RequestContext) -> None:
    logger.info("%s", _payload("REQUEST_STARTED", ctx))


def complete_request(
    ctx: RequestContext,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Record the outcome once and emit REQUEST_COMPLETED or REQUEST_FAILED."""
    if ctx.finished:
        return
    safe_message = _safe_error_message(error_message)
    finish_request(ctx, status, error_type=error_type, error_message=safe_message)
    event = "REQUEST_FAILED" if status == FAILED else "REQUEST_COMPLETED"
    logger.info("%s", _payload(event, ctx))


def observe_sql_generation(
    *,
    status: str,
    perf_start: float,
    start_time,
    perf_end: float | None = None,
    generated_sql: str | None = None,
    error: BaseException | None = None,
) -> None:
    """Append one SQL-generation attempt to the active request and log it.

    No-op when no Phase 1 request is active, so direct generator calls keep
    their current behavior. Does not log the SQL unless explicitly enabled.
    """
    ctx = current_request()
    if ctx is None:
        return
    stopped = time.perf_counter() if perf_end is None else perf_end
    latency_ms = max(0.0, (stopped - perf_start) * 1000.0)
    end_time = datetime.now(timezone.utc)
    error_type = type(error).__name__ if error is not None else None
    error_message = _safe_error_message(public_error_message(error)) if error else None
    attempt = SqlGenerationAttempt(
        attempt=len(ctx.sql_generation_attempts) + 1,
        request_id=ctx.request_id,
        thread_id=ctx.thread_id,
        status=status,
        start_time=start_time,
        end_time=end_time,
        latency_ms=latency_ms,
        generated_sql=generated_sql,
        error_type=error_type,
        error_message=error_message,
    )
    ctx.sql_generation_attempts.append(attempt)
    event = "SQL_GENERATION_FAILED" if status == FAILED else "SQL_GENERATION_COMPLETED"
    logger.info("%s", _sql_generation_payload(event, attempt))


def _sql_generation_payload(event: str, attempt: SqlGenerationAttempt) -> str:
    body: dict[str, object] = {
        "event": event,
        "request_id": attempt.request_id,
        "thread_id": attempt.thread_id,
        "attempt": attempt.attempt,
        "sql_generation_status": attempt.status,
        "sql_generation_latency_ms": int(round(attempt.latency_ms)),
    }
    if attempt.error_type:
        body["error_type"] = attempt.error_type
    if attempt.error_message:
        body["error_message"] = attempt.error_message
    if observability_log_sql() and attempt.generated_sql:
        body["generated_sql"] = attempt.generated_sql
    return redact_secrets(json.dumps(body, ensure_ascii=True))


def observe_sql_execution(
    *,
    status: str,
    perf_start: float | None = None,
    perf_end: float | None = None,
    start_time: datetime | None = None,
    row_count: int | None = None,
    error: BaseException | None = None,
    reason: str | None = None,
) -> None:
    """Append one SQL-execution observation. No-op without an active request.

    Latency is recorded only when a timer was started around the SQLite call.
    Blocked queries pass no timer and keep ``latency_ms`` as None.
    """
    ctx = current_request()
    if ctx is None:
        return
    if perf_start is None:
        latency_ms = None
        end_time = None
    else:
        stopped = time.perf_counter() if perf_end is None else perf_end
        latency_ms = max(0.0, (stopped - perf_start) * 1000.0)
        end_time = datetime.now(timezone.utc)
    if status == BLOCKED:
        error_type = None
        error_message = None
        safe_reason = _safe_error_message(
            reason or (public_error_message(error) if error else None) or "SQL validation failed"
        )
    else:
        error_type = type(error).__name__ if error is not None else None
        error_message = _safe_error_message(public_error_message(error)) if error else None
        safe_reason = None
    attempt = SqlExecutionAttempt(
        attempt=len(ctx.sql_execution_attempts) + 1,
        request_id=ctx.request_id,
        thread_id=ctx.thread_id,
        status=status,
        start_time=start_time,
        end_time=end_time,
        latency_ms=latency_ms,
        row_count=row_count,
        error_type=error_type,
        error_message=error_message,
        reason=safe_reason,
    )
    ctx.sql_execution_attempts.append(attempt)
    if status == BLOCKED:
        event = "SQL_EXECUTION_BLOCKED"
    elif status == FAILED:
        event = "SQL_EXECUTION_FAILED"
    else:
        event = "SQL_EXECUTION_COMPLETED"
    logger.info("%s", _sql_execution_payload(event, attempt))


def _sql_execution_payload(event: str, attempt: SqlExecutionAttempt) -> str:
    body: dict[str, object] = {
        "event": event,
        "request_id": attempt.request_id,
        "thread_id": attempt.thread_id,
        "attempt": attempt.attempt,
        "sql_execution_status": attempt.status,
    }
    if attempt.latency_ms is not None:
        body["sql_execution_latency_ms"] = int(round(attempt.latency_ms))
    if attempt.row_count is not None:
        body["row_count"] = attempt.row_count
    if attempt.error_type:
        body["error_type"] = attempt.error_type
    if attempt.error_message:
        body["error_message"] = attempt.error_message
    if attempt.reason:
        body["reason"] = attempt.reason
    return redact_secrets(json.dumps(body, ensure_ascii=True))


def _safe_error_message(message: str | None) -> str | None:
    text = redact_secrets((message or "").strip())
    if not text:
        return None
    if len(text) > _MAX_ERROR_CHARS:
        return text[: _MAX_ERROR_CHARS - 1] + "…"
    return text


def _payload(event: str, ctx: RequestContext) -> str:
    body: dict[str, object] = {
        "event": event,
        "request_id": ctx.request_id,
        "thread_id": ctx.thread_id,
    }
    if ctx.status:
        body["status"] = ctx.status
    if ctx.total_latency_ms is not None:
        body["total_latency_ms"] = int(round(ctx.total_latency_ms))
    if ctx.error_type:
        body["error_type"] = ctx.error_type
    if ctx.error_message:
        body["error_message"] = ctx.error_message
    if observability_log_question() and ctx.user_question:
        body["user_question"] = ctx.user_question
    return redact_secrets(json.dumps(body, ensure_ascii=True))
