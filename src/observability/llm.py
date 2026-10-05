"""Callback that records each chat-model call on the active request.

The four application call sites are:

- ``classify_automotive_scope``
- ``generate_sql``
- ``generate_answer`` (streamed)
- ``generate_insight`` (streamed)

Charts do not call a model. LangGraph ``invoke`` / ``stream`` are the workflow,
not model calls.

Token counts come only from provider usage metadata. They are not estimated.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.observability.context import FAILED, SUCCESS
from src.observability.logger import observe_llm_call

_COMPONENT: ContextVar[str | None] = ContextVar("analytics_llm_component", default=None)
_PENDING: ContextVar[dict[str, _PendingLlmCall] | None] = ContextVar(
    "analytics_llm_pending",
    default=None,
)


@dataclass
class _PendingLlmCall:
    call_id: str
    component: str
    model: str | None
    perf_start: float
    start_time: datetime


@contextmanager
def llm_component(name: str) -> Iterator[None]:
    """Name the model call that is about to run. Does not invoke the model."""
    token = _COMPONENT.set(name)
    try:
        yield
    finally:
        _COMPONENT.reset(token)


class LlmObservabilityHandler(BaseCallbackHandler):
    """Record one observation per chat-model start/end pair.

    For streams, LangChain calls ``on_llm_end`` only after the caller has
    pulled the generator to completion, so latency includes stream consumption.
    """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        pending = _PendingLlmCall(
            call_id=str(uuid.uuid4()),
            component=_COMPONENT.get() or "llm",
            model=_model_name(serialized, kwargs.get("invocation_params")),
            perf_start=time.perf_counter(),
            start_time=datetime.now(timezone.utc),
        )
        current = dict(_PENDING.get() or {})
        current[str(run_id)] = pending
        _PENDING.set(current)

    def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        pending = _pop_pending(run_id)
        if pending is None:
            return
        message = _message_from_result(response)
        input_tokens, output_tokens, total_tokens, available = _token_usage(message)
        model = pending.model or _model_from_message(message)
        observe_llm_call(
            call_id=pending.call_id,
            component=pending.component,
            model=model,
            perf_start=pending.perf_start,
            start_time=pending.start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_usage_available=available,
            status=SUCCESS,
        )

    def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        pending = _pop_pending(run_id)
        if pending is None:
            return
        observe_llm_call(
            call_id=pending.call_id,
            component=pending.component,
            model=pending.model,
            perf_start=pending.perf_start,
            start_time=pending.start_time,
            token_usage_available=False,
            status=FAILED,
            error=error,
        )


def _pop_pending(run_id: uuid.UUID) -> _PendingLlmCall | None:
    current = dict(_PENDING.get() or {})
    pending = current.pop(str(run_id), None)
    _PENDING.set(current)
    return pending


def _model_name(serialized: dict[str, Any] | None, invocation_params: Any) -> str | None:
    params = invocation_params if isinstance(invocation_params, dict) else {}
    kwargs = (serialized or {}).get("kwargs") if isinstance(serialized, dict) else None
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    for value in (
        params.get("model"),
        params.get("model_name"),
        kwargs.get("model_name"),
        kwargs.get("model"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _model_from_message(message: Any) -> str | None:
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("model_name", "model"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _message_from_result(response: LLMResult) -> Any:
    generations = response.generations or []
    if not generations or not generations[0]:
        return None
    generation = generations[0][0]
    return getattr(generation, "message", None)


def _token_usage(message: Any) -> tuple[int | None, int | None, int | None, bool]:
    """Read provider usage. Missing counts stay None and are never coerced to 0."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return (
            _optional_count(usage.get("input_tokens")),
            _optional_count(usage.get("output_tokens")),
            _optional_count(usage.get("total_tokens")),
            True,
        )
    metadata = getattr(message, "response_metadata", None)
    token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if isinstance(token_usage, dict) and token_usage:
        return (
            _optional_count(token_usage.get("prompt_tokens", token_usage.get("input_tokens"))),
            _optional_count(
                token_usage.get("completion_tokens", token_usage.get("output_tokens"))
            ),
            _optional_count(token_usage.get("total_tokens")),
            True,
        )
    return None, None, None, False


def _optional_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
