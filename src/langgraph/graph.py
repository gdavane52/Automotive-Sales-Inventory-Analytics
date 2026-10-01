"""Analytics LangGraph: question check → SQL generate/validate/execute → answer."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config.logging import get_logger, setup_logging
from src.config.safety import GENERIC_FAILURE, public_error_message
from src.config.settings import LANGGRAPH_RECURSION_LIMIT
from src.langgraph.nodes import (
    execute_sql,
    generate_answer,
    generate_chart,
    generate_insight,
    generate_sql,
    validate_question,
    validate_sql,
    validation_failed,
)
from src.langgraph.routing import route_after_question, route_after_validation
from src.langgraph.state import AnalyticsState
from src.observability.context import (
    FAILED,
    begin_request,
    classify_completed_request,
    reset_request,
)
from src.observability.logger import complete_request, log_request_started

_GRAPH: CompiledStateGraph | None = None
_RECURSION_LIMIT = LANGGRAPH_RECURSION_LIMIT
logger = get_logger(__name__)


def build_analytics_graph() -> CompiledStateGraph:
    """Compile the workflow with StateGraph, START, END, and conditional routing.

    START
      → validate_question
          ├─ in scope → generate_sql
          └─ out of scope → END
      → generate_sql
      → validate_sql
          ├─ valid → execute_sql → generate_answer → generate_chart → generate_insight → END
          ├─ invalid, retries left → generate_sql
          └─ invalid, 3 attempts used → validation_failed → END
    """
    graph = StateGraph(AnalyticsState)

    graph.add_node("validate_question", validate_question)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_chart", generate_chart)
    graph.add_node("generate_insight", generate_insight)
    graph.add_node("validation_failed", validation_failed)

    graph.add_edge(START, "validate_question")
    graph.add_conditional_edges(
        "validate_question",
        route_after_question,
        {
            "generate_sql": "generate_sql",
            "end": END,
        },
    )
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute_sql": "execute_sql",
            "generate_sql": "generate_sql",
            "validation_failed": "validation_failed",
        },
    )
    graph.add_edge("execute_sql", "generate_answer")
    graph.add_edge("generate_answer", "generate_chart")
    graph.add_edge("generate_chart", "generate_insight")
    graph.add_edge("generate_insight", END)
    graph.add_edge("validation_failed", END)

    return graph.compile()


def get_analytics_graph() -> CompiledStateGraph:
    """Return a cached compiled graph."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_analytics_graph()
    return _GRAPH


def run_analytics_question(
    user_question: str,
    chat_history: list[dict[str, str]] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run the compiled graph for one question and return the final state.

    Generates a new ``request_id``. ``thread_id`` is the caller's conversation
    id and is left unchanged.
    """
    setup_logging()
    ctx = begin_request(thread_id=thread_id, user_question=user_question)
    log_request_started(ctx)
    logger.info(
        "analytics_run_start request_id=%s thread_id=%s",
        ctx.request_id,
        ctx.thread_id or "",
    )
    graph = get_analytics_graph()
    try:
        result = graph.invoke(
            _graph_input(user_question, chat_history, ctx),
            {"recursion_limit": _RECURSION_LIMIT},
        )
        status, error_type, error_message = classify_completed_request(result)
        complete_request(ctx, status, error_type, error_message)
        logger.info(
            "analytics_run_complete request_id=%s in_scope=%s valid_sql=%s status=%s",
            ctx.request_id,
            result.get("in_scope"),
            result.get("validation_result"),
            ctx.status,
        )
        return result
    except Exception as exc:
        logger.exception("analytics_run_failed")
        complete_request(
            ctx,
            FAILED,
            error_type=type(exc).__name__,
            error_message=public_error_message(exc),
        )
        return _failed_state(exc, ctx)
    finally:
        _close_request(ctx)


def stream_analytics_question(
    user_question: str,
    chat_history: list[dict[str, str]] | None = None,
    thread_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield LangGraph node updates as they complete. Does not generate SQL itself.

    Timing covers the full time the caller spends consuming this generator.
    """
    setup_logging()
    ctx = begin_request(thread_id=thread_id, user_question=user_question)
    log_request_started(ctx)
    logger.info(
        "analytics_stream_start request_id=%s thread_id=%s history_turns=%s",
        ctx.request_id,
        ctx.thread_id or "",
        len(chat_history or []),
    )
    graph = get_analytics_graph()
    merged: dict[str, Any] = {}
    try:
        for update in graph.stream(
            _graph_input(user_question, chat_history, ctx),
            {"recursion_limit": _RECURSION_LIMIT},
            stream_mode="updates",
        ):
            if isinstance(update, dict):
                for payload in update.values():
                    if isinstance(payload, dict):
                        merged.update(payload)
            yield update
        status, error_type, error_message = classify_completed_request(merged)
        complete_request(ctx, status, error_type, error_message)
        logger.info(
            "analytics_stream_complete request_id=%s status=%s",
            ctx.request_id,
            ctx.status,
        )
    except Exception as exc:
        logger.exception("analytics_stream_failed")
        complete_request(
            ctx,
            FAILED,
            error_type=type(exc).__name__,
            error_message=public_error_message(exc),
        )
        yield {"validation_failed": _failed_state(exc, ctx)}
    finally:
        _close_request(ctx)


def _graph_input(
    user_question: str,
    chat_history: list[dict[str, str]] | None,
    ctx: Any,
) -> dict[str, Any]:
    return {
        "user_question": user_question,
        "retry_count": 0,
        "chat_history": list(chat_history or []),
        "request_id": ctx.request_id,
        "thread_id": ctx.thread_id or "",
    }


def _close_request(ctx: Any) -> None:
    """Finish an abandoned stream, then drop the request context."""
    if not ctx.finished:
        complete_request(
            ctx,
            FAILED,
            error_type="stream_interrupted",
            error_message="The request stream ended before completion.",
        )
    reset_request(ctx)


def _failed_state(exc: BaseException, ctx: Any | None = None) -> dict[str, Any]:
    message = public_error_message(exc) or GENERIC_FAILURE
    payload: dict[str, Any] = {
        "in_scope": False,
        "validation_result": False,
        "sql": "",
        "final_answer": message,
        "query_result": None,
        "chart": None,
        "chart_type": None,
        "business_insights": [],
        "validation_error": message,
    }
    if ctx is not None:
        payload["request_id"] = ctx.request_id
        payload["thread_id"] = ctx.thread_id or ""
    return payload

