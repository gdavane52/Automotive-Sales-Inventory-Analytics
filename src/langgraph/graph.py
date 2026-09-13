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


def run_analytics_question(user_question: str) -> dict[str, Any]:
    """Run the compiled graph for one question and return the final state."""
    setup_logging()
    logger.info("analytics_run_start")
    graph = get_analytics_graph()
    try:
        result = graph.invoke(
            {"user_question": user_question, "retry_count": 0},
            {"recursion_limit": _RECURSION_LIMIT},
        )
        logger.info(
            "analytics_run_complete in_scope=%s valid_sql=%s",
            result.get("in_scope"),
            result.get("validation_result"),
        )
        return result
    except Exception as exc:
        logger.exception("analytics_run_failed")
        return _failed_state(exc)


def stream_analytics_question(user_question: str) -> Iterator[dict[str, Any]]:
    """Yield LangGraph node updates as they complete. Does not generate SQL itself."""
    setup_logging()
    logger.info("analytics_stream_start")
    graph = get_analytics_graph()
    try:
        yield from graph.stream(
            {"user_question": user_question, "retry_count": 0},
            {"recursion_limit": _RECURSION_LIMIT},
            stream_mode="updates",
        )
        logger.info("analytics_stream_complete")
    except Exception as exc:
        logger.exception("analytics_stream_failed")
        yield {"validation_failed": _failed_state(exc)}


def _failed_state(exc: BaseException) -> dict[str, Any]:
    message = public_error_message(exc) or GENERIC_FAILURE
    return {
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

