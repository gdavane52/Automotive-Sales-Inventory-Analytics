"""Out-of-scope questions must stop before SQL generation and SQLite."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.sql_agent import AutomotiveScope, classify_automotive_scope
from src.config.safety import OUT_OF_SCOPE
from src.langgraph.graph import run_analytics_question
from src.langgraph import nodes as graph_nodes

QUESTION = "Which are the top 10 selling chocolates in Pune in 2026?"


def _openai_configured() -> bool:
    from src.agents.llm import get_chat_llm

    try:
        get_chat_llm()
    except RuntimeError:
        return False
    return True


def test_chocolate_question_is_rejected_before_sql_and_sqlite() -> None:
    """Domain rejection must skip SQL generation and never query SQLite."""
    with (
        patch.object(
            graph_nodes,
            "classify_automotive_scope",
            return_value=AutomotiveScope(
                in_scope=False,
                reason=OUT_OF_SCOPE,
            ),
        ) as classify_spy,
        patch.object(graph_nodes, "generate_sql_from_agent") as sql_spy,
        patch.object(graph_nodes, "validate_sql_query") as validate_spy,
        patch.object(graph_nodes, "execute_sql_query") as execute_spy,
        patch.object(graph_nodes, "generate_business_answer") as answer_spy,
        patch.object(graph_nodes, "generate_business_insights") as insight_spy,
        patch.object(graph_nodes, "build_chart") as chart_spy,
    ):
        result = run_analytics_question(QUESTION)

    classify_spy.assert_called_once()
    sql_spy.assert_not_called()
    validate_spy.assert_not_called()
    execute_spy.assert_not_called()
    answer_spy.assert_not_called()
    insight_spy.assert_not_called()
    chart_spy.assert_not_called()

    assert result is not None
    assert result.get("in_scope") is False
    assert not (result.get("sql") or "").strip()
    assert result.get("query_result") is None
    assert result.get("chart") is None
    assert not result.get("business_insights")
    answer = (result.get("final_answer") or "").strip()
    assert answer
    assert "traceback" not in answer.lower()
    assert any(
        token in answer.lower()
        for token in ("automotive", "outside", "vehicle", "stock", "sales")
    )


@pytest.mark.skipif(not _openai_configured(), reason="OPENAI_API_KEY is not set")
def test_chocolate_question_classifier_marks_out_of_scope() -> None:
    """Live domain classifier must not treat chocolate sales as automotive."""
    scope = classify_automotive_scope(QUESTION)
    assert scope.in_scope is False
    assert (scope.reason or "").strip()
