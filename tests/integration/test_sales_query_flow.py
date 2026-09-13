"""End-to-end LangGraph flow for a real automotive ranking question."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from plotly.graph_objects import Figure

from src.agents.sql_agent import AutomotiveScope
from src.config.settings import SQLITE_DB_PATH
from src.db.sql_guard import assert_readonly_select
from src.langgraph.graph import run_analytics_question
from src.langgraph import nodes as graph_nodes

QUESTION = "Which are the top 10 selling car models in Pune in 2026?"

_RANKING_SQL = """
SELECT model, COUNT(*) AS sales_count
FROM sales
WHERE location = 'Pune'
  AND strftime('%Y', sale_date) = '2026'
GROUP BY model
ORDER BY sales_count DESC
LIMIT 10
""".strip()


def _require_demo_db() -> None:
    if not SQLITE_DB_PATH.exists():
        pytest.skip(
            f"SQLite database not found at {SQLITE_DB_PATH}. "
            "Run python scripts/init_db.py first."
        )


def test_top_selling_models_pune_2026_completes_full_workflow() -> None:
    """Question → validate → SQL → validate → SQLite → answer → chart → insight."""
    _require_demo_db()

    def fake_sql(*_args, **_kwargs):
        return {
            "sql": _RANKING_SQL,
            "explanation": "Rank models sold in Pune during 2026.",
            "tables_used": ["sales"],
            "in_scope": True,
        }

    with (
        patch.object(
            graph_nodes,
            "classify_automotive_scope",
            return_value=AutomotiveScope(
                in_scope=True,
                reason="Automotive sales ranking question.",
            ),
        ),
        patch.object(
            graph_nodes,
            "generate_sql_from_agent",
            side_effect=fake_sql,
        ) as sql_spy,
        patch.object(
            graph_nodes,
            "generate_business_answer",
            return_value="The ranking of top selling models in Pune is shown in the result table.",
        ),
        patch.object(
            graph_nodes,
            "generate_business_insights",
            return_value=["A small set of models accounts for most Pune sales in 2026."],
        ),
        patch.object(
            graph_nodes,
            "validate_sql_query",
            wraps=graph_nodes.validate_sql_query,
        ) as validate_spy,
        patch.object(
            graph_nodes,
            "execute_sql_query",
            wraps=graph_nodes.execute_sql_query,
        ) as execute_spy,
    ):
        result = run_analytics_question(QUESTION)

    assert result is not None
    assert result.get("in_scope") is True
    assert result.get("validation_result") is True

    sql = (result.get("sql") or "").strip()
    assert sql
    assert_readonly_select(sql)

    sql_spy.assert_called()
    validate_spy.assert_called()
    execute_spy.assert_called()

    frame = result.get("query_result")
    assert frame is not None
    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty

    assert result.get("final_answer")
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()

    assert result.get("chart") is not None
    assert isinstance(result["chart"], Figure)
    assert result.get("chart_type") in {"bar", "line", "pie"}

    insights = result.get("business_insights")
    assert insights
    assert isinstance(insights, list)
    assert any(str(item).strip() for item in insights)
