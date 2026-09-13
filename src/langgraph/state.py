"""Shared state for the LangGraph analytics workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class AnalyticsState(TypedDict, total=False):
    """Graph state passed between SQL generation, validation, and execution."""

    user_question: str
    in_scope: bool
    schema: str

    sql: str
    sql_explanation: str
    tables_used: list[str]

    validation_result: bool
    validation_error: str

    query_result: Any

    final_answer: str

    chart: Any
    chart_type: str | None

    business_insights: list[str]

    retry_count: int
