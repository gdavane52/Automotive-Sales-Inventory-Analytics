"""Conditional edges for the analytics LangGraph."""

from __future__ import annotations

from typing import Literal

from src.langgraph.nodes import MAX_SQL_GENERATION_ATTEMPTS
from src.langgraph.state import AnalyticsState

AfterQuestion = Literal["generate_sql", "end"]
AfterValidation = Literal["execute_sql", "generate_sql", "validation_failed"]


def route_after_question(state: AnalyticsState) -> AfterQuestion:
    """Continue to SQL generation only when the question is in automotive scope."""
    if state.get("in_scope"):
        return "generate_sql"
    return "end"


def route_after_validation(state: AnalyticsState) -> AfterValidation:
    """Send valid SQL to execute; otherwise retry generate_sql up to 3 attempts."""
    if state.get("validation_result"):
        return "execute_sql"
    attempts = int(state.get("retry_count") or 0)
    if attempts >= MAX_SQL_GENERATION_ATTEMPTS:
        return "validation_failed"
    return "generate_sql"
