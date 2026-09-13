"""LangGraph nodes that wrap existing SQL generate / validate / execute functions."""

from __future__ import annotations

from typing import Any

from src.agents.answer_agent import generate_business_answer
from src.agents.insight_agent import generate_business_insights
from src.agents.sql_agent import classify_automotive_scope
from src.agents.sql_agent import generate_sql as generate_sql_from_agent
from src.charts.service import generate_chart as build_chart
from src.config.logging import get_logger
from src.config.safety import (
    EMPTY_RESULT,
    GENERIC_FAILURE,
    LLM_UNAVAILABLE,
    OUT_OF_SCOPE,
    RETRY_LIMIT,
    public_error_message,
)
from src.config.settings import MAX_SQL_GENERATION_ATTEMPTS as _MAX_ATTEMPTS
from src.db.errors import ReadOnlyQueryError, SQLExecutionError, SQLTimeoutError
from src.db.service import execute_sql as execute_sql_query
from src.db.service import get_database_schema
from src.db.service import validate_sql as validate_sql_query
from src.langgraph.state import AnalyticsState
from src.prompts.sql_prompt import format_schema_for_prompt


MAX_SQL_GENERATION_ATTEMPTS = _MAX_ATTEMPTS
logger = get_logger(__name__)


def validate_question(state: AnalyticsState) -> dict[str, Any]:
    """Accept only automotive-industry questions. Does not generate SQL."""
    question = (state.get("user_question") or "").strip()
    if not question:
        logger.info("question_rejected reason=empty")
        return {
            "in_scope": False,
            "final_answer": "A user question is required.",
        }
    try:
        scope = classify_automotive_scope(question)
    except Exception as exc:
        logger.exception("question_validation_llm_failed")
        return {
            "in_scope": False,
            "final_answer": public_error_message(exc) or LLM_UNAVAILABLE,
        }
    if not scope.in_scope:
        logger.info("question_rejected reason=out_of_scope")
        return {
            "in_scope": False,
            "final_answer": scope.reason or OUT_OF_SCOPE,
        }
    logger.info("question_accepted")
    return {"in_scope": True}


def generate_sql(state: AnalyticsState) -> dict[str, Any]:
    """Generate read-only SQL from the user question. Does not execute it."""
    question = (state.get("user_question") or "").strip()
    attempt = int(state.get("retry_count") or 0) + 1
    previous_sql = ""
    validation_error = ""
    if attempt > 1:
        previous_sql = state.get("sql") or ""
        validation_error = state.get("validation_error") or ""
        logger.info("sql_generation_retry attempt=%s", attempt)
    try:
        schema = get_database_schema()
        generated = generate_sql_from_agent(
            question,
            schema,
            previous_sql=previous_sql,
            validation_error=validation_error,
        )
    except Exception as exc:
        logger.exception("sql_generation_failed attempt=%s", attempt)
        return {
            "sql": "",
            "sql_explanation": public_error_message(exc),
            "tables_used": [],
            "retry_count": attempt,
            "validation_result": False,
            "validation_error": public_error_message(exc),
        }
    sql = generated.get("sql") or ""
    logger.info(
        "sql_generated attempt=%s has_sql=%s tables=%s",
        attempt,
        bool(sql),
        generated.get("tables_used") or [],
    )
    return {
        "schema": format_schema_for_prompt(schema),
        "sql": sql,
        "sql_explanation": generated.get("explanation") or "",
        "tables_used": generated.get("tables_used") or [],
        "retry_count": attempt,
    }


def validate_sql(state: AnalyticsState) -> dict[str, Any]:
    """Validate generated SQL. Does not execute it."""
    sql = (state.get("sql") or "").strip()
    if not sql:
        reason = (
            state.get("validation_error")
            or state.get("sql_explanation")
            or "No SQL to validate."
        )
        logger.warning("sql_validation_skipped reason=empty_sql")
        return {
            "validation_result": False,
            "validation_error": reason,
        }
    try:
        report = validate_sql_query(sql)
    except Exception as exc:
        logger.exception("sql_validation_error")
        return {
            "validation_result": False,
            "validation_error": public_error_message(exc),
        }
    if report.get("valid"):
        logger.info("sql_validation_passed")
        return {"validation_result": True, "validation_error": ""}
    errors = report.get("errors") or []
    logger.warning("sql_validation_failed errors=%s", errors)
    return {
        "validation_result": False,
        "validation_error": "; ".join(errors) or "SQL failed validation.",
    }


def execute_sql(state: AnalyticsState) -> dict[str, Any]:
    """Run validated SQL on SQLite and store a DataFrame. Skips if invalid."""
    if not state.get("validation_result"):
        return {}
    sql = (state.get("sql") or "").strip()
    try:
        frame = execute_sql_query(sql)
        logger.info("sql_executed rows=%s cols=%s", len(frame), list(frame.columns))
        return {"query_result": frame}
    except (ReadOnlyQueryError, SQLExecutionError, SQLTimeoutError, FileNotFoundError) as exc:
        logger.exception("sql_execution_failed")
        return {
            "query_result": None,
            "validation_error": public_error_message(exc),
            "final_answer": public_error_message(exc),
        }


def generate_answer(state: AnalyticsState) -> dict[str, Any]:
    """Turn query_result into a concise business answer. Does not generate SQL or charts."""
    result = state.get("query_result")
    if result is None and state.get("validation_error"):
        return {
            "final_answer": state.get("final_answer")
            or state.get("validation_error")
            or GENERIC_FAILURE
        }
    try:
        answer = generate_business_answer(
            state.get("user_question") or "",
            result,
            state.get("sql") or "",
        )
        logger.info("answer_generated empty_result=%s", answer == EMPTY_RESULT)
        return {"final_answer": answer}
    except Exception as exc:
        logger.exception("answer_generation_failed")
        return {"final_answer": public_error_message(exc)}


def generate_chart(state: AnalyticsState) -> dict[str, Any]:
    """Build a Plotly chart from query_result. Does not generate SQL or Plotly code."""
    try:
        result = build_chart(
            state.get("query_result"),
            state.get("user_question") or "",
        )
        logger.info("chart_generated type=%s", result.get("chart_type"))
        return {
            "chart": result.get("chart"),
            "chart_type": result.get("chart_type"),
        }
    except Exception:
        logger.exception("chart_generation_failed")
        return {"chart": None, "chart_type": None}


def generate_insight(state: AnalyticsState) -> dict[str, Any]:
    """Write 1-3 business insights from query_result. Does not generate SQL."""
    try:
        insights = generate_business_insights(
            state.get("user_question") or "",
            state.get("query_result"),
            state.get("final_answer") or "",
        )
        logger.info("insights_generated count=%s", len(insights))
        return {"business_insights": insights}
    except Exception as exc:
        logger.exception("insight_generation_failed")
        return {"business_insights": [public_error_message(exc)]}


def validation_failed(state: AnalyticsState) -> dict[str, Any]:
    """Stop after the retry limit with a clear error. Does not execute SQL."""
    attempts = int(state.get("retry_count") or 0)
    logger.warning("sql_retry_limit_reached attempts=%s", attempts)
    return {"final_answer": RETRY_LIMIT, "validation_result": False}
