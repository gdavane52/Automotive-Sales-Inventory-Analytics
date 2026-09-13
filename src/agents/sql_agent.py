"""Generate read-only SQLite SQL from a question and schema.

Does not execute SQL. Callers may pass the result to validate_sql / execute_sql.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.llm import get_chat_llm
from src.db.errors import ReadOnlyQueryError
from src.db.sql_guard import (
    assert_readonly_select,
    referenced_qualified_columns,
    referenced_tables,
)
from src.prompts.sql_prompt import (
    AUTOMOTIVE_SCOPE_PROMPT,
    SQL_HUMAN_PROMPT,
    SQL_SYSTEM_PROMPT,
    format_correction_block,
    format_schema_for_prompt,
)

_POSTGRES_ONLY = re.compile(
    r"\bEXTRACT\s*\(|\bDATE_TRUNC\s*\(|::\s*(date|timestamp|int|integer|bigint|numeric|text)\b",
    re.IGNORECASE,
)

_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


class AutomotiveScope(BaseModel):
    """Whether a question is about automotive dealership analytics."""

    in_scope: bool = Field(
        description=(
            "True only if the question is about automobiles, vehicle inventory, "
            "vehicle sales, vehicle checkout, or vehicle trade-ins."
        )
    )
    reason: str = Field(
        description="Short reason, especially when in_scope is false."
    )


class SQLGeneration(BaseModel):
    """Structured LLM output for a single analytics query."""

    sql: str = Field(
        description="A single SQLite SELECT or WITH...SELECT. Empty string if refusing."
    )
    explanation: str = Field(
        description="Short explanation of the SQL, or why the request was refused."
    )
    tables_used: list[str] = Field(
        description="Schema table names used in the SQL. Empty if refusing."
    )


def generate_sql(
    user_question: str,
    database_schema: dict[str, Any],
    previous_sql: str = "",
    validation_error: str = "",
) -> dict[str, Any]:
    """Generate SQLite SQL for ``user_question`` using only ``database_schema``.

    Optional ``previous_sql`` and ``validation_error`` are included so retries
    can correct a failed query. Does not run the query against the database.
    """
    question = (user_question or "").strip()
    if not question:
        return _generation_payload(
            SQLGeneration(
                sql="",
                explanation="A user question is required.",
                tables_used=[],
            ),
            in_scope=False,
        )

    llm = get_chat_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SQL_SYSTEM_PROMPT),
            ("human", SQL_HUMAN_PROMPT),
        ]
    )
    chain = prompt | llm.with_structured_output(SQLGeneration)
    result = chain.invoke(
        {
            "user_question": question,
            "database_schema": format_schema_for_prompt(database_schema),
            "correction_block": format_correction_block(
                previous_sql, validation_error
            ),
        }
    )
    if not isinstance(result, SQLGeneration):
        result = SQLGeneration.model_validate(result)
    return _generation_payload(
        _finalize_generation(result, database_schema),
        in_scope=True,
    )


def classify_automotive_scope(user_question: str) -> AutomotiveScope:
    """Decide if the question is about automotive dealership analytics.

    Uses the meaning of the question, not a keyword denylist.
    """
    llm = get_chat_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AUTOMOTIVE_SCOPE_PROMPT),
            ("human", "Question:\n{user_question}"),
        ]
    )
    chain = prompt | llm.with_structured_output(AutomotiveScope)
    result = chain.invoke({"user_question": user_question})
    if not isinstance(result, AutomotiveScope):
        result = AutomotiveScope.model_validate(result)
    return result


def _generation_payload(result: SQLGeneration, *, in_scope: bool) -> dict[str, Any]:
    payload = result.model_dump()
    payload["in_scope"] = in_scope
    return payload


def _finalize_generation(
    result: SQLGeneration,
    database_schema: dict[str, Any],
) -> SQLGeneration:
    """Enforce read-only SQLite and schema-only identifiers after the LLM returns."""
    sql = _strip_fences(result.sql)
    if not sql:
        return SQLGeneration(
            sql="",
            explanation=result.explanation
            or "No SQL was generated for this request.",
            tables_used=[],
        )

    if _POSTGRES_ONLY.search(sql):
        return SQLGeneration(
            sql="",
            explanation=(
                "Refused: the generated SQL used PostgreSQL-specific syntax. "
                "Use SQLite date()/strftime() instead of EXTRACT, DATE_TRUNC, or :: casts."
            ),
            tables_used=[],
        )

    try:
        sql = assert_readonly_select(sql)
    except ReadOnlyQueryError as exc:
        return SQLGeneration(
            sql="",
            explanation=f"Refused: only read-only SELECT queries are allowed. {exc}",
            tables_used=[],
        )

    known = {name.lower(): name for name in (database_schema.get("table_names") or [])}
    tables_in_sql = referenced_tables(sql)
    unknown_tables = [name for name in tables_in_sql if name.lower() not in known]
    if unknown_tables:
        return SQLGeneration(
            sql="",
            explanation=(
                "Refused: SQL referenced tables that are not in the provided schema: "
                + ", ".join(unknown_tables)
            ),
            tables_used=[],
        )

    table_columns: dict[str, set[str]] = {}
    for name, spec in (database_schema.get("tables") or {}).items():
        table_columns[name.lower()] = {
            col["name"].lower() for col in spec.get("columns") or []
        }
    unknown_cols: list[str] = []
    for table, column in referenced_qualified_columns(sql):
        if table.lower() not in known:
            continue
        if column.lower() not in table_columns.get(table.lower(), set()):
            unknown_cols.append(f"{table}.{column}")
    if unknown_cols:
        return SQLGeneration(
            sql="",
            explanation=(
                "Refused: SQL referenced columns that are not in the provided schema: "
                + ", ".join(unknown_cols)
            ),
            tables_used=[],
        )

    used = []
    seen: set[str] = set()
    for name in result.tables_used or tables_in_sql:
        real = known.get(name.lower())
        if real and real not in seen:
            seen.add(real)
            used.append(real)
    for name in tables_in_sql:
        real = known.get(name.lower())
        if real and real not in seen:
            seen.add(real)
            used.append(real)

    return SQLGeneration(sql=sql, explanation=result.explanation, tables_used=used)


def _strip_fences(sql: str) -> str:
    text = (sql or "").strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    return text.strip().rstrip(";").strip()
