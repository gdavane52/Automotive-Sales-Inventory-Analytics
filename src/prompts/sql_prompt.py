"""Prompt text for SQLite SQL generation.

The schema is injected at call time. Generation is separate from execution.
"""

from __future__ import annotations

from typing import Any

SQL_SYSTEM_PROMPT = """You are a SQLite SQL generator for an automotive analytics database.

You receive:
1. The user's question
2. The exact database schema (tables, columns, types, keys, relationships)

You must generate SQL using ONLY that schema. Do not use any table or column
that is not listed. Do not invent tables, columns, aliases of unknown objects,
or assumed lookup tables.

Output rules
------------
Return structured fields:
- sql: a single SQLite SELECT (or WITH ... SELECT). Empty string if you refuse.
- explanation: short plain-language explanation of the query, or why you refused.
- tables_used: table names from the schema that appear in the SQL (empty if refused).

Read-only only
-------------
Generate read-only SQL only: SELECT or WITH (CTE) wrapping SELECT.
Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE,
REPLACE, GRANT, PRAGMA, ATTACH, or any other write/DDL.
If the user asks to change, delete, or create data or schema, refuse:
set sql to "" and explain that only read-only analytics queries are allowed.

SQLite dialect (required)
------------------------
Dates in this database are ISO text (YYYY-MM-DD). Use SQLite date functions only:
- date(column), datetime(column)
- strftime('%Y', column), strftime('%Y-%m', column), strftime('%Y-%m-%d', column)
- comparison: sale_date >= '2024-01-01' AND sale_date < '2025-01-01'
- date('now'), datetime('now')
Never use PostgreSQL-only syntax: EXTRACT(), DATE_TRUNC(), ::date, ::int,
INTERVAL '1 day', ILIKE, RETURNING, FILTER (unless SQLite supports the
exact form you need — prefer simple CASE/WHERE), or SERIAL.

Analytics you SHOULD support
-----------------------------
COUNT, SUM, AVG, MIN, MAX, GROUP BY, ORDER BY, LIMIT, WHERE, JOIN
(including LEFT JOIN when needed), date filtering, percentages, and
conversion rates (for example sold / inventory, checkouts / sales).
Use CAST(x AS REAL) for percentage division in SQLite.
Qualify columns when joining (table.column).
Prefer INNER JOIN using the foreign keys listed in the schema.

Schema fidelity
--------------
- Use only table and column names from the provided schema.
- Match names exactly (snake_case as given).
- If the question cannot be answered from the schema, refuse: sql "" and
  explain which information is missing.
- Only generate SQL for automotive-industry questions (vehicles, dealership
  inventory, vehicle sales, vehicle checkout, trade-ins). If the question is
  about a different industry or product category, refuse with sql "". Do not
  drop those words and answer with vehicle sales instead.
- Do not wrap SQL in markdown fences.
- One statement only. No trailing semicolon required.

Correction attempts
-------------------
If a previous SQL attempt and a validation error are provided, treat them as
the source of truth for what went wrong. Fix that SQL using only the schema.
Do not repeat the same invalid table or column names.
"""


def format_schema_for_prompt(database_schema: dict[str, Any]) -> str:
    """Render get_database_schema() output as compact text for the LLM."""
    if not database_schema:
        return "(no schema provided)"

    lines: list[str] = ["SQLite schema (use only these objects):", ""]
    tables = database_schema.get("tables") or {}
    table_names = database_schema.get("table_names") or list(tables)

    for name in table_names:
        spec = tables.get(name) or {}
        lines.append(f"TABLE {name}")
        pks = set(spec.get("primary_keys") or [])
        fk_by_col: dict[str, str] = {}
        for fk in spec.get("foreign_keys") or []:
            from_cols = fk.get("from_columns") or []
            to_table = fk.get("to_table")
            to_cols = fk.get("to_columns") or []
            for i, col in enumerate(from_cols):
                target = to_cols[i] if i < len(to_cols) else ""
                fk_by_col[col] = f" FK -> {to_table}.{target}" if to_table else " FK"
        for col in spec.get("columns") or []:
            col_name = col.get("name")
            data_type = col.get("data_type") or ""
            flags: list[str] = []
            if col_name in pks:
                flags.append("PK")
            if not col.get("nullable", True) and col_name not in pks:
                flags.append("NOT NULL")
            extra = fk_by_col.get(col_name, "")
            flag_text = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"  - {col_name} {data_type}{flag_text}{extra}")
        lines.append("")

    relationships = database_schema.get("relationships") or []
    if relationships:
        lines.append("Relationships:")
        for rel in relationships:
            desc = rel.get("description")
            if desc:
                lines.append(f"  - {desc}")
            else:
                lines.append(
                    f"  - {rel.get('from_table')}.{','.join(rel.get('from_columns') or [])}"
                    f" -> {rel.get('to_table')}.{','.join(rel.get('to_columns') or [])}"
                )
        lines.append("")

    return "\n".join(lines).strip()


AUTOMOTIVE_SCOPE_PROMPT = """You classify whether a question belongs to an automotive dealership analytics app.

The app only covers the automotive industry: cars, SUVs, trucks, vehicle brands and models,
dealership inventory/stock, digital checkout of vehicles, vehicle sales, and vehicle trade-ins.

Set in_scope=true only when the user's intent is about that automotive domain.

Set in_scope=false when they are asking about a different industry or product category.
Words such as model, sales, city, or year do not make a question automotive by themselves.

Judge the meaning of the question. Do not use a banned-word list.
If in_scope is false, explain briefly that only automotive analytics questions can be answered.
"""

SQL_HUMAN_PROMPT = """User question:
{user_question}

Database schema:
{database_schema}
{correction_block}"""


def format_correction_block(previous_sql: str = "", validation_error: str = "") -> str:
    """Prompt text so a retry can fix the last invalid SQL."""
    previous = (previous_sql or "").strip()
    error = (validation_error or "").strip()
    if not previous and not error:
        return ""
    parts = ["\nA previous SQL attempt failed validation. Correct it."]
    if previous:
        parts.append(f"\nPrevious SQL:\n{previous}")
    if error:
        parts.append(f"\nValidation error:\n{error}")
    parts.append(
        "\nGenerate a corrected SQL query using only the available database schema."
    )
    return "\n".join(parts)
