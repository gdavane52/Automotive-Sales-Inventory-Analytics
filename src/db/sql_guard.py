"""Reject write/DDL SQL before it reaches the database."""

from __future__ import annotations

import re

from src.db.errors import ReadOnlyQueryError

_ALLOWED_LEADERS = frozenset({"SELECT", "WITH"})

_COMMENT_OR_LITERAL = re.compile(
    r"('(?:''|[^'])*')|(\"(?:[^\"]*)\")|(--[^\n]*)|(/\*[\s\S]*?\*/)",
    re.MULTILINE,
)

_LEADING_KEYWORD = re.compile(r"^[(\s]*([A-Za-z_]+)")

# Statement-level writes and schema changes, including common SQLite verbs.
_DESTRUCTIVE = re.compile(
    r"""
    \b(
        INSERT\s+INTO
        | REPLACE\s+INTO
        | UPSERT\b
        | UPDATE\s+(?!JOIN\b)[A-Za-z_][\w.]*
        | DELETE\s+FROM
        | DROP\s+
        | ALTER\s+
        | TRUNCATE\s+
        | CREATE\s+
        | GRANT\s+
        | REVOKE\s+
        | ATTACH\s+
        | DETACH\s+
        | VACUUM\b
        | REINDEX\b
        | PRAGMA\b
        | MERGE\s+
        | BEGIN\b
        | COMMIT\b
        | ROLLBACK\b
        | SAVEPOINT\b
        | LOAD\s+
        | COPY\s+
        | IMPORT\s+
        | EXEC(?:UTE)?\b
        | CALL\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FORBIDDEN_LEADERS = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "REPLACE",
        "GRANT",
        "REVOKE",
        "ATTACH",
        "DETACH",
        "VACUUM",
        "REINDEX",
        "ANALYZE",
        "PRAGMA",
        "MERGE",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "EXPLAIN",
        "VALUES",
    }
)


def assert_readonly_select(query: str) -> str:
    """Return a stripped SELECT/CTE query, or raise ReadOnlyQueryError."""
    if query is None or not str(query).strip():
        raise ReadOnlyQueryError("Query is empty. Pass a single SELECT statement.")

    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        raise ReadOnlyQueryError("Query is empty. Pass a single SELECT statement.")

    sanitized = _COMMENT_OR_LITERAL.sub(" ", query)
    statements = [part.strip() for part in sanitized.split(";") if part.strip()]
    if not statements:
        raise ReadOnlyQueryError("Query is empty after removing comments.")
    if len(statements) > 1:
        raise ReadOnlyQueryError(
            "Multiple SQL statements are not allowed. Submit one SELECT query."
        )

    leader = _leading_keyword(statements[0])
    if leader in _FORBIDDEN_LEADERS or leader not in _ALLOWED_LEADERS:
        raise ReadOnlyQueryError(
            f"Only read-only SELECT queries are allowed. Rejected statement type: {leader or 'unknown'}."
        )

    destructive = _DESTRUCTIVE.search(sanitized)
    if destructive:
        raise ReadOnlyQueryError(
            "Query contains a disallowed write or schema operation "
            f"({destructive.group(1).split()[0].upper()})."
        )

    return stripped


def _leading_keyword(statement: str) -> str:
    match = _LEADING_KEYWORD.match(statement)
    return match.group(1).upper() if match else ""


_SQL_KEYWORDS = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "LEFT",
        "RIGHT",
        "INNER",
        "OUTER",
        "FULL",
        "CROSS",
        "NATURAL",
        "ON",
        "AND",
        "OR",
        "NOT",
        "AS",
        "GROUP",
        "BY",
        "ORDER",
        "LIMIT",
        "OFFSET",
        "HAVING",
        "UNION",
        "ALL",
        "DISTINCT",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "IN",
        "IS",
        "NULL",
        "LIKE",
        "BETWEEN",
        "EXISTS",
        "WITH",
        "RECURSIVE",
        "ASC",
        "DESC",
        "USING",
        "CAST",
        "TRUE",
        "FALSE",
        "OVER",
        "PARTITION",
        "WINDOW",
        "FILTER",
    }
)

_JOIN_PREFIXES = frozenset(
    {"INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL", "OUTER"}
)

_FROM_JOIN = re.compile(
    r"\b(?:FROM|JOIN)\s+(?!INNER\b|LEFT\b|RIGHT\b|FULL\b|CROSS\b|OUTER\b|NATURAL\b)"
    r"(?:([A-Za-z_][\w]*)\s*\.\s*)?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)

_CTE_AS = re.compile(r"\b([A-Za-z_][\w]*)\s+AS\s*\(", re.IGNORECASE)

_QUALIFIED_COLUMN = re.compile(
    r"\b([A-Za-z_][\w]*)\s*\.\s*([A-Za-z_][\w]*)\b"
)


def strip_comments_and_literals(query: str) -> str:
    return _COMMENT_OR_LITERAL.sub(" ", query)


def referenced_tables(query: str) -> list[str]:
    """Physical table names from FROM/JOIN, excluding CTE names where possible."""
    sanitized = strip_comments_and_literals(query)
    cte = {match.group(1).lower() for match in _CTE_AS.finditer(sanitized)}
    tables: list[str] = []
    seen: set[str] = set()
    for match in _FROM_JOIN.finditer(sanitized):
        name = match.group(2)
        if name.upper() in _JOIN_PREFIXES:
            continue
        key = name.lower()
        if key in cte or key in seen:
            continue
        seen.add(key)
        tables.append(name)
    return tables


_CURRENT_STOCK = re.compile(
    r"""
    current(?:ly)?\s+(?:available\s+)?(?:stock|inventory)
    | available\s+stock
    | currently\s+available
    | available\s+for\s+sale
    | remaining\s+(?:stock|inventory)
    | \bin\s+stock\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_NOT_CURRENT_STOCK = re.compile(
    r"""
    \breserved\b
    | \bsold\b
    | present\s+in
    | vehicle[_\s]stock\s+table
    | total\s+(?:historical\s+)?records
    """,
    re.IGNORECASE | re.VERBOSE,
)
_IN_STOCK_FILTER = re.compile(
    r"""
    (?:\b[\w`]+\s*\.\s*)?
    [`"']?stock_status[`"']?
    \s*=\s*
    (['"])In\s+Stock\1
    """,
    re.IGNORECASE | re.VERBOSE,
)
_CURRENT_STOCK_ERROR = (
    "Current stock must be read from vehicle_stock with "
    "stock_status = 'In Stock'. Reserved and Sold vehicles are not available inventory."
)


def asks_current_stock(question: str) -> bool:
    """True when the question means available inventory, not all stock rows."""
    text = question or ""
    if not _CURRENT_STOCK.search(text):
        return False
    if _NOT_CURRENT_STOCK.search(text) and not re.search(
        r"current(?:ly)?\s+(?:available\s+)?(?:stock|inventory)|available\s+stock|"
        r"currently\s+available|remaining\s+(?:stock|inventory)",
        text,
        re.IGNORECASE,
    ):
        return False
    return True


def has_in_stock_filter(sql: str) -> bool:
    """True when SQL filters stock_status to In Stock, ignoring spacing and qualifiers."""
    return _IN_STOCK_FILTER.search(sql or "") is not None


def current_stock_filter_error(question: str, sql: str) -> str | None:
    """Reject current-stock SQL that does not filter stock_status = 'In Stock'.

    Returns None for other questions, including reserved, sold, and explicit
    total-record questions. Does not rewrite the SQL.
    """
    if not asks_current_stock(question):
        return None
    if "vehicle_stock" not in {name.lower() for name in referenced_tables(sql or "")}:
        return _CURRENT_STOCK_ERROR
    if not has_in_stock_filter(sql or ""):
        return _CURRENT_STOCK_ERROR
    return None


def referenced_qualified_columns(query: str) -> list[tuple[str, str]]:
    """Return (table, column) pairs from table.column references."""
    sanitized = strip_comments_and_literals(query)
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in _QUALIFIED_COLUMN.finditer(sanitized):
        table, column = match.group(1), match.group(2)
        if table.upper() in _SQL_KEYWORDS or column.upper() in _SQL_KEYWORDS:
            continue
        key = (table.lower(), column.lower())
        if key in seen:
            continue
        seen.add(key)
        found.append((table, column))
    return found
