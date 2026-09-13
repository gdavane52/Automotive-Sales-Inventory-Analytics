"""User-facing error text. Never include stack traces or secrets."""

from __future__ import annotations

from src.db.errors import ReadOnlyQueryError, SQLExecutionError, SQLTimeoutError

OUT_OF_SCOPE = (
    "This question is outside the automotive sales and inventory domain. "
    "Please ask about vehicles, stock, sales, checkout, or trade-ins."
)
READ_ONLY_ONLY = (
    "Only read-only analytics queries are allowed. "
    "DELETE, UPDATE, DROP, ALTER, INSERT, CREATE, and TRUNCATE are blocked."
)
INVALID_SQL = (
    "The generated query did not pass validation, so it was not run against "
    "the database. Please try rephrasing the question."
)
SQLITE_FAILED = (
    "The database could not run this query. Please try again or rephrase "
    "the question."
)
EMPTY_RESULT = "No matching data was found for this question."
LLM_UNAVAILABLE = (
    "The analysis service is temporarily unavailable. Please try again."
)
LOOP_STOPPED = (
    "The analysis stopped to prevent an infinite loop. Please rephrase "
    "the question."
)
GENERIC_FAILURE = (
    "Something went wrong while running the analysis. Please try again."
)
RETRY_LIMIT = (
    "Could not produce a valid analytics query after 3 attempts. "
    "Please rephrase the question."
)


def public_error_message(exc: BaseException) -> str:
    """Map an internal exception to a safe message for the UI."""
    if isinstance(exc, ReadOnlyQueryError):
        return READ_ONLY_ONLY
    if isinstance(exc, SQLTimeoutError):
        return "The database query took too long and was cancelled."
    if isinstance(exc, (SQLExecutionError, FileNotFoundError)):
        return SQLITE_FAILED
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "recursion" in name or "recursion" in text:
        return LOOP_STOPPED
    if any(
        token in name or token in text
        for token in ("apierror", "authentication", "ratelimit", "openai", "api key")
    ):
        return LLM_UNAVAILABLE
    return GENERIC_FAILURE
