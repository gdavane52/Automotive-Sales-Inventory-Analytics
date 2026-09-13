"""Errors raised by the database service and agent tools."""


class ReadOnlyQueryError(ValueError):
    """Raised when a query is not a single read-only SELECT."""


class SQLValidationError(ValueError):
    """Raised when generated SQL fails read-only, schema, or syntax checks."""


class SQLExecutionError(RuntimeError):
    """Raised when a validated SELECT fails at the database."""


class SQLTimeoutError(SQLExecutionError):
    """Raised when a query exceeds the configured timeout."""
