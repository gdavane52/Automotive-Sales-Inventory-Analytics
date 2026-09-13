from src.db.connection import connect, create_readonly_engine
from src.db.loader import initialize_database
from src.db.service import execute_sql, get_database_schema, validate_sql

__all__ = [
    "connect",
    "create_readonly_engine",
    "execute_sql",
    "get_database_schema",
    "initialize_database",
    "question_to_dataframe",
    "question_to_result",
    "validate_and_execute",
    "validate_sql",
]


def __getattr__(name: str):
    if name in {"question_to_dataframe", "question_to_result", "validate_and_execute"}:
        from src.db import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "connect",
    "create_readonly_engine",
    "execute_sql",
    "get_database_schema",
    "initialize_database",
    "question_to_dataframe",
    "question_to_result",
    "validate_and_execute",
    "validate_sql",
]
