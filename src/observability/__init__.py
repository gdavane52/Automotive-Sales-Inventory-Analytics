"""Request-scoped observability for analytics runs.

Status values
-------------
SUCCESS
    The graph finished and the question was neither out of scope nor rejected
    by SQL validation. An empty result table is still SUCCESS.
REJECTED
    The graph finished without an application failure, but the question was
    out of domain or SQL was still invalid after the allowed attempts
    (including the ``validation_failed`` stop).
FAILED
    The runner caught an exception, validated SQL failed during execution
    (``query_result`` is None), or a stream was closed before completion.
"""

from src.observability.context import (
    FAILED,
    REJECTED,
    SUCCESS,
    RequestContext,
    begin_request,
    classify_completed_request,
    current_request,
    reset_request,
)
from src.observability.logger import complete_request, log_request_started

__all__ = [
    "FAILED",
    "REJECTED",
    "SUCCESS",
    "RequestContext",
    "begin_request",
    "classify_completed_request",
    "complete_request",
    "current_request",
    "log_request_started",
    "reset_request",
]
