"""Optional token sink so the UI can stream LLM output.

Agents do not import Streamlit. The presentation layer registers a handler.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

TokenHandler = Callable[[str, str], None]

_TOKEN_HANDLER: ContextVar[TokenHandler | None] = ContextVar(
    "analytics_token_handler", default=None
)


def set_token_handler(handler: TokenHandler | None):
    """Register a (channel, token) callback for the current run."""
    return _TOKEN_HANDLER.set(handler)


def reset_token_handler(token) -> None:
    _TOKEN_HANDLER.reset(token)


def emit_token(channel: str, token: str) -> None:
    """Send a streamed token to the UI handler, if one is registered."""
    if not token:
        return
    handler = _TOKEN_HANDLER.get()
    if handler is not None:
        handler(channel, token)
