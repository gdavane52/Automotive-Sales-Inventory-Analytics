"""Application logging that redacts secrets and never prints stack traces to users."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

_CONFIGURED = False
_SECRET_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_ASSIGNED_SECRET = re.compile(
    r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([^\s'\"]+)"
)


def setup_logging(level: int | None = None) -> None:
    """Configure process-wide logging once. Safe to call from app and graph."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_level = level
    if log_level is None:
        name = (os.environ.get("LOG_LEVEL") or "INFO").upper()
        log_level = getattr(logging, name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler.addFilter(_RedactSecretsFilter())
    root = logging.getLogger("automotive_analytics")
    root.setLevel(log_level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if name.startswith("automotive_analytics"):
        return logging.getLogger(name)
    return logging.getLogger(f"automotive_analytics.{name}")


def redact_secrets(value: str) -> str:
    """Remove API keys and other secret values from a log string."""
    text = value or ""
    for secret in _secret_values():
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _ASSIGNED_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return text


def _secret_values() -> list[str]:
    values: list[str] = []
    for key, raw in os.environ.items():
        if not raw or not _SECRET_NAME.search(key):
            continue
        values.append(raw)
    values.sort(key=len, reverse=True)
    return values


class _RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact_secrets(str(val)) if isinstance(val, str) else val
                    for key, val in record.args.items()
                }
            else:
                redacted: list[Any] = []
                for arg in record.args:
                    redacted.append(
                        redact_secrets(str(arg)) if isinstance(arg, str) else arg
                    )
                record.args = tuple(redacted)
        if record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)
        return True
