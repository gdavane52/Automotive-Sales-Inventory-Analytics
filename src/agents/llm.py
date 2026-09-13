"""Shared OpenAI chat model for agents (temperature 0)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config.settings import OPENAI_MODEL, PROJECT_ROOT


def get_chat_llm(*, streaming: bool = False) -> ChatOpenAI:
    """Return the project ChatOpenAI client. Does not execute SQL."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file in the project root."
        )
    model_name = os.environ.get("OPENAI_MODEL", OPENAI_MODEL)
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        streaming=streaming,
    )


def content_to_text(chunk: object) -> str:
    """Extract visible text from a streamed chat chunk."""
    content = getattr(chunk, "content", chunk)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts)
    return str(content)
