"""Generate 1-3 business insights from a query result.

Does not generate SQL and does not generate charts.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from src.agents.answer_agent import _format_query_result
from src.agents.llm import content_to_text, get_chat_llm
from src.agents.stream_sink import emit_token
from src.prompts.insight_prompt import INSIGHT_HUMAN_PROMPT, INSIGHT_SYSTEM_PROMPT

_NO_INSIGHT = (
    "The query result does not support a meaningful insight because no matching "
    "data was found."
)


class BusinessInsights(BaseModel):
    """Structured LLM output: one to three insights grounded in the result table."""

    insights: list[str] = Field(
        description="1-3 concise business insights based only on the query result."
    )

    @field_validator("insights")
    @classmethod
    def _limit_insights(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if (item or "").strip()]
        if not cleaned:
            return [_NO_INSIGHT]
        return cleaned[:3]


def generate_business_insights(
    user_question: str,
    query_result: Any,
    final_answer: str = "",
) -> list[str]:
    """Return 1-3 insights from ``query_result`` only. Does not run SQL."""
    if query_result is None:
        return [_NO_INSIGHT]
    if isinstance(query_result, pd.DataFrame) and query_result.empty:
        return [_NO_INSIGHT]

    table_text = _format_query_result(query_result)
    if table_text is None:
        return [_NO_INSIGHT]

    llm = get_chat_llm(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INSIGHT_SYSTEM_PROMPT),
            ("human", INSIGHT_HUMAN_PROMPT),
        ]
    )
    chain = prompt | llm
    payload = {
        "user_question": (user_question or "").strip() or "(no question provided)",
        "final_answer": (final_answer or "").strip() or "(none)",
        "query_result": table_text,
    }
    collected: list[str] = []
    for chunk in chain.stream(payload):
        token = content_to_text(chunk)
        if not token:
            continue
        collected.append(token)
        emit_token("insight", token)
    return _insights_from_text("".join(collected))


def _insights_from_text(text: str) -> list[str]:
    bullets: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            bullets.append(stripped[2:].strip())
        else:
            numbered = re.sub(r"^\d+[\.)]\s+", "", stripped)
            if numbered != stripped:
                bullets.append(numbered.strip())
    if bullets:
        return BusinessInsights(insights=bullets).insights
    cleaned = (text or "").strip()
    return [cleaned] if cleaned else [_NO_INSIGHT]
