"""Generate a business-facing answer from a query result.

Does not generate SQL and does not generate charts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.llm import content_to_text, get_chat_llm
from src.agents.stream_sink import emit_token
from src.prompts.answer_prompt import ANSWER_HUMAN_PROMPT, ANSWER_SYSTEM_PROMPT

_MAX_RESULT_ROWS = 50
_EMPTY_RESULT_MESSAGE = "No matching data was found for this question."


class BusinessAnswer(BaseModel):
    """Structured LLM output for the final user-facing answer."""

    answer: str = Field(description="Concise business answer based only on the query result.")


def generate_business_answer(
    user_question: str,
    query_result: Any,
    sql: str = "",
) -> str:
    """Write a concise answer from ``query_result`` only. Does not run SQL."""
    question = (user_question or "").strip()
    if query_result is None:
        return "No matching data was found because the query did not return a result."

    if isinstance(query_result, pd.DataFrame) and query_result.empty:
        return _EMPTY_RESULT_MESSAGE

    table_text = _format_query_result(query_result)
    if table_text is None:
        return _EMPTY_RESULT_MESSAGE

    llm = get_chat_llm(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", ANSWER_HUMAN_PROMPT),
        ]
    )
    chain = prompt | llm
    payload = {
        "user_question": question or "(no question provided)",
        "sql": (sql or "").strip() or "(not provided)",
        "query_result": table_text,
    }
    collected: list[str] = []
    for chunk in chain.stream(payload):
        token = content_to_text(chunk)
        if not token:
            continue
        collected.append(token)
        emit_token("answer", token)
    answer = "".join(collected).strip()
    return answer or _EMPTY_RESULT_MESSAGE


def _format_query_result(query_result: Any) -> str | None:
    """Render the result as text. Returns None when there is nothing to report."""
    if isinstance(query_result, pd.DataFrame):
        if query_result.empty:
            return None
        frame = query_result.head(_MAX_RESULT_ROWS)
        lines = [
            f"row_count={len(query_result)}",
            f"columns={', '.join(str(c) for c in query_result.columns)}",
        ]
        if len(query_result) > _MAX_RESULT_ROWS:
            lines.append(f"showing_first_rows={_MAX_RESULT_ROWS}")
        lines.append(frame.to_csv(index=False))
        return "\n".join(lines).strip()

    text = str(query_result).strip()
    return text or None
