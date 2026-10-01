"""Streamlit presentation layer for the automotive analytics LangGraph workflow.

Does not generate SQL, execute SQL, or open database connections.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.stream_sink import reset_token_handler, set_token_handler
from src.config.logging import get_logger, setup_logging
from src.config.safety import (
    EMPTY_RESULT,
    GENERIC_FAILURE,
    INVALID_SQL,
    OUT_OF_SCOPE,
    SQLITE_FAILED,
)
from src.langgraph.graph import stream_analytics_question

APP_TITLE = "🚗 Automotive Sales & Inventory Analytics"

EXAMPLE_QUESTIONS = [
    "Which are the top 10 selling car models in Pune in 2026?",
    "Which models have the highest inventory in Pune?",
]

_SQL_GENERATION_FAILED = (
    "We could not generate a valid analytics query for this question. "
    "Please rephrase it and try again."
)
_NO_CHART = "A chart could not be generated from this result."
_MAX_HISTORY_TURNS = 5
_MAX_HISTORY_ANSWER_CHARS = 300
logger = get_logger("ui")

# Shown after a node finishes, while the next step is running.
_STATUS_AFTER_NODE = {
    "validate_question": "Preparing the analysis…",
    "generate_sql": "Preparing the analysis…",
    "validate_sql": "Loading results…",
    "execute_sql": "Writing the answer…",
    "generate_answer": "Building the chart…",
    "generate_chart": "Writing business insights…",
}


def _use_example(question: str) -> None:
    st.session_state["_pending_question"] = question


def _chat_title(turns: list[dict]) -> str:
    for turn in turns:
        question = (turn.get("question") or "").strip()
        if question:
            return question
    return "Conversation"


def _remember_current_chat() -> None:
    """Keep one history entry per conversation, not per question."""
    turns = st.session_state.turns
    if not turns:
        return
    thread_id = st.session_state.thread_id
    title = _chat_title(turns)
    updated = (turns[-1].get("timestamp") or "").strip() or datetime.now().strftime("%H:%M")
    chats: list[dict] = st.session_state.chats
    for index, chat in enumerate(chats):
        if chat.get("thread_id") == thread_id:
            chat["title"] = title
            chat["updated"] = updated
            chat["turns"] = turns
            if index:
                chats.insert(0, chats.pop(index))
            return
    chats.insert(
        0,
        {
            "thread_id": thread_id,
            "title": title,
            "updated": updated,
            "turns": turns,
        },
    )


def _new_chat() -> None:
    _remember_current_chat()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.turns = []


def _open_chat(thread_id: str) -> None:
    _remember_current_chat()
    for chat in st.session_state.chats:
        if chat.get("thread_id") != thread_id:
            continue
        st.session_state.thread_id = thread_id
        st.session_state.turns = chat.get("turns") or []
        return


def _clear_chat() -> None:
    thread_id = st.session_state.thread_id
    st.session_state.chats = [
        chat for chat in st.session_state.chats if chat.get("thread_id") != thread_id
    ]
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.turns = []


def _ensure_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "chats" not in st.session_state:
        st.session_state.chats = []
    _remember_current_chat()


def main() -> None:
    setup_logging()
    st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="wide")
    _inject_styles()
    _ensure_session_state()

    st.title(APP_TITLE)
    st.caption(
        "Ask a question in plain English. Analysis uses live vehicle stock, "
        "sales, checkout, and trade-in data."
    )

    _render_sidebar()
    _render_conversation()

    pending = st.session_state.pop("_pending_question", None)
    chat_question = st.chat_input("Ask about sales, inventory, checkout, or trade-ins…")
    question = (pending or chat_question or "").strip()

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            result = _stream_analysis(
                question,
                chat_history=_chat_history_for_graph(st.session_state.turns),
                thread_id=st.session_state.thread_id,
            )
        turn = _turn_from_result(question, result)
    except Exception:
        logger.exception("ui_analysis_failed")
        turn = _failed_turn(question, GENERIC_FAILURE)
        with st.chat_message("assistant"):
            st.warning(GENERIC_FAILURE)

    st.session_state.turns.append(turn)
    _remember_current_chat()
    st.rerun()


def _render_conversation() -> None:
    for index, turn in enumerate(st.session_state.turns):
        _render_turn(turn, turn_index=index)


def _render_sidebar() -> None:
    with st.sidebar:
        st.button(
            "➕ New Chat",
            key="new_chat",
            on_click=_new_chat,
            use_container_width=True,
            type="primary",
        )
        st.markdown("### 💬 Conversation History")
        chats = st.session_state.chats
        active_id = st.session_state.thread_id
        if not chats:
            st.caption("No conversations yet.")
        else:
            for chat in chats:
                title = (chat.get("title") or "Conversation").strip()
                label = title if len(title) <= 64 else f"{title[:61]}…"
                count = len(chat.get("turns") or [])
                stamp = chat.get("updated") or ""
                button_label = label
                if count > 1:
                    button_label = f"{button_label} · {count}"
                if stamp:
                    button_label = f"{button_label} · {stamp}"
                st.button(
                    button_label,
                    key=f"history_chat_{chat.get('thread_id')}",
                    on_click=_open_chat,
                    args=(chat.get("thread_id"),),
                    type="primary" if chat.get("thread_id") == active_id else "secondary",
                    use_container_width=True,
                )

        st.button(
            "🗑️ Clear Chat",
            key="clear_chat",
            on_click=_clear_chat,
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**Example questions**")
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            st.button(
                example,
                key=f"example_{index}",
                on_click=_use_example,
                args=(example,),
                use_container_width=True,
            )


def _render_turn(turn: dict, *, turn_index: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn.get("question") or "")

    with st.chat_message("assistant"):
        error = (turn.get("error") or "").strip()
        if error:
            st.warning(error)

        answer = (turn.get("answer") or "").strip()
        if answer:
            st.subheader("Answer")
            st.markdown(answer)

        insights = turn.get("insights")
        if insights:
            st.subheader("Business Insight")
            for item in insights:
                st.markdown(f"- {item}")

        frame = turn.get("table")
        if frame is not None:
            st.subheader("Data table")
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                st.dataframe(frame, use_container_width=True, hide_index=True)
            elif isinstance(frame, pd.DataFrame) and frame.empty:
                st.write(EMPTY_RESULT)
            else:
                st.write("No table to display for this result.")

        chart = turn.get("chart")
        if chart is not None:
            st.subheader("Chart")
            st.plotly_chart(
                chart,
                use_container_width=True,
                key=f"history_chart_{turn_index}",
            )
        elif isinstance(frame, pd.DataFrame) and not frame.empty:
            st.subheader("Chart")
            st.info(_NO_CHART)

        sql = (turn.get("sql") or "").strip()
        if sql:
            with st.expander("SQL Query", expanded=False):
                st.code(sql, language="sql")


def _chat_history_for_graph(turns: list[dict]) -> list[dict[str, str]]:
    """Build compact prior Q&A for LangGraph follow-up context."""
    history: list[dict[str, str]] = []
    for turn in turns[-_MAX_HISTORY_TURNS:]:
        question = (turn.get("question") or "").strip()
        if not question:
            continue
        answer = (turn.get("answer") or "").strip()
        if not answer:
            answer = (turn.get("error") or "").strip()
        if len(answer) > _MAX_HISTORY_ANSWER_CHARS:
            answer = f"{answer[:_MAX_HISTORY_ANSWER_CHARS - 1]}…"
        history.append({"question": question, "answer": answer})
    return history


def _stream_analysis(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
    thread_id: str | None = None,
) -> dict:
    merged: dict = {
        "user_question": question,
        "retry_count": 0,
        "chat_history": list(chat_history or []),
    }
    buffers = {"answer": "", "insight": ""}
    shown = {"table": False, "chart": False, "sql": False}

    status = st.status("Checking the question…", expanded=False)
    banner_ph = st.empty()
    answer_box = st.empty()
    insight_box = st.empty()
    table_box = st.empty()
    chart_box = st.empty()
    sql_box = st.empty()

    def on_token(channel: str, token: str) -> None:
        if channel == "answer":
            buffers["answer"] += token
            _write_text_section(answer_box, "Answer", buffers["answer"])
        elif channel == "insight":
            buffers["insight"] += token
            _write_text_section(insight_box, "Business Insight", buffers["insight"])

    handler_token = set_token_handler(on_token)
    try:
        for update in stream_analytics_question(
            question,
            chat_history=chat_history,
            thread_id=thread_id,
        ):
            if not isinstance(update, dict):
                continue
            for node_name, payload in update.items():
                if isinstance(payload, dict):
                    merged.update(payload)
                if node_name == "generate_insight":
                    status.update(label="Analysis complete", state="complete")
                elif node_name == "validation_failed" or merged.get("in_scope") is False:
                    status.update(label="Analysis complete", state="complete")
                elif node_name in _STATUS_AFTER_NODE:
                    status.update(
                        label=_STATUS_AFTER_NODE[node_name],
                        state="running",
                    )
            _reveal_ready_sections(
                merged,
                banner_ph=banner_ph,
                answer_box=answer_box,
                insight_box=insight_box,
                table_box=table_box,
                chart_box=chart_box,
                sql_box=sql_box,
                streamed_answer=buffers["answer"],
                streamed_insight=buffers["insight"],
                shown=shown,
            )
        status.update(label="Analysis complete", state="complete")
    except Exception:
        status.update(label="Analysis failed", state="error")
        raise
    finally:
        reset_token_handler(handler_token)
    return merged


def _write_text_section(box, title: str, text: str) -> None:
    with box.container():
        st.subheader(title)
        st.markdown(text)


def _reveal_ready_sections(
    result: dict,
    *,
    banner_ph,
    answer_box,
    insight_box,
    table_box,
    chart_box,
    sql_box,
    streamed_answer: str,
    streamed_insight: str,
    shown: dict[str, bool],
) -> None:
    status = _status_from_result(result)
    banner = result.get("_display_error") or _banner_for_status(status)
    if banner:
        if status in {"out_of_scope", "no_data"}:
            banner_ph.info(banner)
        elif status not in {"ok", "pending"}:
            banner_ph.warning(banner)

    if not streamed_answer.strip() and (result.get("final_answer") or "").strip():
        _write_text_section(answer_box, "Answer", result["final_answer"])

    if not streamed_insight.strip() and result.get("business_insights"):
        _write_text_section(
            insight_box,
            "Business Insight",
            "\n".join(f"- {item}" for item in result["business_insights"]),
        )

    if "query_result" in result and not shown["table"]:
        frame = result.get("query_result")
        with table_box.container():
            st.subheader("Data table")
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                st.dataframe(frame, use_container_width=True, hide_index=True)
            elif isinstance(frame, pd.DataFrame) and frame.empty:
                st.write(EMPTY_RESULT)
            else:
                st.write("No table to display for this result.")
        shown["table"] = True

    if "chart" in result and not shown["chart"]:
        with chart_box.container():
            st.subheader("Chart")
            if result.get("chart") is not None:
                st.plotly_chart(
                    result["chart"],
                    use_container_width=True,
                    key=f"live_chart_{id(result)}",
                )
            else:
                st.info(_NO_CHART)
        shown["chart"] = True

    sql = (result.get("sql") or "").strip()
    if (
        sql
        and result.get("validation_result") is True
        and "query_result" in result
        and not shown["sql"]
    ):
        with sql_box.container():
            with st.expander("SQL Query", expanded=False):
                st.code(sql, language="sql")
        shown["sql"] = True


def _turn_from_result(question: str, result: dict) -> dict:
    status = _status_from_result(result)
    banner = result.get("_display_error") or _banner_for_status(status)
    error = None
    if status in {
        "out_of_scope",
        "sql_generation_failed",
        "sql_validation_failed",
        "execution_failed",
    }:
        error = banner or GENERIC_FAILURE
    elif (result.get("_display_error") or "").strip():
        error = result["_display_error"]

    answer = (result.get("final_answer") or "").strip() or None
    insights = result.get("business_insights")
    if not insights:
        insights = None

    sql = (result.get("sql") or "").strip()
    if not sql or result.get("validation_result") is not True:
        sql = None

    table = result.get("query_result") if "query_result" in result else None
    chart = result.get("chart") if result.get("chart") is not None else None

    return {
        "question": question,
        "answer": answer,
        "table": table,
        "chart": chart,
        "insights": insights,
        "sql": sql,
        "error": error,
        "timestamp": datetime.now().strftime("%H:%M"),
    }


def _failed_turn(question: str, message: str) -> dict:
    return {
        "question": question,
        "answer": None,
        "table": None,
        "chart": None,
        "insights": None,
        "sql": None,
        "error": message,
        "timestamp": datetime.now().strftime("%H:%M"),
    }


def _status_from_result(result: dict) -> str:
    if result.get("in_scope") is False:
        return "out_of_scope"
    if result.get("validation_result") is False:
        if (result.get("sql") or "").strip():
            return "sql_validation_failed"
        return "sql_generation_failed"
    if "query_result" not in result:
        return "pending"
    frame = result.get("query_result")
    if frame is None:
        return "execution_failed"
    if isinstance(frame, pd.DataFrame) and frame.empty:
        return "no_data"
    return "ok"


def _banner_for_status(status: str) -> str:
    return {
        "out_of_scope": OUT_OF_SCOPE,
        "sql_generation_failed": _SQL_GENERATION_FAILED,
        "sql_validation_failed": INVALID_SQL,
        "execution_failed": SQLITE_FAILED,
        "no_data": EMPTY_RESULT,
    }.get(status, "")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; max-width: 1100px; }
        h1 { letter-spacing: -0.02em; }
        div[data-testid="stExpander"] { border: 1px solid #e6e9ef; }
        </style>
        """,
        unsafe_allow_html=True,
    )


main()
