"""Streamlit presentation layer for the automotive analytics LangGraph workflow.

Does not generate SQL, execute SQL, or open database connections.
"""

from __future__ import annotations

import sys
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
    "Which brands have the highest trade-in activity?",
    "What is the average trade-in value by brand?",
    "Which models have the highest checkout activity?",
]

_SQL_GENERATION_FAILED = (
    "We could not generate a valid analytics query for this question. "
    "Please rephrase it and try again."
)
_NO_CHART = "A chart could not be generated from this result."
logger = get_logger("ui")

_NODE_STATUS = {
    "validate_question": "Checking the question…",
    "generate_sql": "Preparing the analysis…",
    "validate_sql": "Preparing the analysis…",
    "execute_sql": "Loading results…",
    "generate_answer": "Writing the answer…",
    "generate_chart": "Building the chart…",
    "generate_insight": "Writing business insights…",
    "validation_failed": "The query could not be completed.",
}

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
    st.session_state["_prefill"] = question


def main() -> None:
    setup_logging()
    st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="wide")
    _inject_styles()

    st.title(APP_TITLE)
    st.caption(
        "Ask a question in plain English. Analysis uses live vehicle stock, "
        "sales, checkout, and trade-in data."
    )

    if "_prefill" in st.session_state:
        st.session_state.question_text = st.session_state.pop("_prefill")
    if "question_text" not in st.session_state:
        st.session_state.question_text = ""
    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    st.subheader("Ask a question")
    with st.form("analyze_form", clear_on_submit=False):
        st.text_area(
            "Natural-language question",
            key="question_text",
            height=90,
            placeholder="Example: Which are the top 10 selling car models in Pune in 2026?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    with st.sidebar:
        st.markdown("**Example questions**")
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            st.button(
                example,
                key=f"example_{index}",
                on_click=_use_example,
                args=(example,),
                use_container_width=True,
            )

    if submitted:
        question = (st.session_state.question_text or "").strip()
        if not question:
            st.warning("Please enter a question to analyze.")
        else:
            try:
                st.session_state.analysis = _stream_analysis(question)
            except Exception:
                logger.exception("ui_analysis_failed")
                st.session_state.analysis = {
                    "in_scope": True,
                    "validation_result": False,
                    "sql": "",
                    "final_answer": GENERIC_FAILURE,
                    "query_result": None,
                    "chart": None,
                    "chart_type": None,
                    "business_insights": [],
                    "validation_error": GENERIC_FAILURE,
                    "_display_error": GENERIC_FAILURE,
                }

    elif st.session_state.analysis is not None:
        _render_result(st.session_state.analysis)


def _stream_analysis(question: str) -> dict:
    merged: dict = {"user_question": question, "retry_count": 0}
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
        for update in stream_analytics_question(question):
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
                st.plotly_chart(result["chart"], use_container_width=True)
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


def _render_result(result: dict) -> None:
    status = _status_from_result(result)
    banner = result.get("_display_error") or _banner_for_status(status)
    if banner:
        if status in {"out_of_scope", "no_data", "no_chart_only"}:
            st.info(banner)
        else:
            st.warning(banner)

    st.markdown("---")
    st.subheader("Answer")
    answer = (result.get("final_answer") or "").strip()
    if answer:
        st.write(answer)
    elif status == "out_of_scope":
        st.write(OUT_OF_SCOPE)
    else:
        st.write("No answer is available for this question.")

    st.subheader("Business Insight")
    insights = result.get("business_insights") or []
    if status in {"out_of_scope", "sql_generation_failed", "sql_validation_failed", "execution_failed"}:
        st.write("No business insight is available because the analysis did not complete.")
    elif not insights:
        st.write("The data does not support a meaningful insight.")
    else:
        for item in insights:
            st.markdown(f"- {item}")

    st.subheader("Data table")
    frame = result.get("query_result")
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    elif status == "no_data":
        st.write(EMPTY_RESULT)
    else:
        st.write("No table to display for this result.")

    st.subheader("Chart")
    chart = result.get("chart")
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)
    elif isinstance(frame, pd.DataFrame) and not frame.empty:
        st.info(_NO_CHART)
    else:
        st.write("No chart to display for this result.")

    sql = (result.get("sql") or "").strip()
    with st.expander("SQL Query", expanded=False):
        if sql:
            st.code(sql, language="sql")
        else:
            st.write("No SQL query was produced for this question.")


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
