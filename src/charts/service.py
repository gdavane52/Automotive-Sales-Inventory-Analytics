"""Deterministic Plotly charts from a SQL result DataFrame.

Does not call an LLM and does not generate Plotly/Python source code.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

_ID_NAME = re.compile(r"(^id$|_id$)", re.IGNORECASE)
_DATE_NAME = re.compile(r"(date|time|month|week|day|period)", re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_YEAR_NAME = re.compile(r"year", re.IGNORECASE)
_PCT_NAME = re.compile(r"(percent|pct|share|proportion|rate)", re.IGNORECASE)
_TOP_HINT = re.compile(
    r"\b(top|rank|ranking|highest|lowest|most|least)\b", re.IGNORECASE
)
_PIE_HINT = re.compile(
    r"\b(share|percentage|percent|proportion|mix|breakdown|composition|"
    r"distribution|split)\b",
    re.IGNORECASE,
)

_MAX_BAR_CATEGORIES = 30
_MAX_PIE_SLICES = 8
_MAX_LINE_POINTS = 2000
_MIN_ROWS = 2


def generate_chart(
    dataframe: pd.DataFrame | None,
    user_question: str = "",
) -> dict[str, Any]:
    """Build a Plotly figure from ``dataframe``, or return no chart.

    Output: ``chart`` (Figure or None) and ``chart_type`` (bar/line/pie or None).
    """
    question = (user_question or "").strip()
    if dataframe is None or not isinstance(dataframe, pd.DataFrame):
        return _none()
    if dataframe.empty or len(dataframe) < _MIN_ROWS:
        return _none()

    work = dataframe.copy()
    dates, categories, numerics = _classify_columns(work)
    if not numerics:
        return _none()

    y_col = numerics[0]
    title = question or f"{y_col}"

    if dates:
        return _line_chart(work, dates[0], y_col, title)

    if categories:
        return _category_chart(work, categories[0], y_col, title, question)

    return _none()


def _none() -> dict[str, Any]:
    return {"chart": None, "chart_type": None}


def _classify_columns(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    dates: list[str] = []
    categories: list[str] = []
    numerics: list[str] = []
    for name in frame.columns:
        series = frame[name]
        label = str(name)
        if _ID_NAME.search(label):
            nunique = int(series.nunique(dropna=True))
            if _MIN_ROWS <= nunique <= _MAX_BAR_CATEGORIES:
                categories.append(label)
            continue
        if _is_date_column(label, series):
            dates.append(label)
            continue
        if _is_numeric_column(series):
            numerics.append(label)
            continue
        if _is_category_column(series):
            categories.append(label)
    return dates, categories, numerics


def _is_date_column(name: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if _YEAR_NAME.search(name) and pd.api.types.is_numeric_dtype(series):
        years = pd.to_numeric(series, errors="coerce").dropna()
        return bool(len(years) and years.between(1990, 2100).all())
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    text = series.astype(str).str.strip()
    if len(text) and float(text.str.match(_ISO_DATE).mean()) >= 0.8:
        return True
    if _DATE_NAME.search(name):
        parsed = pd.to_datetime(series, errors="coerce")
        sample = parsed.dropna()
        return len(sample) >= _MIN_ROWS and int(sample.nunique()) >= _MIN_ROWS
    return False


def _is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean()) >= 0.8


def _is_category_column(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    nunique = int(series.nunique(dropna=True))
    if nunique < _MIN_ROWS:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    return str(series.dtype) == "category"


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _line_chart(
    frame: pd.DataFrame, x_col: str, y_col: str, title: str
) -> dict[str, Any]:
    work = frame[[x_col, y_col]].copy()
    if _YEAR_NAME.search(str(x_col)) and not pd.api.types.is_datetime64_any_dtype(
        work[x_col]
    ):
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    else:
        work[x_col] = pd.to_datetime(work[x_col], errors="coerce")
    work[y_col] = _numeric_series(work, y_col)
    work = work.dropna(subset=[x_col, y_col])
    if len(work) < _MIN_ROWS:
        return _none()
    work = work.groupby(x_col, as_index=False)[y_col].sum()
    work = work.sort_values(x_col)
    if len(work) < _MIN_ROWS or len(work) > _MAX_LINE_POINTS:
        return _none()
    fig = px.line(work, x=x_col, y=y_col, markers=True, title=title)
    _style(fig)
    return {"chart": fig, "chart_type": "line"}


def _category_chart(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    question: str,
) -> dict[str, Any]:
    work = frame[[x_col, y_col]].copy()
    work[x_col] = work[x_col].astype(str)
    work[y_col] = _numeric_series(work, y_col)
    work = work.dropna(subset=[y_col])
    work = work.groupby(x_col, as_index=False)[y_col].sum()
    n_cats = int(work[x_col].nunique())
    if n_cats < _MIN_ROWS:
        return _none()
    if n_cats > _MAX_BAR_CATEGORIES:
        return _none()
    work = work.sort_values(y_col, ascending=False)

    if _use_pie(work, y_col, n_cats, question):
        fig = px.pie(work, names=x_col, values=y_col, title=title)
        _style(fig)
        return {"chart": fig, "chart_type": "pie"}

    fig = px.bar(work, x=x_col, y=y_col, title=title)
    fig.update_layout(xaxis_tickangle=-30)
    _style(fig)
    return {"chart": fig, "chart_type": "bar"}


def _use_pie(work: pd.DataFrame, y_col: str, n_cats: int, question: str) -> bool:
    values = work[y_col]
    if n_cats < _MIN_ROWS or n_cats > _MAX_PIE_SLICES:
        return False
    if (values < 0).any():
        return False
    if _TOP_HINT.search(question):
        return False
    if _PCT_NAME.search(str(y_col)) or _PIE_HINT.search(question):
        return True
    return n_cats <= 5


def _style(fig: Figure) -> None:
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=60))
