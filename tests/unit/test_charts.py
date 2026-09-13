from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from plotly.graph_objects import Figure

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.charts.service import generate_chart
from src.langgraph.nodes import generate_chart as generate_chart_node


class ChartServiceTests(unittest.TestCase):
    def test_category_and_numeric_makes_bar(self) -> None:
        frame = pd.DataFrame(
            {
                "model": [f"Model {i}" for i in range(10)],
                "total_sales": list(range(10, 0, -1)),
            }
        )
        result = generate_chart(
            frame,
            "Which are the top 10 selling car models in Pune in 2026?",
        )
        self.assertEqual(result["chart_type"], "bar")
        self.assertIsInstance(result["chart"], Figure)
        self.assertEqual(result["chart"].data[0].type, "bar")

    def test_date_and_numeric_makes_line(self) -> None:
        frame = pd.DataFrame(
            {
                "sale_date": pd.to_datetime(
                    ["2026-01-01", "2026-02-01", "2026-03-01"]
                ),
                "total_sales": [10, 15, 12],
            }
        )
        result = generate_chart(frame, "Monthly vehicle sales in 2026")
        self.assertEqual(result["chart_type"], "line")
        self.assertEqual(result["chart"].data[0].type, "scatter")

    def test_proportion_makes_pie(self) -> None:
        frame = pd.DataFrame(
            {
                "fuel_type": ["Petrol", "Diesel", "Electric", "CNG"],
                "share": [40, 30, 20, 10],
            }
        )
        result = generate_chart(
            frame, "What is the fuel type mix of vehicles in stock?"
        )
        self.assertEqual(result["chart_type"], "pie")
        self.assertEqual(result["chart"].data[0].type, "pie")

    def test_empty_or_single_row_returns_no_chart(self) -> None:
        empty = generate_chart(pd.DataFrame(columns=["model", "total_sales"]))
        self.assertIsNone(empty["chart"])
        self.assertIsNone(empty["chart_type"])

        one = generate_chart(
            pd.DataFrame({"model": ["Creta"], "total_sales": [12]}),
            "Top selling model",
        )
        self.assertIsNone(one["chart"])
        self.assertIsNone(one["chart_type"])

    def test_too_many_categories_returns_no_chart(self) -> None:
        frame = pd.DataFrame(
            {
                "model": [f"Model {i}" for i in range(40)],
                "total_sales": list(range(40)),
            }
        )
        result = generate_chart(frame, "Sales by model")
        self.assertIsNone(result["chart"])
        self.assertIsNone(result["chart_type"])

    def test_node_writes_chart_state(self) -> None:
        frame = pd.DataFrame(
            {"model": ["Creta", "Nexon"], "total_sales": [120, 90]}
        )
        update = generate_chart_node(
            {
                "user_question": (
                    "Which are the top 10 selling car models in Pune in 2026?"
                ),
                "query_result": frame,
            }
        )
        self.assertEqual(update["chart_type"], "bar")
        self.assertIsInstance(update["chart"], Figure)


if __name__ == "__main__":
    unittest.main()
