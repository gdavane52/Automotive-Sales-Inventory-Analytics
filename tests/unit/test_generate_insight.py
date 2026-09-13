from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.insight_agent import generate_business_insights
from src.langgraph.nodes import generate_insight


class GenerateInsightTests(unittest.TestCase):
    def test_empty_dataframe_does_not_call_llm(self) -> None:
        with patch("src.agents.insight_agent.get_chat_llm") as llm:
            insights = generate_business_insights(
                "Which are the top 10 selling car models in Pune in 2026?",
                pd.DataFrame(columns=["model", "total_sales"]),
                "No matching data was found for this question.",
            )

        llm.assert_not_called()
        self.assertEqual(len(insights), 1)
        self.assertIn("does not support a meaningful insight", insights[0].lower())

    def test_uses_query_result_not_external_facts(self) -> None:
        fake_llm = MagicMock()
        fake_chain = MagicMock()
        chunk = MagicMock()
        chunk.content = (
            "- Creta recorded the highest sales among the returned models, "
            "indicating that it was the strongest-selling model in the "
            "analyzed Pune sales data."
        )
        fake_chain.stream.return_value = [chunk]
        fake_prompt = MagicMock()
        fake_prompt.__or__ = MagicMock(return_value=fake_chain)

        frame = pd.DataFrame(
            {"model": ["Creta", "Nexon", "Swift"], "total_sales": [120, 90, 80]}
        )
        with (
            patch("src.agents.insight_agent.get_chat_llm", return_value=fake_llm),
            patch(
                "src.agents.insight_agent.ChatPromptTemplate.from_messages",
                return_value=fake_prompt,
            ),
        ):
            insights = generate_business_insights(
                "Which are the top 10 selling car models in Pune in 2026?",
                frame,
                "Creta led with 120 units, followed by Nexon and Swift.",
            )

        payload = fake_chain.stream.call_args.args[0]
        self.assertIn("Creta", payload["query_result"])
        self.assertIn("120", payload["query_result"])
        self.assertIn("Creta led with 120 units", payload["final_answer"])
        self.assertEqual(len(insights), 1)
        self.assertIn("Creta", insights[0])

    def test_parses_streamed_bullets(self) -> None:
        from src.agents.insight_agent import _insights_from_text

        insights = _insights_from_text(
            "- Creta led this result.\n- Nexon was second.\n- Swift was third.\n- Extra."
        )
        self.assertEqual(len(insights), 3)
        self.assertEqual(insights[0], "Creta led this result.")

    def test_node_writes_business_insights(self) -> None:
        frame = pd.DataFrame({"model": ["Creta"], "total_sales": [120]})
        with patch(
            "src.langgraph.nodes.generate_business_insights",
            return_value=["Creta had the highest sales in this result."],
        ) as mocked:
            update = generate_insight(
                {
                    "user_question": (
                        "Which are the top 10 selling car models in Pune in 2026?"
                    ),
                    "query_result": frame,
                    "final_answer": "Creta led with 120 units.",
                }
            )

        mocked.assert_called_once()
        self.assertEqual(
            update["business_insights"],
            ["Creta had the highest sales in this result."],
        )


if __name__ == "__main__":
    unittest.main()
