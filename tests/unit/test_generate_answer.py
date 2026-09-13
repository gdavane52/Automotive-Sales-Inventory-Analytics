from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.answer_agent import generate_business_answer
from src.langgraph.nodes import generate_answer


class GenerateBusinessAnswerTests(unittest.TestCase):
    def test_empty_dataframe_does_not_call_llm(self) -> None:
        with patch("src.agents.answer_agent.get_chat_llm") as llm:
            answer = generate_business_answer(
                "Which are the top 10 selling car models in Pune in 2026?",
                pd.DataFrame(columns=["model", "units"]),
                "SELECT model FROM sales",
            )

        llm.assert_not_called()
        self.assertIn("no matching data", answer.lower())

    def test_none_result_does_not_call_llm(self) -> None:
        with patch("src.agents.answer_agent.get_chat_llm") as llm:
            answer = generate_business_answer("Top models?", None, "SELECT 1")

        llm.assert_not_called()
        self.assertIn("no matching data", answer.lower())

    def test_uses_query_result_only(self) -> None:
        fake_llm = MagicMock()
        fake_chain = MagicMock()
        chunk = MagicMock()
        chunk.content = "Creta led sales in Pune with 120 units."
        fake_chain.stream.return_value = [chunk]
        fake_prompt = MagicMock()
        fake_prompt.__or__ = MagicMock(return_value=fake_chain)

        frame = pd.DataFrame({"model": ["Creta"], "units": [120]})
        with (
            patch("src.agents.answer_agent.get_chat_llm", return_value=fake_llm),
            patch(
                "src.agents.answer_agent.ChatPromptTemplate.from_messages",
                return_value=fake_prompt,
            ),
        ):
            answer = generate_business_answer(
                "Which model sold the most in Pune?",
                frame,
                "SELECT model, COUNT(*) AS units FROM sales GROUP BY model",
            )

        payload = fake_chain.stream.call_args.args[0]
        self.assertIn("Creta", payload["query_result"])
        self.assertIn("120", payload["query_result"])
        self.assertNotIn("SELECT", answer)
        self.assertEqual(answer, "Creta led sales in Pune with 120 units.")

    def test_node_writes_final_answer(self) -> None:
        frame = pd.DataFrame({"model": ["Creta"], "units": [120]})
        with patch(
            "src.langgraph.nodes.generate_business_answer",
            return_value="Creta led with 120 units.",
        ) as mocked:
            update = generate_answer(
                {
                    "user_question": "Top model in Pune?",
                    "query_result": frame,
                    "sql": "SELECT model FROM sales",
                }
            )

        mocked.assert_called_once()
        self.assertEqual(update["final_answer"], "Creta led with 120 units.")


if __name__ == "__main__":
    unittest.main()
