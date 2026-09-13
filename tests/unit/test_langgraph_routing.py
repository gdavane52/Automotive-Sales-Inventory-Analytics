from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.langgraph.graph import build_analytics_graph
from src.langgraph.routing import route_after_question, route_after_validation
from src.prompts.sql_prompt import format_correction_block


class RouteAfterQuestionTests(unittest.TestCase):
    def test_in_scope_goes_to_generate_sql(self) -> None:
        self.assertEqual(
            route_after_question({"in_scope": True}),
            "generate_sql",
        )

    def test_out_of_scope_ends(self) -> None:
        self.assertEqual(
            route_after_question({"in_scope": False, "final_answer": "refused"}),
            "end",
        )


class RouteAfterValidationTests(unittest.TestCase):
    def test_valid_sql_goes_to_execute(self) -> None:
        self.assertEqual(
            route_after_validation({"validation_result": True, "retry_count": 1}),
            "execute_sql",
        )

    def test_invalid_sql_retries_generate(self) -> None:
        self.assertEqual(
            route_after_validation(
                {
                    "validation_result": False,
                    "retry_count": 1,
                    "validation_error": "Unknown table(s): vehicle",
                }
            ),
            "generate_sql",
        )

    def test_invalid_sql_stops_after_three_attempts(self) -> None:
        self.assertEqual(
            route_after_validation(
                {
                    "validation_result": False,
                    "retry_count": 3,
                    "validation_error": "Unknown table(s): vehicle",
                }
            ),
            "validation_failed",
        )


class CorrectionBlockTests(unittest.TestCase):
    def test_includes_previous_sql_and_error(self) -> None:
        text = format_correction_block(
            "SELECT car_model FROM vehicle",
            "Table 'vehicle' does not exist.",
        )
        self.assertIn("Previous SQL:", text)
        self.assertIn("SELECT car_model FROM vehicle", text)
        self.assertIn("Validation error:", text)
        self.assertIn("Table 'vehicle' does not exist.", text)
        self.assertIn("Generate a corrected SQL query", text)

    def test_empty_when_no_feedback(self) -> None:
        self.assertEqual(format_correction_block("", ""), "")


class CompiledGraphTests(unittest.TestCase):
    def test_graph_includes_core_nodes_and_conditional_routing(self) -> None:
        compiled = build_analytics_graph()
        drawable = compiled.get_graph()
        names = set(drawable.nodes.keys())
        for node in (
            "validate_question",
            "generate_sql",
            "validate_sql",
            "execute_sql",
            "generate_answer",
            "generate_chart",
            "generate_insight",
        ):
            self.assertIn(node, names)

        edge_pairs = {(edge.source, edge.target) for edge in drawable.edges}
        self.assertIn(("validate_question", "generate_sql"), edge_pairs)
        self.assertIn(("generate_sql", "validate_sql"), edge_pairs)
        self.assertIn(("validate_sql", "execute_sql"), edge_pairs)
        self.assertIn(("validate_sql", "generate_sql"), edge_pairs)
        self.assertIn(("execute_sql", "generate_answer"), edge_pairs)
        self.assertIn(("generate_answer", "generate_chart"), edge_pairs)
        self.assertIn(("generate_chart", "generate_insight"), edge_pairs)


if __name__ == "__main__":
    unittest.main()
