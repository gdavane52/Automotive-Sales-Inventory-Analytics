from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.schema import DDL
from src.db.service import get_database_schema
from src.tools.database import get_database_schema as tool_get_database_schema


def _memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    return conn


class GetDatabaseSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _memory_db()

    def tearDown(self) -> None:
        self.conn.close()

    def test_returns_all_user_tables(self) -> None:
        schema = get_database_schema(conn=self.conn)
        self.assertEqual(
            schema["table_names"],
            ["digital_checkout", "sales", "trade_in", "vehicle_stock"],
        )

    def test_column_types_and_primary_keys(self) -> None:
        schema = get_database_schema(conn=self.conn)
        sales = schema["tables"]["sales"]
        by_name = {col["name"]: col for col in sales["columns"]}
        self.assertEqual(sales["primary_keys"], ["sale_id"])
        self.assertEqual(schema["primary_keys"]["vehicle_stock"], ["vehicle_id"])
        self.assertEqual(len(sales["columns"]), 10)
        self.assertEqual(len(schema["tables"]["vehicle_stock"]["columns"]), 10)
        self.assertEqual(by_name["sale_id"]["data_type"], "TEXT")
        self.assertEqual(by_name["selling_price"]["data_type"], "INTEGER")
        self.assertFalse(by_name["sale_id"]["nullable"])

    def test_foreign_keys_and_relationships(self) -> None:
        schema = get_database_schema(conn=self.conn)
        vehicle_fk = next(
            fk
            for fk in schema["foreign_keys"]
            if fk["from_table"] == "sales" and fk["from_columns"] == ["vehicle_id"]
        )
        self.assertEqual(vehicle_fk["to_table"], "vehicle_stock")
        self.assertEqual(vehicle_fk["to_columns"], ["vehicle_id"])

        rels = {(r["from_table"], tuple(r["from_columns"])): r for r in schema["relationships"]}
        self.assertEqual(rels[("sales", ("vehicle_id",))]["relationship_type"], "one-to-one")
        self.assertEqual(
            rels[("digital_checkout", ("vehicle_id",))]["relationship_type"],
            "many-to-one",
        )
        self.assertEqual(rels[("trade_in", ("new_vehicle_id",))]["to_table"], "vehicle_stock")

    def test_tool_wrapper_matches_service(self) -> None:
        schema = get_database_schema(conn=self.conn)
        via_tool = tool_get_database_schema(conn=self.conn)
        self.assertEqual(schema, via_tool)


if __name__ == "__main__":
    unittest.main()
