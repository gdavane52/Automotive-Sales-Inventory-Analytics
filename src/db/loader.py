from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from src.config.settings import RAW_DATA_DIR, SQLITE_DB_PATH
from src.db.connection import connect
from src.db.schema import DDL

CSV_FILES = {
    "vehicle_stock": "vehicle_stock.csv",
    "digital_checkout": "digital_checkout.csv",
    "sales": "sales.csv",
    "trade_in": "trade_in.csv",
}


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _int(value: str | None) -> int | None:
    text = _blank(value)
    return int(float(text)) if text is not None else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)


def load_vehicle_stock(conn: sqlite3.Connection, rows: list[dict[str, str]]) -> int:
    payload = [
        (
            row["vehicle_id"],
            row["brand"],
            row["model"],
            row["variant"],
            row["fuel_type"],
            row["location"],
            row["dealer_id"],
            row["stock_date"],
            row["stock_status"],
            _int(row["listed_price"]),
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO vehicle_stock (
            vehicle_id, brand, model, variant, fuel_type, location, dealer_id,
            stock_date, stock_status, listed_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def load_digital_checkout(conn: sqlite3.Connection, rows: list[dict[str, str]]) -> int:
    payload = [
        (
            row["checkout_id"],
            row["customer_id"],
            row["vehicle_id"],
            row["checkout_date"],
            row["brand"],
            row["model"],
            row["location"],
            row["checkout_stage"],
            row["checkout_status"],
            row["channel"],
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO digital_checkout (
            checkout_id, customer_id, vehicle_id, checkout_date, brand, model,
            location, checkout_stage, checkout_status, channel
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def load_sales(conn: sqlite3.Connection, rows: list[dict[str, str]]) -> int:
    payload = [
        (
            row["sale_id"],
            row["customer_id"],
            row["vehicle_id"],
            row["sale_date"],
            row["brand"],
            row["model"],
            row["location"],
            row["dealer_id"],
            _int(row["selling_price"]),
            row["trade_in_used"],
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO sales (
            sale_id, customer_id, vehicle_id, sale_date, brand, model, location,
            dealer_id, selling_price, trade_in_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def load_trade_in(conn: sqlite3.Connection, rows: list[dict[str, str]]) -> int:
    payload = [
        (
            row["trade_in_id"],
            row["customer_id"],
            row["new_vehicle_id"],
            row["trade_in_date"],
            row["trade_in_brand"],
            row["trade_in_model"],
            _int(row["trade_in_year"]),
            _int(row["trade_in_mileage"]),
            _int(row["trade_in_value"]),
            row["location"],
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO trade_in (
            trade_in_id, customer_id, new_vehicle_id, trade_in_date,
            trade_in_brand, trade_in_model, trade_in_year, trade_in_mileage,
            trade_in_value, location
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def initialize_database(
    db_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> dict[str, int]:
    """Create schema and load all four CSVs. Replaces existing tables."""
    csv_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    conn = connect(db_path)
    try:
        apply_schema(conn)
        counts = {
            "vehicle_stock": load_vehicle_stock(
                conn,
                _read_csv(csv_dir / CSV_FILES["vehicle_stock"]),
            ),
            "digital_checkout": load_digital_checkout(
                conn,
                _read_csv(csv_dir / CSV_FILES["digital_checkout"]),
            ),
            "sales": load_sales(
                conn,
                _read_csv(csv_dir / CSV_FILES["sales"]),
            ),
            "trade_in": load_trade_in(
                conn,
                _read_csv(csv_dir / CSV_FILES["trade_in"]),
            ),
        }
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def default_db_path() -> Path:
    return SQLITE_DB_PATH
