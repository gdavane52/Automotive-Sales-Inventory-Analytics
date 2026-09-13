"""Initialize SQLite from the four raw CSV datasets.

Usage (from the automotive-analytics directory):

    python scripts/init_db.py
    python scripts/init_db.py --db data/db/automotive.db --raw data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.loader import default_db_path, initialize_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SQLite tables and load CSV data.")
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db_path(),
        help="SQLite database path (default: data/db/automotive.db)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Directory containing vehicle_stock.csv, digital_checkout.csv, sales.csv, trade_in.csv",
    )
    args = parser.parse_args()

    counts = initialize_database(db_path=args.db, raw_dir=args.raw)
    print(f"Database: {args.db.resolve()}")
    print(f"Loaded from: {args.raw.resolve()}")
    for table, count in counts.items():
        print(f"  {table}: {count}")


if __name__ == "__main__":
    main()
