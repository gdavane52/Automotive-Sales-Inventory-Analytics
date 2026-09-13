"""Generate four synthetic automotive CSVs for SQLite analytics.

Usage (from automotive-analytics/):

    py -3 data/generate_data.py
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_STOCK = 5000
N_CHECKOUT = 10000
N_SALES = 5000
N_TRADE = 2500

START = date(2025, 1, 2)
END = date(2026, 12, 20)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw"

LOCATIONS = [
    "Pune",
    "Mumbai",
    "Bengaluru",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Ahmedabad",
    "Gurgaon",
]
LOCATION_WEIGHTS = [12, 16, 14, 15, 11, 12, 9, 11]

DEALERS = {
    "Pune": ["DLR-PUN-01", "DLR-PUN-02", "DLR-PUN-03"],
    "Mumbai": ["DLR-MUM-01", "DLR-MUM-02", "DLR-MUM-03"],
    "Bengaluru": ["DLR-BLR-01", "DLR-BLR-02", "DLR-BLR-03"],
    "Delhi": ["DLR-DEL-01", "DLR-DEL-02", "DLR-DEL-03"],
    "Hyderabad": ["DLR-HYD-01", "DLR-HYD-02"],
    "Chennai": ["DLR-CHN-01", "DLR-CHN-02", "DLR-CHN-03"],
    "Ahmedabad": ["DLR-AMD-01", "DLR-AMD-02"],
    "Gurgaon": ["DLR-GGN-01", "DLR-GGN-02"],
}

# brand, model, variant, fuel, price_lo, price_hi, stock_weight, checkout_weight
# Higher stock_weight + lower checkout_weight => high inventory / slower demand.
CATALOG = [
    ("Maruti Suzuki", "Swift", "VXI", "Petrol", 649000, 790000, 13, 9),
    ("Maruti Suzuki", "Baleno", "Zeta", "Petrol", 720000, 890000, 10, 8),
    ("Maruti Suzuki", "Brezza", "ZXI", "Petrol", 980000, 1280000, 9, 8),
    ("Maruti Suzuki", "Ertiga", "ZXI+", "CNG", 990000, 1290000, 7, 6),
    ("Hyundai", "Creta", "SX", "Petrol", 1380000, 1890000, 11, 14),
    ("Hyundai", "Creta", "SX(O)", "Diesel", 1550000, 2080000, 6, 9),
    ("Hyundai", "Venue", "S", "Petrol", 940000, 1290000, 8, 7),
    ("Hyundai", "i20", "Asta", "Petrol", 780000, 990000, 6, 5),
    ("Tata", "Nexon", "Fearless+", "Electric", 1490000, 1890000, 8, 13),
    ("Tata", "Nexon", "Creative", "Petrol", 980000, 1380000, 7, 7),
    ("Tata", "Punch", "Adventure", "Petrol", 640000, 910000, 12, 4),
    ("Tata", "Tiago", "XZ+", "Petrol", 560000, 740000, 12, 3),
    ("Mahindra", "XUV700", "AX7", "Diesel", 1890000, 2650000, 6, 8),
    ("Mahindra", "Scorpio-N", "Z8L", "Diesel", 1690000, 2420000, 5, 7),
    ("Mahindra", "Thar", "LX", "Diesel", 1420000, 1980000, 5, 12),
    ("Kia", "Seltos", "HTX", "Petrol", 1280000, 1780000, 7, 8),
    ("Toyota", "Innova Hycross", "ZX", "Hybrid", 2280000, 3250000, 4, 6),
    ("Honda", "City", "ZX", "Petrol", 1280000, 1680000, 5, 5),
    ("MG", "Hector", "Sharp", "Petrol", 1620000, 2240000, 3, 4),
    ("Volkswagen", "Virtus", "GT", "Petrol", 1380000, 1860000, 3, 3),
    ("Skoda", "Kushaq", "Style", "Petrol", 1290000, 1820000, 3, 3),
    ("BMW", "3 Series", "330Li", "Petrol", 4800000, 5900000, 2, 2),
]

TRADE_MODELS = [
    ("Maruti Suzuki", "Swift"),
    ("Maruti Suzuki", "Dzire"),
    ("Hyundai", "i20"),
    ("Hyundai", "Venue"),
    ("Honda", "City"),
    ("Honda", "Amaze"),
    ("Tata", "Nexon"),
    ("Tata", "Tiago"),
    ("Hyundai", "Creta"),
    ("Kia", "Seltos"),
    ("Mahindra", "XUV500"),
    ("Ford", "EcoSport"),
    ("Renault", "Kwid"),
    ("Toyota", "Innova"),
]

CHANNELS = ["Website", "Mobile App", "Walk-in", "Call Center"]
# Walk-in converts better; website/app abandon more often.
CHANNEL_COMPLETE = {
    "Website": 0.32,
    "Mobile App": 0.38,
    "Walk-in": 0.72,
    "Call Center": 0.48,
}

OLD_CSV_NAMES = ("vehicles.csv", "checkout.csv")


def _pick_date(rng: random.Random, start: date, end: date) -> date:
    span = max((end - start).days, 0)
    return start + timedelta(days=rng.randint(0, span))


def _money(rng: random.Random, lo: int, hi: int) -> int:
    step = 1000 if hi - lo > 50000 else 500
    lo_s, hi_s = lo // step, hi // step
    return rng.randint(lo_s, hi_s) * step


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _trade_value(rng: random.Random, year: int, mileage: int) -> int:
    age = 2026 - year
    base = rng.randint(180000, 720000)
    value = int(base * max(0.22, 1 - 0.07 * age))
    if mileage > 100000:
        value = int(value * rng.uniform(0.62, 0.78))
    elif mileage > 60000:
        value = int(value * rng.uniform(0.78, 0.90))
    return max(45000, value // 1000 * 1000)


def generate() -> dict[str, int]:
    rng = random.Random(SEED)
    stock_weights = [row[6] for row in CATALOG]
    checkout_weights = [row[7] for row in CATALOG]

    vehicles: list[dict] = []
    for i in range(1, N_STOCK + 1):
        spec = rng.choices(CATALOG, weights=stock_weights, k=1)[0]
        brand, model, variant, fuel, lo, hi, _, _ = spec
        location = rng.choices(LOCATIONS, weights=LOCATION_WEIGHTS, k=1)[0]
        dealer_id = rng.choice(DEALERS[location])
        stock_date = _pick_date(rng, START, END - timedelta(days=21))
        listed = _money(rng, lo, hi)
        vehicles.append(
            {
                "vehicle_id": f"VEH-{i:05d}",
                "brand": brand,
                "model": model,
                "variant": variant,
                "fuel_type": fuel,
                "location": location,
                "dealer_id": dealer_id,
                "stock_date": stock_date.isoformat(),
                "stock_status": "In Stock",
                "listed_price": listed,
                "_spec_key": (brand, model),
            }
        )

    slow_keys = {("Tata", "Tiago"), ("Tata", "Punch")}
    # Unique vehicle_id in sales, so leftover inventory means fewer than 5,000 sales.
    # Keep 800 unsold (In Stock / Reserved) and sell the rest (4,200). Then sell extra
    # from the unsold pool until sales hit 5,000 — that would wipe leftover. Instead,
    # sell exactly 5,000 units (all stock) and encode "high inventory / low demand"
    # via model mix, dwell time, and checkout conversion.
    sold_vehicles = list(vehicles)
    rng.shuffle(sold_vehicles)
    sold_vehicles = sold_vehicles[:N_SALES]
    sold_ids = {v["vehicle_id"] for v in sold_vehicles}
    leftover = [v for v in vehicles if v["vehicle_id"] not in sold_ids]
    rng.shuffle(leftover)
    reserved_ids = {v["vehicle_id"] for v in leftover[: min(120, len(leftover))]}
    for row in vehicles:
        if row["vehicle_id"] in sold_ids:
            row["stock_status"] = "Sold"
        elif row["vehicle_id"] in reserved_ids:
            row["stock_status"] = "Reserved"
        else:
            row["stock_status"] = "In Stock"

    sales: list[dict] = []
    sold_by_id = {v["vehicle_id"]: v for v in sold_vehicles}
    for i, veh in enumerate(sold_vehicles, start=1):
        stock_day = date.fromisoformat(veh["stock_date"])
        # Slow models sit longer before sale.
        if veh["_spec_key"] in slow_keys:
            dwell = rng.randint(35, 140)
        else:
            dwell = rng.randint(4, 75)
        sale_day = stock_day + timedelta(days=dwell)
        if sale_day > END:
            sale_day = END
        listed = int(veh["listed_price"])
        selling = int(listed * rng.uniform(0.96, 1.02))
        selling = max(selling // 1000 * 1000, listed - 40000)
        sales.append(
            {
                "sale_id": f"SAL-{i:05d}",
                "customer_id": f"CUS-{i:05d}",
                "vehicle_id": veh["vehicle_id"],
                "sale_date": sale_day.isoformat(),
                "brand": veh["brand"],
                "model": veh["model"],
                "location": veh["location"],
                "dealer_id": veh["dealer_id"],
                "selling_price": selling,
                "trade_in_used": "false",
            }
        )

    rng.shuffle(sales)
    trade_sales = sales[:N_TRADE]
    trade_ids = {row["sale_id"] for row in trade_sales}
    for row in sales:
        row["trade_in_used"] = "true" if row["sale_id"] in trade_ids else "false"
    sales.sort(key=lambda r: r["sale_id"])

    sale_by_vehicle = {row["vehicle_id"]: row for row in sales}
    trade_ins: list[dict] = []
    for i, sale in enumerate(trade_sales, start=1):
        brand, model = rng.choice(TRADE_MODELS)
        year = rng.randint(2012, 2023)
        mileage = rng.randint(18000, 145000)
        sale_day = date.fromisoformat(sale["sale_date"])
        trade_day = sale_day - timedelta(days=rng.randint(0, 4))
        trade_ins.append(
            {
                "trade_in_id": f"TRD-{i:05d}",
                "customer_id": sale["customer_id"],
                "new_vehicle_id": sale["vehicle_id"],
                "trade_in_date": trade_day.isoformat(),
                "trade_in_brand": brand,
                "trade_in_model": model,
                "trade_in_year": year,
                "trade_in_mileage": mileage,
                "trade_in_value": _trade_value(rng, year, mileage),
                "location": sale["location"],
            }
        )

    checkouts: list[dict] = []
    checkout_n = 1
    # One completed checkout per sale (does not mean every checkout becomes a sale).
    for sale in sales:
        veh = sold_by_id[sale["vehicle_id"]]
        sale_day = date.fromisoformat(sale["sale_date"])
        checkout_day = sale_day - timedelta(days=rng.randint(0, 6))
        stock_day = date.fromisoformat(veh["stock_date"])
        if checkout_day < stock_day:
            checkout_day = stock_day
        channel = rng.choices(CHANNELS, weights=[22, 28, 38, 12], k=1)[0]
        checkouts.append(
            {
                "checkout_id": f"CHK-{checkout_n:05d}",
                "customer_id": sale["customer_id"],
                "vehicle_id": sale["vehicle_id"],
                "checkout_date": checkout_day.isoformat(),
                "brand": sale["brand"],
                "model": sale["model"],
                "location": sale["location"],
                "checkout_stage": "Completed",
                "checkout_status": "Completed",
                "channel": channel,
            }
        )
        checkout_n += 1

    remaining = N_CHECKOUT - len(checkouts)
    by_key: dict[tuple[str, str], list[dict]] = {}
    for veh in vehicles:
        by_key.setdefault(veh["_spec_key"], []).append(veh)

    for _ in range(remaining):
        spec = rng.choices(CATALOG, weights=checkout_weights, k=1)[0]
        key = (spec[0], spec[1])
        pool = by_key.get(key) or vehicles
        veh = rng.choice(pool)
        stock_day = date.fromisoformat(veh["stock_date"])
        checkout_day = _pick_date(rng, stock_day, min(END, stock_day + timedelta(days=120)))
        channel = rng.choices(CHANNELS, weights=[34, 30, 18, 18], k=1)[0]
        complete_p = CHANNEL_COMPLETE[channel]
        roll = rng.random()
        if roll < complete_p * 0.25:
            stage, status = "Payment", "Payment"
        elif roll < complete_p * 0.25 + 0.22:
            stage, status = "Started", "Started"
        elif roll < 0.78:
            abandoned_stage = rng.choice(["Started", "Payment"])
            stage, status = abandoned_stage, "Abandoned"
        else:
            stage, status = "Payment", "Payment"
        # These extra checkouts are not sales; keep them non-completed overall.
        if status == "Completed":
            stage, status = "Payment", "Abandoned"
        checkouts.append(
            {
                "checkout_id": f"CHK-{checkout_n:05d}",
                "customer_id": f"CUS-{rng.randint(20000, 99999):05d}",
                "vehicle_id": veh["vehicle_id"],
                "checkout_date": checkout_day.isoformat(),
                "brand": veh["brand"],
                "model": veh["model"],
                "location": veh["location"],
                "checkout_stage": stage,
                "checkout_status": status,
                "channel": channel,
            }
        )
        checkout_n += 1

    for row in vehicles:
        row.pop("_spec_key", None)

    stock_fields = [
        "vehicle_id",
        "brand",
        "model",
        "variant",
        "fuel_type",
        "location",
        "dealer_id",
        "stock_date",
        "stock_status",
        "listed_price",
    ]
    checkout_fields = [
        "checkout_id",
        "customer_id",
        "vehicle_id",
        "checkout_date",
        "brand",
        "model",
        "location",
        "checkout_stage",
        "checkout_status",
        "channel",
    ]
    sales_fields = [
        "sale_id",
        "customer_id",
        "vehicle_id",
        "sale_date",
        "brand",
        "model",
        "location",
        "dealer_id",
        "selling_price",
        "trade_in_used",
    ]
    trade_fields = [
        "trade_in_id",
        "customer_id",
        "new_vehicle_id",
        "trade_in_date",
        "trade_in_brand",
        "trade_in_model",
        "trade_in_year",
        "trade_in_mileage",
        "trade_in_value",
        "location",
    ]

    _write_csv(OUT_DIR / "vehicle_stock.csv", vehicles, stock_fields)
    _write_csv(OUT_DIR / "digital_checkout.csv", checkouts, checkout_fields)
    _write_csv(OUT_DIR / "sales.csv", sales, sales_fields)
    _write_csv(OUT_DIR / "trade_in.csv", trade_ins, trade_fields)

    for name in OLD_CSV_NAMES:
        old = OUT_DIR / name
        if old.exists():
            old.unlink()

    return {
        "vehicle_stock": len(vehicles),
        "digital_checkout": len(checkouts),
        "sales": len(sales),
        "trade_in": len(trade_ins),
    }


def main() -> None:
    counts = generate()
    print(f"Wrote CSVs to {OUT_DIR}")
    for name, count in counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
