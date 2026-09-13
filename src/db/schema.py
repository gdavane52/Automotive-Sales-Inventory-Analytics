"""SQLite schema for automotive sales and inventory analytics.

Tables (exactly 10 columns each; matches the four CSVs)
----------------------------------------------------------
vehicle_stock
digital_checkout.vehicle_id  →  vehicle_stock.vehicle_id
sales.vehicle_id             →  vehicle_stock.vehicle_id
trade_in.customer_id        →  sales.customer_id
trade_in.new_vehicle_id      →  sales.vehicle_id (when trade_in_used = true)

CSV mapping
-----------
vehicle_stock.csv    → vehicle_stock
digital_checkout.csv → digital_checkout
sales.csv            → sales
trade_in.csv         → trade_in
"""

DDL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS trade_in;
DROP TABLE IF EXISTS trade_ins;
DROP TABLE IF EXISTS digital_checkout;
DROP TABLE IF EXISTS checkouts;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS vehicle_stock;
DROP TABLE IF EXISTS vehicles;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS dealerships;

CREATE TABLE vehicle_stock (
    vehicle_id TEXT PRIMARY KEY,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    variant TEXT NOT NULL,
    fuel_type TEXT NOT NULL,
    location TEXT NOT NULL,
    dealer_id TEXT NOT NULL,
    stock_date TEXT NOT NULL,
    stock_status TEXT NOT NULL,
    listed_price INTEGER NOT NULL
);

CREATE TABLE digital_checkout (
    checkout_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    vehicle_id TEXT NOT NULL REFERENCES vehicle_stock (vehicle_id),
    checkout_date TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    location TEXT NOT NULL,
    checkout_stage TEXT NOT NULL,
    checkout_status TEXT NOT NULL,
    channel TEXT NOT NULL
);

CREATE TABLE sales (
    sale_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    vehicle_id TEXT NOT NULL UNIQUE REFERENCES vehicle_stock (vehicle_id),
    sale_date TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    location TEXT NOT NULL,
    dealer_id TEXT NOT NULL,
    selling_price INTEGER NOT NULL,
    trade_in_used TEXT NOT NULL
);

CREATE TABLE trade_in (
    trade_in_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    new_vehicle_id TEXT NOT NULL REFERENCES vehicle_stock (vehicle_id),
    trade_in_date TEXT NOT NULL,
    trade_in_brand TEXT NOT NULL,
    trade_in_model TEXT NOT NULL,
    trade_in_year INTEGER NOT NULL,
    trade_in_mileage INTEGER NOT NULL,
    trade_in_value INTEGER NOT NULL,
    location TEXT NOT NULL
);

CREATE INDEX idx_stock_location ON vehicle_stock (location);
CREATE INDEX idx_stock_brand_model ON vehicle_stock (brand, model);
CREATE INDEX idx_stock_dealer ON vehicle_stock (dealer_id);
CREATE INDEX idx_stock_status ON vehicle_stock (stock_status);
CREATE INDEX idx_checkout_vehicle ON digital_checkout (vehicle_id);
CREATE INDEX idx_checkout_date ON digital_checkout (checkout_date);
CREATE INDEX idx_checkout_status ON digital_checkout (checkout_status);
CREATE INDEX idx_sales_date ON sales (sale_date);
CREATE INDEX idx_sales_customer ON sales (customer_id);
CREATE INDEX idx_sales_location ON sales (location);
CREATE INDEX idx_trade_in_customer ON trade_in (customer_id);
CREATE INDEX idx_trade_in_vehicle ON trade_in (new_vehicle_id);
"""
