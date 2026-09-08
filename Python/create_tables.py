import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "instacart_master.parquet"
DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

con = duckdb.connect(str(DB_FILE))

print("Creating analytical database...")


# --------------------------------------------------
# Orders
# --------------------------------------------------

print("Creating orders table...")

con.execute(f"""
CREATE OR REPLACE TABLE orders AS
SELECT DISTINCT
    order_id,
    user_id,
    order_number,
    order_dow,
    order_hour_of_day,
    days_since_prior_order
FROM read_parquet('{DATA_FILE}');
""")


# --------------------------------------------------
# Products
# --------------------------------------------------

print("Creating products table...")

con.execute(f"""
CREATE OR REPLACE TABLE products AS
SELECT DISTINCT
    product_id,
    product_name,
    aisle_id,
    aisle,
    department_id,
    department
FROM read_parquet('{DATA_FILE}');
""")


# --------------------------------------------------
# Order Products
# --------------------------------------------------

print("Creating order_products table...")

con.execute(f"""
CREATE OR REPLACE TABLE order_products AS
SELECT
    order_id,
    product_id,
    add_to_cart_order,
    reordered
FROM read_parquet('{DATA_FILE}');
""")


# --------------------------------------------------
# Verify
# --------------------------------------------------

print("\nTABLE SUMMARY")
print("=" * 60)

for table in ["orders", "products", "order_products"]:

    result = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()

    print(f"{table:20} {result[0]:,} rows")


con.close()

print("\nDatabase created successfully:")
print(DB_FILE)