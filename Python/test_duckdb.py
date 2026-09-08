import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

prior_path = RAW_DIR / "order_products__prior.csv"

con = duckdb.connect()

print("Testing DuckDB...")

result = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT order_id) AS unique_orders,
        COUNT(DISTINCT product_id) AS unique_products
    FROM read_csv_auto('{prior_path}')
""").fetchone()

print(f"Total rows: {result[0]:,}")
print(f"Unique orders: {result[1]:,}")
print(f"Unique products: {result[2]:,}")

con.close()