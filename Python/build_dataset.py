import duckdb
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "instacart_master.parquet"

# --------------------------------------------------
# File paths
# --------------------------------------------------

orders = RAW_DIR / "orders.csv"
products = RAW_DIR / "products.csv"
aisles = RAW_DIR / "aisles.csv"
departments = RAW_DIR / "departments.csv"
prior = RAW_DIR / "order_products__prior.csv"
train = RAW_DIR / "order_products__train.csv"

# --------------------------------------------------
# DuckDB connection
# --------------------------------------------------

con = duckdb.connect()

print("Building Instacart master dataset...")
print("This may take a few minutes.")

# --------------------------------------------------
# Create master dataset
# --------------------------------------------------

query = f"""
COPY (

    SELECT
        op.order_id,
        o.user_id,
        o.order_number,
        o.order_dow,
        o.order_hour_of_day,
        o.days_since_prior_order,

        op.product_id,
        p.product_name,

        a.aisle_id,
        a.aisle,

        d.department_id,
        d.department,

        op.add_to_cart_order,
        op.reordered

    FROM (
        SELECT *
        FROM read_csv_auto('{prior}')

        UNION ALL

        SELECT *
        FROM read_csv_auto('{train}')
    ) op

    LEFT JOIN read_csv_auto('{orders}') o
        ON op.order_id = o.order_id

    LEFT JOIN read_csv_auto('{products}') p
        ON op.product_id = p.product_id

    LEFT JOIN read_csv_auto('{aisles}') a
        ON p.aisle_id = a.aisle_id

    LEFT JOIN read_csv_auto('{departments}') d
        ON p.department_id = d.department_id

) TO '{OUTPUT_FILE}'
(
    FORMAT PARQUET,
    COMPRESSION ZSTD
);
"""

con.execute(query)

# --------------------------------------------------
# Verify
# --------------------------------------------------

result = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT user_id) AS customers,
        COUNT(DISTINCT product_id) AS products
    FROM read_parquet('{OUTPUT_FILE}')
""").fetchone()

print("\n========================================")
print("DATASET CREATED")
print("========================================")
print(f"Rows:      {result[0]:,}")
print(f"Orders:    {result[1]:,}")
print(f"Customers: {result[2]:,}")
print(f"Products:  {result[3]:,}")
print(f"\nSaved to:")
print(OUTPUT_FILE)

con.close()