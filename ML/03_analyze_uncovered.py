import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"
ORDERS_CSV = BASE_DIR / "data" / "raw" / "orders.csv"

con = duckdb.connect(str(DB_FILE))

# Recreate order metadata for this DuckDB session
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_orders AS

SELECT
    order_id,
    user_id,
    eval_set,
    order_number,
    order_dow,
    order_hour_of_day,
    days_since_prior_order

FROM read_csv_auto('{ORDERS_CSV}')
""")

print("=" * 70)
print("ANALYZING UNCOVERED NEXT-ORDER PRODUCTS")
print("=" * 70)

result = con.execute("""
WITH target_products AS (

    SELECT DISTINCT
        o.user_id,
        op.product_id

    FROM order_products op

    JOIN ml_orders o
        ON op.order_id = o.order_id

    WHERE o.eval_set = 'train'
),

historical_products AS (

    SELECT DISTINCT
        o.user_id,
        op.product_id

    FROM order_products op

    JOIN ml_orders o
        ON op.order_id = o.order_id

    WHERE o.eval_set = 'prior'
)

SELECT

    COUNT(*) AS target_products,

    SUM(
        CASE
            WHEN hp.product_id IS NULL
            THEN 1
            ELSE 0
        END
    ) AS new_products,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN hp.product_id IS NULL
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS new_product_percentage

FROM target_products tp

LEFT JOIN historical_products hp
    ON tp.user_id = hp.user_id
    AND tp.product_id = hp.product_id
""").fetchone()

print(f"Target products       : {result[0]:,}")
print(f"New-to-customer       : {result[1]:,}")
print(f"New product percentage: {result[2]}%")


# ============================================================
# 2. Compare covered vs uncovered products
# ============================================================

print("\n2. COVERED vs UNCOVERED")
print("-" * 70)

result = con.execute("""
WITH target_products AS (

    SELECT DISTINCT
        o.user_id,
        op.product_id

    FROM order_products op

    JOIN ml_orders o
        ON op.order_id = o.order_id

    WHERE o.eval_set = 'train'
),

historical_products AS (

    SELECT DISTINCT
        o.user_id,
        op.product_id

    FROM order_products op

    JOIN ml_orders o
        ON op.order_id = o.order_id

    WHERE o.eval_set = 'prior'
),

classified AS (

    SELECT

        tp.user_id,
        tp.product_id,

        CASE
            WHEN hp.product_id IS NOT NULL
            THEN 'Previously Purchased'
            ELSE 'New Product'
        END AS product_type

    FROM target_products tp

    LEFT JOIN historical_products hp

        ON tp.user_id = hp.user_id
        AND tp.product_id = hp.product_id
)

SELECT

    product_type,

    COUNT(*) AS products,

    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM classified

GROUP BY product_type
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 3. How many products are candidates per customer?
# ============================================================

print("\n3. CANDIDATES PER CUSTOMER")
print("-" * 70)

result = con.execute("""
SELECT

    ROUND(AVG(candidate_count), 2)
        AS avg_candidates,

    MIN(candidate_count)
        AS min_candidates,

    MAX(candidate_count)
        AS max_candidates,

    ROUND(
        MEDIAN(candidate_count),
        2
    ) AS median_candidates

FROM (

    SELECT
        user_id,
        COUNT(*) AS candidate_count

    FROM ml_candidates

    GROUP BY user_id

)
""").fetchone()

print(f"Average candidates : {result[0]}")
print(f"Median candidates  : {result[3]}")
print(f"Minimum candidates : {result[1]}")
print(f"Maximum candidates : {result[2]}")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

con.close()