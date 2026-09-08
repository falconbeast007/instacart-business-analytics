import duckdb
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

ORDERS_CSV = BASE_DIR / "data" / "raw" / "orders.csv"
DEPARTMENTS_CSV = BASE_DIR / "data" / "raw" / "departments.csv"


# ============================================================
# CONNECT TO DUCKDB
# ============================================================

con = duckdb.connect(str(DB_FILE))

print("=" * 70)
print("ANALYZING NEW-TO-CUSTOMER PRODUCT BEHAVIOR")
print("=" * 70)


# ============================================================
# 1. RECREATE ORDER METADATA
#
# ml_orders is TEMP because eval_set was not stored in the
# permanent DuckDB orders table.
# ============================================================

print("\n1. LOADING ORDER METADATA")
print("-" * 70)

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


# ============================================================
# 2. LOAD DEPARTMENTS
#
# departments is not a permanent DuckDB table, so load the
# small CSV as a temporary table.
# ============================================================

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_departments AS

SELECT
    department_id,
    department

FROM read_csv_auto('{DEPARTMENTS_CSV}')
""")


# ============================================================
# 3. PRODUCT POPULARITY
#
# Popularity is calculated ONLY from prior orders.
# This prevents future information from entering the analysis.
# ============================================================

print("\n2. PRODUCT POPULARITY")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE product_popularity AS

SELECT

    op.product_id,

    COUNT(*) AS total_purchases,

    COUNT(DISTINCT o.user_id)
        AS unique_customers

FROM order_products op

JOIN ml_orders o
    ON op.order_id = o.order_id

WHERE o.eval_set = 'prior'

GROUP BY
    op.product_id
""")


# ============================================================
# 4. IDENTIFY NEW-TO-CUSTOMER TARGET PRODUCTS
#
# These are products appearing in the customer's train order
# that the customer NEVER purchased in their prior history.
# ============================================================

print("\n3. NEW-TO-CUSTOMER PRODUCTS")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE new_target_products AS

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

    tp.user_id,
    tp.product_id

FROM target_products tp

LEFT JOIN historical_products hp

    ON tp.user_id = hp.user_id
    AND tp.product_id = hp.product_id

WHERE hp.product_id IS NULL
""")


# ============================================================
# 5. NEW PRODUCT POPULARITY SUMMARY
# ============================================================

result = con.execute("""
SELECT

    COUNT(*) AS new_products,

    ROUND(
        AVG(pp.total_purchases),
        2
    ) AS avg_prior_purchases,

    MIN(pp.total_purchases)
        AS min_prior_purchases,

    MAX(pp.total_purchases)
        AS max_prior_purchases,

    ROUND(
        MEDIAN(pp.total_purchases),
        2
    ) AS median_prior_purchases

FROM new_target_products nt

JOIN product_popularity pp

    ON nt.product_id = pp.product_id
""").fetchone()

print(f"New target products       : {result[0]:,}")
print(f"Average prior purchases  : {result[1]:,.2f}")
print(f"Median prior purchases   : {result[4]:,.2f}")
print(f"Minimum prior purchases  : {result[2]:,}")
print(f"Maximum prior purchases  : {result[3]:,}")


# ============================================================
# 6. GLOBAL POPULARITY COVERAGE
#
# Check how many new-to-customer products are among the
# globally most popular products.
# ============================================================

print("\n4. GLOBAL POPULARITY COVERAGE")
print("-" * 70)

result = con.execute("""
WITH ranked_products AS (

    SELECT

        product_id,

        total_purchases,

        ROW_NUMBER() OVER (
            ORDER BY total_purchases DESC
        ) AS popularity_rank

    FROM product_popularity
),

new_products AS (

    SELECT

        user_id,
        product_id

    FROM new_target_products
)

SELECT

    COUNT(*) AS new_target_products,

    SUM(
        CASE
            WHEN rp.popularity_rank <= 25
            THEN 1
            ELSE 0
        END
    ) AS top_25,

    SUM(
        CASE
            WHEN rp.popularity_rank <= 50
            THEN 1
            ELSE 0
        END
    ) AS top_50,

    SUM(
        CASE
            WHEN rp.popularity_rank <= 100
            THEN 1
            ELSE 0
        END
    ) AS top_100,

    SUM(
        CASE
            WHEN rp.popularity_rank <= 500
            THEN 1
            ELSE 0
        END
    ) AS top_500,

    SUM(
        CASE
            WHEN rp.popularity_rank <= 1000
            THEN 1
            ELSE 0
        END
    ) AS top_1000

FROM new_products np

JOIN ranked_products rp

    ON np.product_id = rp.product_id
""").fetchone()

total = result[0]

print(f"New target products : {total:,}")

print(
    f"Top 25 coverage     : "
    f"{result[1]:,} "
    f"({100 * result[1] / total:.2f}%)"
)

print(
    f"Top 50 coverage     : "
    f"{result[2]:,} "
    f"({100 * result[2] / total:.2f}%)"
)

print(
    f"Top 100 coverage    : "
    f"{result[3]:,} "
    f"({100 * result[3] / total:.2f}%)"
)

print(
    f"Top 500 coverage    : "
    f"{result[4]:,} "
    f"({100 * result[4] / total:.2f}%)"
)

print(
    f"Top 1000 coverage   : "
    f"{result[5]:,} "
    f"({100 * result[5] / total:.2f}%)"
)


# ============================================================
# 7. DEPARTMENT DISTRIBUTION
#
# Which departments contain the new-to-customer products?
# ============================================================

print("\n5. DEPARTMENTS OF NEW-TO-CUSTOMER PRODUCTS")
print("-" * 70)

result = con.execute("""
SELECT

    d.department,

    COUNT(*) AS new_products,

    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM new_target_products nt

JOIN products p
    ON nt.product_id = p.product_id

JOIN ml_departments d
    ON p.department_id = d.department_id

GROUP BY
    d.department

ORDER BY
    new_products DESC

LIMIT 15
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 8. DEPARTMENT-LEVEL PRODUCT POPULARITY
#
# This helps us understand whether department-level candidate
# generation could be useful.
# ============================================================

print("\n6. NEW PRODUCTS BY DEPARTMENT + POPULARITY")
print("-" * 70)

result = con.execute("""
SELECT

    d.department,

    COUNT(DISTINCT nt.product_id)
        AS unique_products,

    COUNT(*) AS target_occurrences,

    ROUND(
        AVG(pp.total_purchases),
        2
    ) AS avg_global_purchases,

    ROUND(
        MEDIAN(pp.total_purchases),
        2
    ) AS median_global_purchases

FROM new_target_products nt

JOIN products p
    ON nt.product_id = p.product_id

JOIN ml_departments d
    ON p.department_id = d.department_id

JOIN product_popularity pp
    ON nt.product_id = pp.product_id

GROUP BY
    d.department

ORDER BY
    target_occurrences DESC

LIMIT 15
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 9. TOP NEW-TO-CUSTOMER PRODUCTS
#
# Which products are most frequently purchased as "new"
# products by customers?
# ============================================================

print("\n7. MOST COMMON NEW-TO-CUSTOMER PRODUCTS")
print("-" * 70)

result = con.execute("""
SELECT

    p.product_name,

    COUNT(*) AS new_customer_purchases,

    pp.total_purchases AS global_prior_purchases,

    pp.unique_customers

FROM new_target_products nt

JOIN products p
    ON nt.product_id = p.product_id

JOIN product_popularity pp
    ON nt.product_id = pp.product_id

GROUP BY

    p.product_name,
    pp.total_purchases,
    pp.unique_customers

ORDER BY
    new_customer_purchases DESC

LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

con.close()