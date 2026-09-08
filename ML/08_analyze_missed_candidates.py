import duckdb
import time
import pandas as pd

DB_PATH = "data/processed/instacart.duckdb"
ORDERS_PATH = "data/raw/orders.csv"

con = duckdb.connect(DB_PATH)

print("=" * 90)
print("MISSED CANDIDATE ANALYSIS")
print("=" * 90)

start = time.time()

# ================================================================
# 1. Load orders with eval_set
# ================================================================

print("\n[1/10] Loading orders...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_orders AS
SELECT *
FROM read_csv_auto('{ORDERS_PATH}')
""")

# ================================================================
# 2. Target customers
# ================================================================

print("[2/10] Identifying target customers...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_customers AS
SELECT
    user_id,
    order_id AS target_order_id,
    order_number AS target_order_number
FROM ml_orders
WHERE eval_set = 'train'
""")

# ================================================================
# 3. Actual target products
# ================================================================

print("[3/10] Building actual target products...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_products AS
SELECT DISTINCT
    tc.user_id,
    tc.target_order_id,
    tc.target_order_number,
    op.product_id
FROM target_customers tc
JOIN order_products op
    ON tc.target_order_id = op.order_id
""")

target_count = con.execute("""
SELECT COUNT(*)
FROM target_products
""").fetchone()[0]

print(f"Target customer-product pairs: {target_count:,}")

# ================================================================
# 4. Historical customer-product purchases
# ================================================================

print("[4/10] Building historical customer-product purchases...")

con.execute("""
CREATE OR REPLACE TEMP TABLE historical_customer_products AS
SELECT
    tc.user_id,
    op.product_id,
    COUNT(*) AS previous_purchases,
    MAX(o.order_number) AS last_purchase_order
FROM target_customers tc
JOIN ml_orders o
    ON tc.user_id = o.user_id
JOIN order_products op
    ON o.order_id = op.order_id
WHERE o.eval_set = 'prior'
  AND o.order_number < tc.target_order_number
GROUP BY
    tc.user_id,
    op.product_id
""")

# ================================================================
# 5. Strategy B candidates
# ================================================================

print("[5/10] Rebuilding Strategy B candidates...")

# ----------------------------
# Global popularity
# ----------------------------

con.execute("""
CREATE OR REPLACE TEMP TABLE product_popularity AS
SELECT
    op.product_id,
    COUNT(*) AS prior_purchases,
    COUNT(DISTINCT o.user_id) AS prior_customers
FROM order_products op
JOIN ml_orders o
    ON op.order_id = o.order_id
WHERE o.eval_set = 'prior'
GROUP BY op.product_id
""")

# ----------------------------
# Historical candidates
# ----------------------------

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates_historical AS
SELECT DISTINCT
    user_id,
    product_id
FROM historical_customer_products
""")

# ----------------------------
# Global Top 100
# ----------------------------

con.execute("""
CREATE OR REPLACE TEMP TABLE global_products AS
SELECT
    product_id
FROM product_popularity
ORDER BY prior_purchases DESC
LIMIT 100
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates_global AS
SELECT
    tc.user_id,
    gp.product_id
FROM (
    SELECT DISTINCT user_id
    FROM target_customers
) tc
CROSS JOIN global_products gp
""")

# ================================================================
# 6. Customer department activity
# ================================================================

print("[6/10] Building customer department profiles...")

con.execute("""
CREATE OR REPLACE TEMP TABLE customer_departments AS
SELECT
    hcp.user_id,
    p.department_id,
    SUM(hcp.previous_purchases) AS department_purchases
FROM historical_customer_products hcp
JOIN products p
    ON hcp.product_id = p.product_id
GROUP BY
    hcp.user_id,
    p.department_id
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE top_customer_departments AS
SELECT
    user_id,
    department_id
FROM (
    SELECT
        user_id,
        department_id,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY department_purchases DESC
        ) AS rn
    FROM customer_departments
)
WHERE rn <= 3
""")

# ================================================================
# 7. Department candidates
# ================================================================

print("[7/10] Building department candidates...")

con.execute("""
CREATE OR REPLACE TEMP TABLE department_product_rank AS
SELECT
    p.department_id,
    p.product_id,
    pp.prior_purchases,
    ROW_NUMBER() OVER (
        PARTITION BY p.department_id
        ORDER BY pp.prior_purchases DESC
    ) AS department_rank
FROM products p
JOIN product_popularity pp
    ON p.product_id = pp.product_id
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates_department AS
SELECT
    tcd.user_id,
    dpr.product_id
FROM top_customer_departments tcd
JOIN department_product_rank dpr
    ON tcd.department_id = dpr.department_id
WHERE dpr.department_rank <= 15
""")

# ================================================================
# 8. Final Strategy B candidate set
# ================================================================

print("[8/10] Creating Strategy B candidate set...")

con.execute("""
CREATE OR REPLACE TEMP TABLE strategy_b AS
SELECT DISTINCT
    user_id,
    product_id
FROM (
    SELECT user_id, product_id
    FROM candidates_historical

    UNION ALL

    SELECT user_id, product_id
    FROM candidates_global

    UNION ALL

    SELECT user_id, product_id
    FROM candidates_department
)
""")

strategy_b_rows = con.execute("""
SELECT COUNT(*)
FROM strategy_b
""").fetchone()[0]

print(f"Strategy B candidates: {strategy_b_rows:,}")

# ================================================================
# 9. Identify missed target products
# ================================================================

print("[9/10] Identifying missed target products...")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed_products AS
SELECT
    tp.user_id,
    tp.product_id,
    tp.target_order_id,
    tp.target_order_number
FROM target_products tp
LEFT JOIN strategy_b sb
    ON tp.user_id = sb.user_id
   AND tp.product_id = sb.product_id
WHERE sb.product_id IS NULL
""")

missed_count = con.execute("""
SELECT COUNT(*)
FROM missed_products
""").fetchone()[0]

missed_percentage = missed_count / target_count * 100

print(f"Missed target pairs: {missed_count:,}")
print(f"Missed percentage: {missed_percentage:.2f}%")

# ================================================================
# 10. Enrich missed products
# ================================================================

print("[10/10] Analyzing missed products...")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed_enriched AS
SELECT
    mp.user_id,
    mp.product_id,
    mp.target_order_id,
    mp.target_order_number,

    p.product_name,
    p.aisle_id,
    p.department_id,

    pp.prior_purchases,
    pp.prior_customers,

    CASE
        WHEN hcp.product_id IS NOT NULL
        THEN 1
        ELSE 0
    END AS historically_purchased,

    COALESCE(hcp.previous_purchases, 0)
        AS previous_purchases,

    CASE
        WHEN tcd.department_id IS NOT NULL
        THEN 1
        ELSE 0
    END AS in_top_3_department

FROM missed_products mp

JOIN products p
    ON mp.product_id = p.product_id

LEFT JOIN product_popularity pp
    ON mp.product_id = pp.product_id

LEFT JOIN historical_customer_products hcp
    ON mp.user_id = hcp.user_id
   AND mp.product_id = hcp.product_id

LEFT JOIN top_customer_departments tcd
    ON mp.user_id = tcd.user_id
   AND p.department_id = tcd.department_id
""")

# ================================================================
# BASIC SUMMARY
# ================================================================

print("\n" + "=" * 90)
print("1. MISSED PRODUCT SUMMARY")
print("=" * 90)

summary = con.execute("""
SELECT
    COUNT(*) AS missed_pairs,

    COUNT(DISTINCT user_id) AS affected_customers,

    COUNT(DISTINCT product_id) AS distinct_products,

    AVG(prior_purchases) AS avg_global_purchases,

    MEDIAN(prior_purchases) AS median_global_purchases,

    AVG(prior_customers) AS avg_global_customers,

    MEDIAN(prior_customers) AS median_global_customers

FROM missed_enriched
""").fetchdf()

print(summary.to_string(index=False))

# ================================================================
# NEW VS EXISTING
# ================================================================

print("\n" + "=" * 90)
print("2. MISSED PRODUCTS: NEW VS PREVIOUSLY PURCHASED")
print("=" * 90)

new_existing = con.execute("""
SELECT
    CASE
        WHEN historically_purchased = 1
        THEN 'Previously purchased'
        ELSE 'New-to-customer'
    END AS product_type,

    COUNT(*) AS target_pairs,

    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM missed_enriched
GROUP BY 1
ORDER BY target_pairs DESC
""").fetchdf()

print(new_existing.to_string(index=False))

# ================================================================
# TOP 3 DEPARTMENT ANALYSIS
# ================================================================

print("\n" + "=" * 90)
print("3. MISSED PRODUCTS BY CUSTOMER'S TOP 3 DEPARTMENTS")
print("=" * 90)

department_status = con.execute("""
SELECT
    CASE
        WHEN in_top_3_department = 1
        THEN 'Inside top 3 department'
        ELSE 'Outside top 3 departments'
    END AS department_status,

    COUNT(*) AS target_pairs,

    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM missed_enriched
GROUP BY 1
ORDER BY target_pairs DESC
""").fetchdf()

print(department_status.to_string(index=False))

# ================================================================
# GLOBAL POPULARITY BUCKETS
# ================================================================

print("\n" + "=" * 90)
print("4. MISSED PRODUCTS BY GLOBAL POPULARITY")
print("=" * 90)

popularity = con.execute("""
SELECT
    CASE
        WHEN prior_purchases <= 10
            THEN '1-10'
        WHEN prior_purchases <= 50
            THEN '11-50'
        WHEN prior_purchases <= 100
            THEN '51-100'
        WHEN prior_purchases <= 500
            THEN '101-500'
        WHEN prior_purchases <= 1000
            THEN '501-1,000'
        WHEN prior_purchases <= 5000
            THEN '1,001-5,000'
        WHEN prior_purchases <= 10000
            THEN '5,001-10,000'
        ELSE '10,000+'
    END AS popularity_bucket,

    COUNT(*) AS target_pairs,

    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM missed_enriched
GROUP BY 1
ORDER BY
    MIN(prior_purchases)
""").fetchdf()

print(popularity.to_string(index=False))

# ================================================================
# TOP DEPARTMENTS
# ================================================================

print("\n" + "=" * 90)
print("5. MISSED PRODUCTS BY DEPARTMENT")
print("=" * 90)

departments = con.execute("""
SELECT
    p.department_id,
    d.department,

    COUNT(*) AS missed_target_pairs,

    COUNT(DISTINCT me.product_id) AS distinct_products,

    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM missed_enriched me
JOIN products p
    ON me.product_id = p.product_id
JOIN read_csv_auto('data/raw/departments.csv') d
    ON p.department_id = d.department_id

GROUP BY
    p.department_id,
    d.department

ORDER BY missed_target_pairs DESC
LIMIT 20
""").fetchdf()

print(departments.to_string(index=False))

# ================================================================
# TOP MISSED PRODUCTS
# ================================================================

print("\n" + "=" * 90)
print("6. TOP MISSED PRODUCTS")
print("=" * 90)

top_missed = con.execute("""
SELECT
    product_id,
    product_name,
    prior_purchases,
    prior_customers,
    COUNT(*) AS missed_customer_targets
FROM missed_enriched
GROUP BY
    product_id,
    product_name,
    prior_purchases,
    prior_customers
ORDER BY missed_customer_targets DESC
LIMIT 30
""").fetchdf()

print(top_missed.to_string(index=False))

# ================================================================
# MISSED PRODUCTS THAT ARE POPULAR
# ================================================================

print("\n" + "=" * 90)
print("7. POPULAR MISSED PRODUCTS")
print("=" * 90)

popular_missed = con.execute("""
SELECT
    product_id,
    product_name,
    prior_purchases,
    prior_customers,
    COUNT(*) AS missed_customer_targets
FROM missed_enriched
WHERE prior_purchases >= 1000
GROUP BY
    product_id,
    product_name,
    prior_purchases,
    prior_customers
ORDER BY missed_customer_targets DESC
LIMIT 30
""").fetchdf()

print(popular_missed.to_string(index=False))

# ================================================================
# SAVE DETAILED RESULTS
# ================================================================

output_path = "data/processed/missed_candidate_analysis.csv"

con.execute(f"""
COPY (
    SELECT *
    FROM missed_enriched
)
TO '{output_path}'
WITH (HEADER, DELIMITER ',')
""")

elapsed = time.time() - start

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)

print(f"Missed target pairs: {missed_count:,}")
print(f"Missed percentage: {missed_percentage:.2f}%")
print(f"Detailed results saved to: {output_path}")
print(f"Runtime: {elapsed:.2f} seconds")

con.close()