import duckdb
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent

DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"
ORDERS_CSV = BASE_DIR / "data" / "raw" / "orders.csv"

con = duckdb.connect(str(DB_FILE))

start = time.time()

print("=" * 70)
print("BUILDING TEMPORAL ML TRAINING DATASET")
print("=" * 70)


# ============================================================
# 1. Load raw order metadata including eval_set
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

result = con.execute("""
SELECT
    eval_set,
    COUNT(*) AS orders,
    COUNT(DISTINCT user_id) AS customers
FROM ml_orders
GROUP BY eval_set
ORDER BY eval_set
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 2. Customer-product historical features
#    ONLY PRIOR orders
# ============================================================

print("\n2. BUILDING CUSTOMER-PRODUCT FEATURES")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_customer_product_features AS

SELECT
    o.user_id,
    op.product_id,

    COUNT(*) AS previous_purchases,

    SUM(op.reordered) AS previous_reorders,

    ROUND(
        AVG(op.reordered),
        4
    ) AS customer_product_reorder_rate,

    MIN(o.order_number) AS first_purchase_order,

    MAX(o.order_number) AS last_purchase_order,

    ROUND(
        AVG(op.add_to_cart_order),
        2
    ) AS avg_cart_position

FROM order_products op

JOIN ml_orders o
    ON op.order_id = o.order_id

WHERE o.eval_set = 'prior'

GROUP BY
    o.user_id,
    op.product_id
""")

count = con.execute("""
SELECT COUNT(*)
FROM ml_customer_product_features
""").fetchone()[0]

print(f"Customer-product feature rows: {count:,}")


# ============================================================
# 3. Customer-level historical features
#    ONLY PRIOR orders
# ============================================================

print("\n3. BUILDING CUSTOMER FEATURES")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_order_sizes AS

SELECT
    o.user_id,
    o.order_id,
    COUNT(*) AS basket_size

FROM order_products op

JOIN ml_orders o
    ON op.order_id = o.order_id

WHERE o.eval_set = 'prior'

GROUP BY
    o.user_id,
    o.order_id
""")

con.execute("""
CREATE OR REPLACE TABLE ml_customer_features AS

WITH customer_stats AS (

    SELECT
        o.user_id,

        COUNT(DISTINCT o.order_id)
            AS previous_orders,

        ROUND(
            AVG(os.basket_size),
            2
        ) AS avg_basket_size,

        ROUND(
            AVG(o.days_since_prior_order),
            2
        ) AS avg_days_between_orders

    FROM ml_orders o

    JOIN ml_order_sizes os
        ON o.order_id = os.order_id

    WHERE o.eval_set = 'prior'

    GROUP BY
        o.user_id
),

reorder_stats AS (

    SELECT
        o.user_id,

        ROUND(
            AVG(op.reordered),
            4
        ) AS customer_reorder_rate

    FROM order_products op

    JOIN ml_orders o
        ON op.order_id = o.order_id

    WHERE o.eval_set = 'prior'

    GROUP BY
        o.user_id
)

SELECT
    cs.user_id,
    cs.previous_orders,
    cs.avg_basket_size,
    cs.avg_days_between_orders,
    rs.customer_reorder_rate

FROM customer_stats cs

LEFT JOIN reorder_stats rs
    ON cs.user_id = rs.user_id
""")

count = con.execute("""
SELECT COUNT(*)
FROM ml_customer_features
""").fetchone()[0]

print(f"Customer feature rows: {count:,}")


# ============================================================
# 4. Product-level historical features
#    ONLY PRIOR orders
# ============================================================

print("\n4. BUILDING PRODUCT FEATURES")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_product_features AS

SELECT
    op.product_id,

    COUNT(*) AS product_purchase_count,

    COUNT(DISTINCT o.user_id)
        AS product_unique_customers,

    ROUND(
        AVG(op.reordered),
        4
    ) AS product_reorder_rate

FROM order_products op

JOIN ml_orders o
    ON op.order_id = o.order_id

WHERE o.eval_set = 'prior'

GROUP BY
    op.product_id
""")

count = con.execute("""
SELECT COUNT(*)
FROM ml_product_features
""").fetchone()[0]

print(f"Product feature rows: {count:,}")


# ============================================================
# 5. Candidate generation
#
# Candidates = products previously purchased by customers
# who have a TRAIN order.
# ============================================================

print("\n5. BUILDING CANDIDATE SET")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_candidates AS

SELECT DISTINCT
    cpf.user_id,
    cpf.product_id

FROM ml_customer_product_features cpf

JOIN ml_orders target_order
    ON cpf.user_id = target_order.user_id

WHERE target_order.eval_set = 'train'
""")

candidate_count = con.execute("""
SELECT COUNT(*)
FROM ml_candidates
""").fetchone()[0]

customer_count = con.execute("""
SELECT COUNT(DISTINCT user_id)
FROM ml_candidates
""").fetchone()[0]

print(f"Candidate rows : {candidate_count:,}")
print(f"Customers      : {customer_count:,}")


# ============================================================
# 6. Target products
# ============================================================

print("\n6. BUILDING TARGET LABELS")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_target_products AS

SELECT DISTINCT
    o.user_id,
    op.product_id

FROM order_products op

JOIN ml_orders o
    ON op.order_id = o.order_id

WHERE o.eval_set = 'train'
""")

target_count = con.execute("""
SELECT COUNT(*)
FROM ml_target_products
""").fetchone()[0]

print(f"Actual next-order products: {target_count:,}")


# ============================================================
# 7. Build final ML training dataset
# ============================================================

print("\n7. BUILDING FINAL ML TRAINING DATA")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_training_data AS

SELECT

    c.user_id,
    c.product_id,

    -- Customer features
    cf.previous_orders,
    cf.avg_basket_size,
    cf.avg_days_between_orders,
    cf.customer_reorder_rate,

    -- Customer-product features
    cpf.previous_purchases,
    cpf.previous_reorders,
    cpf.customer_product_reorder_rate,
    cpf.first_purchase_order,
    cpf.last_purchase_order,
    cpf.avg_cart_position,

    -- Product features
    pf.product_purchase_count,
    pf.product_unique_customers,
    pf.product_reorder_rate,

    -- Product information
    p.product_name,
    p.aisle_id,
    p.department_id,

    -- Target
    CASE
        WHEN tp.product_id IS NOT NULL
        THEN 1
        ELSE 0
    END AS target

FROM ml_candidates c

LEFT JOIN ml_customer_features cf
    ON c.user_id = cf.user_id

LEFT JOIN ml_customer_product_features cpf
    ON c.user_id = cpf.user_id
    AND c.product_id = cpf.product_id

LEFT JOIN ml_product_features pf
    ON c.product_id = pf.product_id

LEFT JOIN products p
    ON c.product_id = p.product_id

LEFT JOIN ml_target_products tp
    ON c.user_id = tp.user_id
    AND c.product_id = tp.product_id
""")


# ============================================================
# 8. Final dataset summary
# ============================================================

print("\n8. FINAL DATASET SUMMARY")
print("-" * 70)

summary = con.execute("""
SELECT

    COUNT(*) AS rows,

    COUNT(DISTINCT user_id) AS customers,

    COUNT(DISTINCT product_id) AS products,

    SUM(target) AS positive_examples,

    COUNT(*) - SUM(target) AS negative_examples,

    ROUND(
        100.0 * SUM(target) / COUNT(*),
        2
    ) AS positive_rate

FROM ml_training_data
""").fetchone()

print(f"Rows              : {summary[0]:,}")
print(f"Customers         : {summary[1]:,}")
print(f"Products          : {summary[2]:,}")
print(f"Positive examples : {summary[3]:,}")
print(f"Negative examples : {summary[4]:,}")
print(f"Positive rate     : {summary[5]}%")


# ============================================================
# 9. NULL check
# ============================================================

print("\n9. NULL CHECK")
print("-" * 70)

null_check = con.execute("""
SELECT

    COUNT(*) AS total_rows,

    SUM(
        CASE WHEN previous_orders IS NULL
        THEN 1 ELSE 0 END
    ) AS null_previous_orders,

    SUM(
        CASE WHEN avg_basket_size IS NULL
        THEN 1 ELSE 0 END
    ) AS null_avg_basket,

    SUM(
        CASE WHEN previous_purchases IS NULL
        THEN 1 ELSE 0 END
    ) AS null_previous_purchases,

    SUM(
        CASE WHEN product_purchase_count IS NULL
        THEN 1 ELSE 0 END
    ) AS null_product_features

FROM ml_training_data
""").fetchone()

print(f"Total rows                  : {null_check[0]:,}")
print(f"NULL previous_orders        : {null_check[1]:,}")
print(f"NULL avg_basket_size        : {null_check[2]:,}")
print(f"NULL previous_purchases     : {null_check[3]:,}")
print(f"NULL product_purchase_count : {null_check[4]:,}")


# ============================================================
# 10. Sample records
# ============================================================

print("\n10. SAMPLE TRAINING DATA")
print("-" * 70)

sample = con.execute("""
SELECT *
FROM ml_training_data
LIMIT 10
""").fetchdf()

print(sample.to_string(index=False))


# ============================================================
# 11. Target distribution
# ============================================================

print("\n11. TARGET DISTRIBUTION")
print("-" * 70)

target_distribution = con.execute("""
SELECT

    target,

    COUNT(*) AS records,

    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM ml_training_data

GROUP BY target

ORDER BY target DESC
""").fetchdf()

print(target_distribution.to_string(index=False))


elapsed = time.time() - start

print("\n" + "=" * 70)
print("ML TRAINING DATASET COMPLETE")
print(f"Time taken: {elapsed / 60:.2f} minutes")
print("=" * 70)

con.close()