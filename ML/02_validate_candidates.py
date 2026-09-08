import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

con = duckdb.connect(str(DB_FILE))

print("=" * 70)
print("CANDIDATE COVERAGE & DATA VALIDATION")
print("=" * 70)


# ============================================================
# 1. Actual next-order products
# ============================================================

print("\n1. ACTUAL NEXT-ORDER PRODUCTS")
print("-" * 70)

result = con.execute("""
SELECT
    COUNT(*) AS target_products,
    COUNT(DISTINCT user_id) AS customers
FROM ml_target_products
""").fetchone()

print(f"Target product records : {result[0]:,}")
print(f"Customers              : {result[1]:,}")


# ============================================================
# 2. Candidate coverage
#
# What percentage of actual next-order products exist
# in our candidate set?
# ============================================================

print("\n2. CANDIDATE COVERAGE")
print("-" * 70)

result = con.execute("""
WITH coverage AS (

    SELECT
        tp.user_id,
        tp.product_id,

        CASE
            WHEN c.product_id IS NOT NULL
            THEN 1
            ELSE 0
        END AS covered

    FROM ml_target_products tp

    LEFT JOIN ml_candidates c
        ON tp.user_id = c.user_id
        AND tp.product_id = c.product_id
)

SELECT
    COUNT(*) AS actual_products,

    SUM(covered) AS covered_products,

    COUNT(*) - SUM(covered) AS uncovered_products,

    ROUND(
        100.0 * SUM(covered) / COUNT(*),
        2
    ) AS coverage_percentage

FROM coverage
""").fetchone()

print(f"Actual products       : {result[0]:,}")
print(f"Covered by candidates : {result[1]:,}")
print(f"Uncovered products    : {result[2]:,}")
print(f"Coverage              : {result[3]}%")


# ============================================================
# 3. Customer-level coverage
#
# For each customer, did we cover ALL products in their
# actual next order?
# ============================================================

print("\n3. CUSTOMER-LEVEL COVERAGE")
print("-" * 70)

result = con.execute("""
WITH target_counts AS (

    SELECT
        user_id,
        COUNT(*) AS actual_products

    FROM ml_target_products

    GROUP BY user_id
),

covered_counts AS (

    SELECT
        tp.user_id,
        COUNT(*) AS covered_products

    FROM ml_target_products tp

    INNER JOIN ml_candidates c
        ON tp.user_id = c.user_id
        AND tp.product_id = c.product_id

    GROUP BY tp.user_id
)

SELECT

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN COALESCE(cc.covered_products, 0)
                 = tc.actual_products
            THEN 1
            ELSE 0
        END
    ) AS fully_covered_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN COALESCE(cc.covered_products, 0)
                     = tc.actual_products
                THEN 1
                ELSE 0
            END
        )
        / COUNT(*),
        2
    ) AS full_coverage_percentage

FROM target_counts tc

LEFT JOIN covered_counts cc
    ON tc.user_id = cc.user_id
""").fetchone()

print(f"Customers                 : {result[0]:,}")
print(f"Fully covered customers   : {result[1]:,}")
print(f"Full coverage             : {result[2]}%")


# ============================================================
# 4. Positive rate by customer
# ============================================================

print("\n4. POSITIVE RATE BY CUSTOMER")
print("-" * 70)

result = con.execute("""
SELECT

    ROUND(
        AVG(positive_rate) * 100,
        2
    ) AS avg_customer_positive_rate,

    ROUND(
        MIN(positive_rate) * 100,
        2
    ) AS minimum_positive_rate,

    ROUND(
        MAX(positive_rate) * 100,
        2
    ) AS maximum_positive_rate

FROM (

    SELECT
        user_id,

        AVG(target) AS positive_rate

    FROM ml_training_data

    GROUP BY user_id

)
""").fetchone()

print(
    f"Average customer positive rate : "
    f"{result[0]}%"
)

print(
    f"Minimum customer positive rate : "
    f"{result[1]}%"
)

print(
    f"Maximum customer positive rate : "
    f"{result[2]}%"
)


# ============================================================
# 5. Feature sanity check
# ============================================================

print("\n5. FEATURE SANITY CHECK")
print("-" * 70)

result = con.execute("""
SELECT

    MIN(previous_purchases) AS min_previous_purchases,
    MAX(previous_purchases) AS max_previous_purchases,

    MIN(previous_orders) AS min_previous_orders,
    MAX(previous_orders) AS max_previous_orders,

    MIN(product_purchase_count) AS min_product_purchases,
    MAX(product_purchase_count) AS max_product_purchases,

    MIN(customer_product_reorder_rate)
        AS min_cp_reorder_rate,

    MAX(customer_product_reorder_rate)
        AS max_cp_reorder_rate

FROM ml_training_data
""").fetchone()

print(f"Previous purchases : {result[0]} → {result[1]}")
print(f"Previous orders    : {result[2]} → {result[3]}")
print(f"Product purchases  : {result[4]} → {result[5]}")
print(f"CP reorder rate    : {result[6]} → {result[7]}")


# ============================================================
# 6. Check target relationship with previous purchases
# ============================================================

print("\n6. TARGET vs PREVIOUS PURCHASES")
print("-" * 70)

result = con.execute("""
SELECT

    previous_purchases,

    COUNT(*) AS records,

    SUM(target) AS positive,

    ROUND(
        100.0 * AVG(target),
        2
    ) AS positive_rate

FROM ml_training_data

GROUP BY previous_purchases

ORDER BY previous_purchases

LIMIT 15
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 7. Check target relationship with reorder rate
# ============================================================

print("\n7. TARGET vs CUSTOMER-PRODUCT REORDER RATE")
print("-" * 70)

result = con.execute("""
SELECT

    CASE

        WHEN customer_product_reorder_rate = 0
            THEN '0%'

        WHEN customer_product_reorder_rate < 0.25
            THEN '1-24%'

        WHEN customer_product_reorder_rate < 0.50
            THEN '25-49%'

        WHEN customer_product_reorder_rate < 0.75
            THEN '50-74%'

        ELSE '75%+'

    END AS reorder_group,

    COUNT(*) AS records,

    SUM(target) AS positive,

    ROUND(
        100.0 * AVG(target),
        2
    ) AS positive_rate

FROM ml_training_data

GROUP BY reorder_group

ORDER BY reorder_group
""").fetchdf()

print(result.to_string(index=False))


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

con.close()