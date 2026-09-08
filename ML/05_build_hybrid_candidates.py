import duckdb
from pathlib import Path
import time

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

ORDERS_CSV = BASE_DIR / "data" / "raw" / "orders.csv"
DEPARTMENTS_CSV = BASE_DIR / "data" / "raw" / "departments.csv"

START_TIME = time.time()

# Candidate configuration
GLOBAL_TOP_K = 500
DEPARTMENT_TOP_K = 30


# ============================================================
# CONNECT
# ============================================================

con = duckdb.connect(str(DB_FILE))

print("=" * 70)
print("BUILDING HYBRID CANDIDATE SET")
print("=" * 70)


# ============================================================
# 1. LOAD ORDER METADATA
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
# 2. LOAD DEPARTMENTS
# ============================================================

print("\n2. LOADING DEPARTMENTS")
print("-" * 70)

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_departments AS

SELECT
    department_id,
    department

FROM read_csv_auto('{DEPARTMENTS_CSV}')
""")

print("Departments loaded.")


# ============================================================
# 3. CREATE PRODUCT POPULARITY
#
# Popularity is based ONLY on prior orders.
# ============================================================

print("\n3. CALCULATING PRODUCT POPULARITY")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE ml_product_popularity AS

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
# 4. GLOBAL TOP PRODUCTS
# ============================================================

print("\n4. BUILDING GLOBAL POPULARITY CANDIDATES")
print("-" * 70)

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_global_candidates AS

SELECT
    product_id

FROM (

    SELECT

        product_id,

        ROW_NUMBER() OVER (
            ORDER BY total_purchases DESC
        ) AS popularity_rank

    FROM ml_product_popularity

)

WHERE popularity_rank <= {GLOBAL_TOP_K}
""")

global_count = con.execute("""
SELECT COUNT(*)
FROM ml_global_candidates
""").fetchone()[0]

print(f"Global products added: {global_count}")


# ============================================================
# 5. CUSTOMER'S HISTORICAL PRODUCTS
#
# These are our repeat-purchase candidates.
# ============================================================

print("\n5. BUILDING HISTORICAL PRODUCT CANDIDATES")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE ml_history_candidates AS

SELECT DISTINCT

    cpf.user_id,
    cpf.product_id

FROM ml_customer_product_features cpf

JOIN ml_orders o
    ON cpf.user_id = o.user_id

WHERE o.eval_set = 'train'
""")

history_count = con.execute("""
SELECT COUNT(*)
FROM ml_history_candidates
""").fetchone()[0]

print(f"Historical candidates: {history_count:,}")


# ============================================================
# 6. CUSTOMER ACTIVE DEPARTMENTS
#
# We identify departments the customer has actually bought
# from historically.
# ============================================================

print("\n6. IDENTIFYING CUSTOMER ACTIVE DEPARTMENTS")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE ml_customer_departments AS

SELECT DISTINCT

    o.user_id,
    p.department_id

FROM order_products op

JOIN ml_orders o
    ON op.order_id = o.order_id

JOIN products p
    ON op.product_id = p.product_id

WHERE o.eval_set = 'prior'
""")


# ============================================================
# 7. DEPARTMENT POPULARITY
#
# Rank products within each department based on historical
# purchases.
# ============================================================

print("\n7. CALCULATING DEPARTMENT PRODUCT POPULARITY")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE ml_department_products AS

SELECT

    p.department_id,
    p.product_id,

    pp.total_purchases,

    ROW_NUMBER() OVER (
        PARTITION BY p.department_id
        ORDER BY
            pp.total_purchases DESC,
            p.product_id
    ) AS department_rank

FROM ml_product_popularity pp

JOIN products p
    ON pp.product_id = p.product_id
""")


# ============================================================
# 8. CUSTOMER-DEPARTMENT CANDIDATES
#
# For every customer's historically active department,
# add its top products.
# ============================================================

print("\n8. BUILDING DEPARTMENT CANDIDATES")
print("-" * 70)

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_department_candidates AS

SELECT DISTINCT

    cd.user_id,
    dp.product_id

FROM ml_customer_departments cd

JOIN ml_department_products dp
    ON cd.department_id = dp.department_id

WHERE dp.department_rank <= {DEPARTMENT_TOP_K}
""")

department_count = con.execute("""
SELECT COUNT(*)
FROM ml_department_candidates
""").fetchone()[0]

print(f"Department candidates: {department_count:,}")


# ============================================================
# 9. EXPAND GLOBAL CANDIDATES TO CUSTOMERS
# ============================================================

print("\n9. ADDING GLOBAL CANDIDATES")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE ml_global_customer_candidates AS

SELECT

    c.user_id,
    g.product_id

FROM (
    SELECT DISTINCT user_id
    FROM ml_orders
    WHERE eval_set = 'train'
) c

CROSS JOIN ml_global_candidates g
""")

global_customer_count = con.execute("""
SELECT COUNT(*)
FROM ml_global_customer_candidates
""").fetchone()[0]

print(
    f"Customer-global candidate rows: "
    f"{global_customer_count:,}"
)


# ============================================================
# 10. COMBINE ALL CANDIDATES
#
# UNION removes duplicates automatically.
# ============================================================

print("\n10. COMBINING CANDIDATE SOURCES")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_hybrid_candidates AS

SELECT
    user_id,
    product_id
FROM ml_history_candidates

UNION

SELECT
    user_id,
    product_id
FROM ml_department_candidates

UNION

SELECT
    user_id,
    product_id
FROM ml_global_customer_candidates
""")

hybrid_count = con.execute("""
SELECT COUNT(*)
FROM ml_hybrid_candidates
""").fetchone()[0]

hybrid_customers = con.execute("""
SELECT COUNT(DISTINCT user_id)
FROM ml_hybrid_candidates
""").fetchone()[0]

hybrid_products = con.execute("""
SELECT COUNT(DISTINCT product_id)
FROM ml_hybrid_candidates
""").fetchone()[0]

print(f"Hybrid candidate rows : {hybrid_count:,}")
print(f"Customers              : {hybrid_customers:,}")
print(f"Products               : {hybrid_products:,}")


# ============================================================
# 11. CANDIDATES PER CUSTOMER
# ============================================================

print("\n11. CANDIDATES PER CUSTOMER")
print("-" * 70)

result = con.execute("""
SELECT

    ROUND(AVG(candidate_count), 2)
        AS avg_candidates,

    ROUND(MEDIAN(candidate_count), 2)
        AS median_candidates,

    MIN(candidate_count)
        AS min_candidates,

    MAX(candidate_count)
        AS max_candidates

FROM (

    SELECT

        user_id,

        COUNT(*) AS candidate_count

    FROM ml_hybrid_candidates

    GROUP BY user_id
)
""").fetchone()

print(f"Average candidates : {result[0]}")
print(f"Median candidates  : {result[1]}")
print(f"Minimum candidates : {result[2]}")
print(f"Maximum candidates : {result[3]}")


# ============================================================
# 12. BUILD TARGET PRODUCTS
# ============================================================

print("\n12. BUILDING TARGET PRODUCTS")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE ml_target_products AS

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

print(f"Actual target products: {target_count:,}")


# ============================================================
# 13. CANDIDATE COVERAGE
# ============================================================

print("\n13. HYBRID CANDIDATE COVERAGE")
print("-" * 70)

result = con.execute("""
WITH coverage AS (

    SELECT

        tp.user_id,
        tp.product_id,

        CASE

            WHEN hc.product_id IS NOT NULL
            THEN 1

            ELSE 0

        END AS covered

    FROM ml_target_products tp

    LEFT JOIN ml_hybrid_candidates hc

        ON tp.user_id = hc.user_id
        AND tp.product_id = hc.product_id
)

SELECT

    COUNT(*) AS actual_products,

    SUM(covered) AS covered_products,

    COUNT(*) - SUM(covered)
        AS uncovered_products,

    ROUND(
        100.0 * SUM(covered) / COUNT(*),
        2
    ) AS coverage_percentage

FROM coverage
""").fetchone()

print(f"Actual products       : {result[0]:,}")
print(f"Covered products      : {result[1]:,}")
print(f"Uncovered products    : {result[2]:,}")
print(f"Coverage              : {result[3]}%")


# ============================================================
# 14. CUSTOMER FULL-COVERAGE
# ============================================================

print("\n14. CUSTOMER-LEVEL COVERAGE")
print("-" * 70)

result = con.execute("""
WITH target_counts AS (

    SELECT

        user_id,

        COUNT(*) AS target_products

    FROM ml_target_products

    GROUP BY user_id
),

covered_counts AS (

    SELECT

        tp.user_id,

        COUNT(*) AS covered_products

    FROM ml_target_products tp

    INNER JOIN ml_hybrid_candidates hc

        ON tp.user_id = hc.user_id
        AND tp.product_id = hc.product_id

    GROUP BY tp.user_id
)

SELECT

    COUNT(*) AS customers,

    SUM(

        CASE

            WHEN COALESCE(
                cc.covered_products,
                0
            ) = tc.target_products

            THEN 1

            ELSE 0

        END

    ) AS fully_covered_customers,

    ROUND(

        100.0 *

        SUM(

            CASE

                WHEN COALESCE(
                    cc.covered_products,
                    0
                ) = tc.target_products

                THEN 1

                ELSE 0

            END

        ) / COUNT(*),

        2

    ) AS full_coverage_percentage

FROM target_counts tc

LEFT JOIN covered_counts cc

    ON tc.user_id = cc.user_id
""").fetchone()

print(f"Customers               : {result[0]:,}")
print(f"Fully covered customers : {result[1]:,}")
print(f"Full coverage           : {result[2]}%")


# ============================================================
# 15. COVERAGE BY CANDIDATE SOURCE
#
# This tells us how many target products are recovered by
# each candidate strategy.
# ============================================================

print("\n15. COVERAGE BY CANDIDATE SOURCE")
print("-" * 70)

result = con.execute("""
WITH target_products AS (

    SELECT
        user_id,
        product_id
    FROM ml_target_products
),

history AS (

    SELECT
        tp.user_id,
        tp.product_id
    FROM target_products tp

    INNER JOIN ml_history_candidates hc

        ON tp.user_id = hc.user_id
        AND tp.product_id = hc.product_id
),

global_products AS (

    SELECT
        tp.user_id,
        tp.product_id
    FROM target_products tp

    INNER JOIN ml_global_customer_candidates gc

        ON tp.user_id = gc.user_id
        AND tp.product_id = gc.product_id
),

department_products AS (

    SELECT
        tp.user_id,
        tp.product_id
    FROM target_products tp

    INNER JOIN ml_department_candidates dc

        ON tp.user_id = dc.user_id
        AND tp.product_id = dc.product_id
)

SELECT

    (SELECT COUNT(*)
     FROM history)
        AS history_coverage,

    (SELECT COUNT(*)
     FROM global_products)
        AS global_coverage,

    (SELECT COUNT(*)
     FROM department_products)
        AS department_coverage
""").fetchone()

total_target = target_count

print(
    f"Historical candidates : "
    f"{result[0]:,} "
    f"({100 * result[0] / total_target:.2f}%)"
)

print(
    f"Global candidates     : "
    f"{result[1]:,} "
    f"({100 * result[1] / total_target:.2f}%)"
)

print(
    f"Department candidates : "
    f"{result[2]:,} "
    f"({100 * result[2] / total_target:.2f}%)"
)


# ============================================================
# 16. SAVE SUMMARY
# ============================================================

print("\n16. SAVING SUMMARY")
print("-" * 70)

con.execute("""
CREATE OR REPLACE TABLE ml_candidate_summary AS

SELECT

    'Historical' AS candidate_source,

    COUNT(*) AS candidate_rows

FROM ml_history_candidates

UNION ALL

SELECT

    'Global' AS candidate_source,

    COUNT(*) AS candidate_rows

FROM ml_global_customer_candidates

UNION ALL

SELECT

    'Department' AS candidate_source,

    COUNT(*) AS candidate_rows

FROM ml_department_candidates

UNION ALL

SELECT

    'Hybrid' AS candidate_source,

    COUNT(*) AS candidate_rows

FROM ml_hybrid_candidates
""")

print("Candidate summary saved.")


# ============================================================
# COMPLETE
# ============================================================

elapsed = time.time() - START_TIME

print("\n" + "=" * 70)
print("HYBRID CANDIDATE GENERATION COMPLETE")
print(f"Time taken: {elapsed / 60:.2f} minutes")
print("=" * 70)

con.close()