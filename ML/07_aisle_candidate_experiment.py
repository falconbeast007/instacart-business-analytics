import duckdb
import time
import pandas as pd

DB_PATH = "data/processed/instacart.duckdb"
ORDERS_PATH = "data/raw/orders.csv"

con = duckdb.connect(DB_PATH)

print("=" * 90)
print("AISLE-LEVEL CANDIDATE EXPERIMENT")
print("=" * 90)

# ================================================================
# 1. Load orders with eval_set
# ================================================================

print("\n[1/9] Loading orders...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_orders AS
SELECT *
FROM read_csv_auto('{ORDERS_PATH}')
""")

# ================================================================
# 2. Target customers
# ================================================================

print("[2/9] Identifying target customers...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_customers AS
SELECT
    order_id,
    user_id,
    order_number
FROM ml_orders
WHERE eval_set = 'train'
""")

target_customers = con.execute("""
SELECT COUNT(DISTINCT user_id)
FROM target_customers
""").fetchone()[0]

print(f"Target customers: {target_customers:,}")

# ================================================================
# 3. Historical customer-product purchases
# ================================================================

print("[3/9] Building historical customer-product purchases...")

con.execute("""
CREATE OR REPLACE TEMP TABLE historical_customer_products AS
SELECT
    o.user_id,
    op.product_id,
    COUNT(*) AS previous_purchases
FROM ml_orders o
JOIN order_products op
    ON o.order_id = op.order_id
JOIN target_customers tc
    ON o.user_id = tc.user_id
WHERE o.eval_set = 'prior'
  AND o.order_number < tc.order_number
GROUP BY
    o.user_id,
    op.product_id
""")

# ================================================================
# 4. Historical customer aisle activity
# ================================================================

print("[4/9] Building customer aisle activity...")

con.execute("""
CREATE OR REPLACE TEMP TABLE customer_aisles AS
SELECT
    hcp.user_id,
    p.aisle_id,
    SUM(hcp.previous_purchases) AS aisle_purchases
FROM historical_customer_products hcp
JOIN products p
    ON hcp.product_id = p.product_id
GROUP BY
    hcp.user_id,
    p.aisle_id
""")

# ================================================================
# 5. Product popularity
# ================================================================

print("[5/9] Calculating global product popularity...")

con.execute("""
CREATE OR REPLACE TEMP TABLE product_popularity AS
SELECT
    op.product_id,
    COUNT(*) AS prior_purchases
FROM order_products op
JOIN ml_orders o
    ON op.order_id = o.order_id
WHERE o.eval_set = 'prior'
GROUP BY op.product_id
""")

# ================================================================
# 6. Department product ranking
# ================================================================

print("[6/9] Ranking products within departments...")

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

# ================================================================
# 7. Aisle product ranking
# ================================================================

print("[7/9] Ranking products within aisles...")

con.execute("""
CREATE OR REPLACE TEMP TABLE aisle_product_rank AS
SELECT
    p.aisle_id,
    p.product_id,
    pp.prior_purchases,
    ROW_NUMBER() OVER (
        PARTITION BY p.aisle_id
        ORDER BY pp.prior_purchases DESC
    ) AS aisle_rank
FROM products p
JOIN product_popularity pp
    ON p.product_id = pp.product_id
""")

# ================================================================
# 8. Actual target products
# ================================================================

print("[8/9] Preparing actual target products...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_products AS
SELECT DISTINCT
    o.user_id,
    op.product_id
FROM ml_orders o
JOIN order_products op
    ON o.order_id = op.order_id
WHERE o.eval_set = 'train'
""")

target_product_count = con.execute("""
SELECT COUNT(*)
FROM target_products
""").fetchone()[0]

target_customer_count = con.execute("""
SELECT COUNT(DISTINCT user_id)
FROM target_products
""").fetchone()[0]

print(f"Target customer-product pairs: {target_product_count:,}")

# ================================================================
# 9. Base Strategy B + Aisle candidates
# ================================================================

print("[9/9] Building candidate sources...")

start = time.time()

# ------------------------------------------------
# Historical
# ------------------------------------------------

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates_historical AS
SELECT DISTINCT
    user_id,
    product_id
FROM historical_customer_products
""")

historical_rows = con.execute("""
SELECT COUNT(*)
FROM candidates_historical
""").fetchone()[0]

# ------------------------------------------------
# Global Top 100
# ------------------------------------------------

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

global_rows = con.execute("""
SELECT COUNT(*)
FROM candidates_global
""").fetchone()[0]

# ------------------------------------------------
# Top 3 customer departments
# ------------------------------------------------

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
    FROM (
        SELECT
            user_id,
            department_id,
            SUM(aisle_purchases) AS department_purchases
        FROM (
            SELECT
                ca.user_id,
                ca.aisle_id,
                ca.aisle_purchases,
                p.department_id
            FROM customer_aisles ca
            JOIN products p
                ON ca.aisle_id = p.aisle_id
        )
        GROUP BY
            user_id,
            department_id
    )
)
WHERE rn <= 3
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

department_rows = con.execute("""
SELECT COUNT(*)
FROM candidates_department
""").fetchone()[0]

# ------------------------------------------------
# Top 5 customer aisles × Top 10 products
# ------------------------------------------------

con.execute("""
CREATE OR REPLACE TEMP TABLE top_customer_aisles AS
SELECT
    user_id,
    aisle_id
FROM (
    SELECT
        user_id,
        aisle_id,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY aisle_purchases DESC
        ) AS rn
    FROM customer_aisles
)
WHERE rn <= 5
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates_aisle AS
SELECT
    tca.user_id,
    apr.product_id
FROM top_customer_aisles tca
JOIN aisle_product_rank apr
    ON tca.aisle_id = apr.aisle_id
WHERE apr.aisle_rank <= 10
""")

aisle_rows = con.execute("""
SELECT COUNT(*)
FROM candidates_aisle
""").fetchone()[0]

# ================================================================
# Evaluate each candidate source
# ================================================================

def evaluate_source(source_name, table_name):

    covered = con.execute(f"""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN {table_name} c
      ON tp.user_id = c.user_id
     AND tp.product_id = c.product_id
    """).fetchone()[0]

    coverage = covered / target_product_count * 100

    customers_covered = con.execute(f"""
    SELECT COUNT(DISTINCT tp.user_id)
    FROM target_products tp
    JOIN {table_name} c
      ON tp.user_id = c.user_id
     AND tp.product_id = c.product_id
    """).fetchone()[0]

    return {
        "source": source_name,
        "candidate_rows": con.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0],
        "covered_products": covered,
        "coverage": coverage,
        "customers_with_at_least_one_match": customers_covered,
    }


source_results = []

source_results.append(
    evaluate_source(
        "Historical",
        "candidates_historical"
    )
)

source_results.append(
    evaluate_source(
        "Global Top 100",
        "candidates_global"
    )
)

source_results.append(
    evaluate_source(
        "Top 3 Departments × 15",
        "candidates_department"
    )
)

source_results.append(
    evaluate_source(
        "Top 5 Aisles × 10",
        "candidates_aisle"
    )
)

# ================================================================
# BASE STRATEGY B
# ================================================================

print("\n" + "=" * 90)
print("BASE STRATEGY B")
print("=" * 90)

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

base_rows = con.execute("""
SELECT COUNT(*)
FROM strategy_b
""").fetchone()[0]

base_covered = con.execute("""
SELECT COUNT(*)
FROM target_products tp
JOIN strategy_b sc
  ON tp.user_id = sc.user_id
 AND tp.product_id = sc.product_id
""").fetchone()[0]

base_coverage = base_covered / target_product_count * 100

# ================================================================
# STRATEGY B + AISLE
# ================================================================

print("\n" + "=" * 90)
print("STRATEGY B + AISLE")
print("=" * 90)

con.execute("""
CREATE OR REPLACE TEMP TABLE strategy_b_aisle AS
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

    UNION ALL

    SELECT user_id, product_id
    FROM candidates_aisle
)
""")

combined_rows = con.execute("""
SELECT COUNT(*)
FROM strategy_b_aisle
""").fetchone()[0]

combined_covered = con.execute("""
SELECT COUNT(*)
FROM target_products tp
JOIN strategy_b_aisle sc
  ON tp.user_id = sc.user_id
 AND tp.product_id = sc.product_id
""").fetchone()[0]

combined_coverage = combined_covered / target_product_count * 100

# ================================================================
# Fully covered customers
# ================================================================

base_full_customers = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT
        tp.user_id,
        COUNT(*) AS target_count,
        COUNT(sc.product_id) AS covered_count
    FROM target_products tp
    LEFT JOIN strategy_b sc
      ON tp.user_id = sc.user_id
     AND tp.product_id = sc.product_id
    GROUP BY tp.user_id
)
WHERE target_count = covered_count
""").fetchone()[0]

combined_full_customers = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT
        tp.user_id,
        COUNT(*) AS target_count,
        COUNT(sc.product_id) AS covered_count
    FROM target_products tp
    LEFT JOIN strategy_b_aisle sc
      ON tp.user_id = sc.user_id
     AND tp.product_id = sc.product_id
    GROUP BY tp.user_id
)
WHERE target_count = covered_count
""").fetchone()[0]

# ================================================================
# Candidate/customer statistics
# ================================================================

base_avg = con.execute("""
SELECT AVG(cnt)
FROM (
    SELECT user_id, COUNT(*) cnt
    FROM strategy_b
    GROUP BY user_id
)
""").fetchone()[0]

base_median = con.execute("""
SELECT MEDIAN(cnt)
FROM (
    SELECT user_id, COUNT(*) cnt
    FROM strategy_b
    GROUP BY user_id
)
""").fetchone()[0]

combined_avg = con.execute("""
SELECT AVG(cnt)
FROM (
    SELECT user_id, COUNT(*) cnt
    FROM strategy_b_aisle
    GROUP BY user_id
)
""").fetchone()[0]

combined_median = con.execute("""
SELECT MEDIAN(cnt)
FROM (
    SELECT user_id, COUNT(*) cnt
    FROM strategy_b_aisle
    GROUP BY user_id
)
""").fetchone()[0]

elapsed = time.time() - start

# ================================================================
# RESULTS
# ================================================================

print("\nCandidate source contribution")
print("-" * 90)

for r in source_results:
    print(
        f"{r['source']:<35}"
        f"Rows: {r['candidate_rows']:>12,}    "
        f"Coverage: {r['coverage']:>7.2f}%"
    )

print("\n" + "=" * 90)
print("BASE B VS B + AISLE")
print("=" * 90)

print(
    f"{'Metric':<35}"
    f"{'Strategy B':>20}"
    f"{'B + Aisle':>20}"
)

print("-" * 90)

print(
    f"{'Candidate rows':<35}"
    f"{base_rows:>20,}"
    f"{combined_rows:>20,}"
)

print(
    f"{'Average candidates/customer':<35}"
    f"{base_avg:>20.2f}"
    f"{combined_avg:>20.2f}"
)

print(
    f"{'Median candidates/customer':<35}"
    f"{base_median:>20.0f}"
    f"{combined_median:>20.0f}"
)

print(
    f"{'Target coverage':<35}"
    f"{base_coverage:>19.2f}%"
    f"{combined_coverage:>19.2f}%"
)

print(
    f"{'Fully covered customers':<35}"
    f"{base_full_customers:>20,}"
    f"{combined_full_customers:>20,}"
)

print(
    f"{'Full customer coverage':<35}"
    f"{base_full_customers / target_customer_count * 100:>19.2f}%"
    f"{combined_full_customers / target_customer_count * 100:>19.2f}%"
)

print(
    f"{'Additional coverage':<35}"
    f"{'—':>20}"
    f"{combined_coverage - base_coverage:>19.2f}%"
)

print(
    f"{'Additional candidates':<35}"
    f"{'—':>20}"
    f"{combined_rows - base_rows:>20,}"
)

print(f"\nRuntime: {elapsed:.2f} seconds")

# ================================================================
# SAVE RESULTS
# ================================================================

comparison = pd.DataFrame([
    {
        "strategy": "B",
        "candidate_rows": base_rows,
        "avg_candidates_per_customer": base_avg,
        "median_candidates_per_customer": base_median,
        "target_coverage": base_coverage,
        "fully_covered_customers": base_full_customers,
        "full_customer_coverage": (
            base_full_customers / target_customer_count * 100
        ),
    },
    {
        "strategy": "B + Aisle",
        "candidate_rows": combined_rows,
        "avg_candidates_per_customer": combined_avg,
        "median_candidates_per_customer": combined_median,
        "target_coverage": combined_coverage,
        "fully_covered_customers": combined_full_customers,
        "full_customer_coverage": (
            combined_full_customers / target_customer_count * 100
        ),
    }
])

output_path = "data/processed/aisle_candidate_experiment.csv"
comparison.to_csv(output_path, index=False)

print("\nSaved results to:")
print(output_path)

con.close()

print("\n" + "=" * 90)
print("AISLE EXPERIMENT COMPLETE")
print("=" * 90)