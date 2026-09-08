import duckdb
import time
import pandas as pd

DB_PATH = "data/processed/instacart.duckdb"
ORDERS_PATH = "data/raw/orders.csv"

con = duckdb.connect(DB_PATH)

print("=" * 90)
print("DEPARTMENT DEPTH CANDIDATE EXPERIMENT")
print("=" * 90)

start = time.time()

# ================================================================
# 1. Load orders
# ================================================================

print("\n[1/8] Loading orders...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_orders AS
SELECT *
FROM read_csv_auto('{ORDERS_PATH}')
""")

# ================================================================
# 2. Target customers
# ================================================================

print("[2/8] Identifying target customers...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_customers AS
SELECT
    user_id,
    order_id AS target_order_id,
    order_number AS target_order_number
FROM ml_orders
WHERE eval_set = 'train'
""")

target_customer_count = con.execute("""
SELECT COUNT(DISTINCT user_id)
FROM target_customers
""").fetchone()[0]

print(f"Target customers: {target_customer_count:,}")

# ================================================================
# 3. Target products
# ================================================================

print("[3/8] Building target products...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_products AS
SELECT DISTINCT
    tc.user_id,
    op.product_id
FROM target_customers tc
JOIN order_products op
    ON tc.target_order_id = op.order_id
""")

target_product_count = con.execute("""
SELECT COUNT(*)
FROM target_products
""").fetchone()[0]

print(f"Target customer-product pairs: {target_product_count:,}")

# ================================================================
# 4. Historical customer-product purchases
# ================================================================

print("[4/8] Building historical customer-product purchases...")

con.execute("""
CREATE OR REPLACE TEMP TABLE historical_customer_products AS
SELECT
    tc.user_id,
    op.product_id,
    COUNT(*) AS previous_purchases
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
# 5. Global product popularity
# ================================================================

print("[5/8] Calculating product popularity...")

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
# 6. Customer department activity
# ================================================================

print("[6/8] Building customer department profiles...")

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

# Top 3 departments
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
# 7. Rank products inside departments
# ================================================================

print("[7/8] Ranking products within departments...")

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
# 8. Fixed candidates: Historical + Global Top 100
# ================================================================

print("[8/8] Building fixed candidate sources...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates_historical AS
SELECT DISTINCT
    user_id,
    product_id
FROM historical_customer_products
""")

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
# Experiment function
# ================================================================

def run_experiment(depth):

    print("\n" + "=" * 90)
    print(f"TOP 3 DEPARTMENTS × TOP {depth} PRODUCTS")
    print("=" * 90)

    experiment_start = time.time()

    # ------------------------------------------------------------
    # Department candidates
    # ------------------------------------------------------------

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE candidates_department AS
    SELECT
        tcd.user_id,
        dpr.product_id
    FROM top_customer_departments tcd
    JOIN department_product_rank dpr
        ON tcd.department_id = dpr.department_id
    WHERE dpr.department_rank <= {depth}
    """)

    department_rows = con.execute("""
    SELECT COUNT(*)
    FROM candidates_department
    """).fetchone()[0]

    # ------------------------------------------------------------
    # Combined candidates
    # ------------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE experiment_candidates AS
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

    candidate_rows = con.execute("""
    SELECT COUNT(*)
    FROM experiment_candidates
    """).fetchone()[0]

    # ------------------------------------------------------------
    # Average / median candidates
    # ------------------------------------------------------------

    avg_candidates = con.execute("""
    SELECT AVG(candidate_count)
    FROM (
        SELECT
            user_id,
            COUNT(*) AS candidate_count
        FROM experiment_candidates
        GROUP BY user_id
    )
    """).fetchone()[0]

    median_candidates = con.execute("""
    SELECT MEDIAN(candidate_count)
    FROM (
        SELECT
            user_id,
            COUNT(*) AS candidate_count
        FROM experiment_candidates
        GROUP BY user_id
    )
    """).fetchone()[0]

    min_candidates, max_candidates = con.execute("""
    SELECT
        MIN(candidate_count),
        MAX(candidate_count)
    FROM (
        SELECT
            user_id,
            COUNT(*) AS candidate_count
        FROM experiment_candidates
        GROUP BY user_id
    )
    """).fetchone()

    # ------------------------------------------------------------
    # Target coverage
    # ------------------------------------------------------------

    covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN experiment_candidates ec
        ON tp.user_id = ec.user_id
       AND tp.product_id = ec.product_id
    """).fetchone()[0]

    coverage = covered / target_product_count * 100

    # ------------------------------------------------------------
    # Full customer coverage
    # ------------------------------------------------------------

    fully_covered = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT
            tp.user_id,
            COUNT(*) AS target_count,
            COUNT(ec.product_id) AS covered_count
        FROM target_products tp
        LEFT JOIN experiment_candidates ec
            ON tp.user_id = ec.user_id
           AND tp.product_id = ec.product_id
        GROUP BY tp.user_id
    )
    WHERE target_count = covered_count
    """).fetchone()[0]

    full_customer_coverage = (
        fully_covered / target_customer_count * 100
    )

    # ------------------------------------------------------------
    # Incremental department coverage
    # ------------------------------------------------------------

    department_covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN candidates_department cd
        ON tp.user_id = cd.user_id
       AND tp.product_id = cd.product_id
    """).fetchone()[0]

    department_coverage = (
        department_covered / target_product_count * 100
    )

    elapsed = time.time() - experiment_start

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    print(f"\nDepartment candidate rows:     {department_rows:,}")
    print(f"Total candidate rows:          {candidate_rows:,}")
    print(f"Average candidates/customer:   {avg_candidates:.2f}")
    print(f"Median candidates/customer:    {median_candidates:.0f}")
    print(f"Minimum candidates/customer:   {min_candidates:,}")
    print(f"Maximum candidates/customer:   {max_candidates:,}")

    print("\nCoverage")
    print("-" * 50)
    print(f"Overall target coverage:       {coverage:.2f}%")
    print(
        f"Fully covered customers:       "
        f"{fully_covered:,} / {target_customer_count:,} "
        f"({full_customer_coverage:.2f}%)"
    )
    print(f"Department standalone coverage:{department_coverage:.2f}%")

    print(f"\nRuntime: {elapsed:.2f} seconds")

    return {
        "department_depth": depth,
        "department_candidate_rows": department_rows,
        "total_candidate_rows": candidate_rows,
        "avg_candidates_per_customer": avg_candidates,
        "median_candidates_per_customer": median_candidates,
        "min_candidates_per_customer": min_candidates,
        "max_candidates_per_customer": max_candidates,
        "target_coverage": coverage,
        "fully_covered_customers": fully_covered,
        "full_customer_coverage": full_customer_coverage,
        "department_standalone_coverage": department_coverage,
        "runtime_seconds": elapsed,
    }


# ================================================================
# Run experiments
# ================================================================

results = []

for depth in [15, 30, 50, 100]:
    results.append(run_experiment(depth))

# ================================================================
# Final comparison
# ================================================================

print("\n\n" + "=" * 110)
print("FINAL DEPARTMENT DEPTH COMPARISON")
print("=" * 110)

print(
    f"{'Depth':<10}"
    f"{'Candidates':>16}"
    f"{'Avg/User':>12}"
    f"{'Median':>10}"
    f"{'Coverage':>12}"
    f"{'Full Cust.':>14}"
)

print("-" * 110)

for r in results:
    print(
        f"{r['department_depth']:<10}"
        f"{r['total_candidate_rows']:>16,}"
        f"{r['avg_candidates_per_customer']:>12.2f}"
        f"{r['median_candidates_per_customer']:>10.0f}"
        f"{r['target_coverage']:>11.2f}%"
        f"{r['full_customer_coverage']:>13.2f}%"
    )

# ================================================================
# Coverage gain
# ================================================================

baseline = results[0]["target_coverage"]

print("\nCoverage improvement over Top 3 × 15:")
print("-" * 60)

for r in results:
    gain = r["target_coverage"] - baseline

    print(
        f"Top 3 × {r['department_depth']:<3}"
        f" → {r['target_coverage']:.2f}% "
        f"(+{gain:.2f} pp)"
    )

# ================================================================
# Save
# ================================================================

output_path = "data/processed/department_depth_experiment.csv"

pd.DataFrame(results).to_csv(
    output_path,
    index=False
)

print("\nSaved results to:")
print(output_path)

print(f"\nTotal runtime: {time.time() - start:.2f} seconds")

con.close()

print("\n" + "=" * 90)
print("DEPARTMENT DEPTH EXPERIMENT COMPLETE")
print("=" * 90)