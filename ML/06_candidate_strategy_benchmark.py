import duckdb
import time
import os

DB_PATH = "data/processed/instacart.duckdb"
ORDERS_PATH = "data/raw/orders.csv"

con = duckdb.connect(DB_PATH)

print("=" * 80)
print("CANDIDATE STRATEGY BENCHMARK")
print("=" * 80)

# -------------------------------------------------------------------
# 1. Recreate orders with eval_set
# -------------------------------------------------------------------

print("\n[1/7] Loading orders with eval_set...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_orders AS
SELECT *
FROM read_csv_auto('{ORDERS_PATH}')
""")

# -------------------------------------------------------------------
# 2. Target customers
# -------------------------------------------------------------------

print("[2/7] Identifying target customers...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_customers AS
SELECT
    order_id,
    user_id,
    order_number
FROM ml_orders
WHERE eval_set = 'train'
""")

target_customer_count = con.execute("""
SELECT COUNT(DISTINCT user_id)
FROM target_customers
""").fetchone()[0]

print(f"Target customers: {target_customer_count:,}")

# -------------------------------------------------------------------
# 3. Historical customer-product purchases
# -------------------------------------------------------------------

print("[3/7] Building historical customer-product data...")

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

# -------------------------------------------------------------------
# 4. Historical customer department activity
# -------------------------------------------------------------------

print("[4/7] Building customer department activity...")

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

# -------------------------------------------------------------------
# 5. Product popularity
# -------------------------------------------------------------------

print("[5/7] Calculating product popularity...")

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

# -------------------------------------------------------------------
# 6. Department product ranking
# -------------------------------------------------------------------

print("[6/7] Ranking products within departments...")

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

# -------------------------------------------------------------------
# 7. Actual target products
# -------------------------------------------------------------------

print("[7/7] Preparing target products...")

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

target_products = con.execute("""
SELECT COUNT(*)
FROM target_products
""").fetchone()[0]

print(f"Actual target customer-product pairs: {target_products:,}")

# ===================================================================
# BENCHMARK FUNCTION
# ===================================================================

def run_strategy(name, global_k, department_k, products_per_department):

    print("\n" + "=" * 80)
    print(f"STRATEGY {name}")
    print("=" * 80)

    start = time.time()

    # ---------------------------------------------------------------
    # Historical candidates
    # ---------------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE candidates_historical AS
    SELECT
        user_id,
        product_id,
        'historical' AS source
    FROM historical_customer_products
    """)

    # ---------------------------------------------------------------
    # Global candidates
    # ---------------------------------------------------------------

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE global_products AS
    SELECT
        product_id
    FROM product_popularity
    ORDER BY prior_purchases DESC
    LIMIT {global_k}
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE candidates_global AS
    SELECT
        tc.user_id,
        gp.product_id,
        'global' AS source
    FROM (
        SELECT DISTINCT user_id
        FROM target_customers
    ) tc
    CROSS JOIN global_products gp
    """)

    # ---------------------------------------------------------------
    # Top customer departments
    # ---------------------------------------------------------------

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE top_customer_departments AS
    SELECT
        user_id,
        department_id,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY department_purchases DESC
        ) AS department_rank
    FROM customer_departments
    QUALIFY department_rank <= {department_k}
    """)

    # ---------------------------------------------------------------
    # Department candidates
    # ---------------------------------------------------------------

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE candidates_department AS
    SELECT
        tcd.user_id,
        dpr.product_id,
        'department' AS source
    FROM top_customer_departments tcd
    JOIN department_product_rank dpr
        ON tcd.department_id = dpr.department_id
    WHERE dpr.department_rank <= {products_per_department}
    """)

    # ---------------------------------------------------------------
    # Combine candidates
    # ---------------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE strategy_candidates AS

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

    # ---------------------------------------------------------------
    # Candidate statistics
    # ---------------------------------------------------------------

    candidate_rows = con.execute("""
    SELECT COUNT(*)
    FROM strategy_candidates
    """).fetchone()[0]

    avg_candidates = con.execute("""
    SELECT AVG(candidate_count)
    FROM (
        SELECT
            user_id,
            COUNT(*) AS candidate_count
        FROM strategy_candidates
        GROUP BY user_id
    )
    """).fetchone()[0]

    median_candidates = con.execute("""
    SELECT MEDIAN(candidate_count)
    FROM (
        SELECT
            user_id,
            COUNT(*) AS candidate_count
        FROM strategy_candidates
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
        FROM strategy_candidates
        GROUP BY user_id
    )
    """).fetchone()

    # ---------------------------------------------------------------
    # Overall coverage
    # ---------------------------------------------------------------

    covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN strategy_candidates sc
      ON tp.user_id = sc.user_id
     AND tp.product_id = sc.product_id
    """).fetchone()[0]

    coverage = covered / target_products * 100

    # ---------------------------------------------------------------
    # Customer-level coverage
    # ---------------------------------------------------------------

    customers_with_targets = con.execute("""
    SELECT COUNT(DISTINCT user_id)
    FROM target_products
    """).fetchone()[0]

    fully_covered = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT
            tp.user_id,
            COUNT(*) AS target_count,
            COUNT(sc.product_id) AS covered_count
        FROM target_products tp
        LEFT JOIN strategy_candidates sc
          ON tp.user_id = sc.user_id
         AND tp.product_id = sc.product_id
        GROUP BY tp.user_id
    )
    WHERE target_count = covered_count
    """).fetchone()[0]

    full_customer_coverage = (
        fully_covered / customers_with_targets * 100
    )

    # ---------------------------------------------------------------
    # Source coverage
    # ---------------------------------------------------------------

    historical_covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN candidates_historical c
      ON tp.user_id = c.user_id
     AND tp.product_id = c.product_id
    """).fetchone()[0]

    global_covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN candidates_global c
      ON tp.user_id = c.user_id
     AND tp.product_id = c.product_id
    """).fetchone()[0]

    department_covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN candidates_department c
      ON tp.user_id = c.user_id
     AND tp.product_id = c.product_id
    """).fetchone()[0]

    elapsed = time.time() - start

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print(f"\nGlobal top K:                 {global_k}")
    print(f"Top customer departments:    {department_k}")
    print(f"Products per department:     {products_per_department}")

    print("\nCandidate volume")
    print("-" * 40)
    print(f"Total candidate rows:        {candidate_rows:,}")
    print(f"Average candidates/customer: {avg_candidates:.2f}")
    print(f"Median candidates/customer:  {median_candidates:.2f}")
    print(f"Minimum candidates/customer: {min_candidates:,}")
    print(f"Maximum candidates/customer: {max_candidates:,}")

    print("\nCoverage")
    print("-" * 40)
    print(f"Overall coverage:             {coverage:.2f}%")
    print(
        f"Fully covered customers:      "
        f"{fully_covered:,} / {customers_with_targets:,} "
        f"({full_customer_coverage:.2f}%)"
    )

    print("\nSource coverage")
    print("-" * 40)
    print(
        f"Historical:                   "
        f"{historical_covered:,} "
        f"({historical_covered / target_products * 100:.2f}%)"
    )
    print(
        f"Global:                       "
        f"{global_covered:,} "
        f"({global_covered / target_products * 100:.2f}%)"
    )
    print(
        f"Department:                   "
        f"{department_covered:,} "
        f"({department_covered / target_products * 100:.2f}%)"
    )

    print(f"\nRuntime: {elapsed:.2f} seconds")

    return {
        "strategy": name,
        "global_k": global_k,
        "department_k": department_k,
        "products_per_department": products_per_department,
        "candidate_rows": candidate_rows,
        "avg_candidates": avg_candidates,
        "median_candidates": median_candidates,
        "coverage": coverage,
        "fully_covered_customers": fully_covered,
        "full_customer_coverage": full_customer_coverage,
        "historical_coverage": historical_covered / target_products * 100,
        "global_coverage": global_covered / target_products * 100,
        "department_coverage": department_covered / target_products * 100,
        "runtime_seconds": elapsed,
    }


# ===================================================================
# RUN BENCHMARKS
# ===================================================================

results = []

results.append(
    run_strategy(
        "A",
        global_k=50,
        department_k=2,
        products_per_department=10,
    )
)

results.append(
    run_strategy(
        "B",
        global_k=100,
        department_k=3,
        products_per_department=15,
    )
)

results.append(
    run_strategy(
        "C",
        global_k=200,
        department_k=3,
        products_per_department=20,
    )
)

# ===================================================================
# FINAL COMPARISON
# ===================================================================

print("\n\n" + "=" * 100)
print("FINAL STRATEGY COMPARISON")
print("=" * 100)

print(
    f"{'Strategy':<10}"
    f"{'Candidates':>15}"
    f"{'Avg/User':>12}"
    f"{'Median':>10}"
    f"{'Coverage':>12}"
    f"{'Full Cust.':>14}"
)

print("-" * 100)

for r in results:
    print(
        f"{r['strategy']:<10}"
        f"{r['candidate_rows']:>15,}"
        f"{r['avg_candidates']:>12.2f}"
        f"{r['median_candidates']:>10.0f}"
        f"{r['coverage']:>11.2f}%"
        f"{r['full_customer_coverage']:>13.2f}%"
    )

# Save benchmark results
import pandas as pd

results_df = pd.DataFrame(results)

output_path = "data/processed/candidate_strategy_benchmark.csv"
results_df.to_csv(output_path, index=False)

print("\nSaved benchmark results to:")
print(output_path)

con.close()

print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)