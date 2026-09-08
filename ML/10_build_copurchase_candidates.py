import duckdb
import time
import pandas as pd

DB_PATH = "data/processed/instacart.duckdb"
ORDERS_PATH = "data/raw/orders.csv"

con = duckdb.connect(DB_PATH)

print("=" * 100)
print("CO-PURCHASE CANDIDATE EXPERIMENT")
print("=" * 100)

start = time.time()

# ================================================================
# 1. Load orders with eval_set
# ================================================================

print("\n[1/11] Loading orders...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ml_orders AS
SELECT *
FROM read_csv_auto('{ORDERS_PATH}')
""")

# ================================================================
# 2. Target customers
# ================================================================

print("[2/11] Identifying target customers...")

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

print("[3/11] Building target products...")

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

print("[4/11] Building historical customer-product purchases...")

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

historical_rows = con.execute("""
SELECT COUNT(*)
FROM historical_customer_products
""").fetchone()[0]

print(f"Historical customer-product rows: {historical_rows:,}")

# ================================================================
# 5. Build customer-product pairs from PRIOR history
# ================================================================

print("\n[5/11] Building historical customer-product pairs...")

# A customer is counted once for a product pair,
# regardless of how many orders contained the pair.
#
# This prevents heavy buyers / repeated orders from dominating
# the co-purchase relationship.

con.execute("""
CREATE OR REPLACE TEMP TABLE customer_product_presence AS
SELECT DISTINCT
    o.user_id,
    op.product_id
FROM target_customers tc
JOIN ml_orders o
    ON tc.user_id = o.user_id
JOIN order_products op
    ON o.order_id = op.order_id
WHERE o.eval_set = 'prior'
  AND o.order_number < tc.target_order_number
""")

presence_rows = con.execute("""
SELECT COUNT(*)
FROM customer_product_presence
""").fetchone()[0]

print(f"Customer-product presence rows: {presence_rows:,}")

# ================================================================
# 6. Build product-product co-purchase relationships
# ================================================================

print("\n[6/11] Building product-product co-purchase relationships...")

relationship_start = time.time()

con.execute("""
CREATE OR REPLACE TEMP TABLE product_pair_counts AS
SELECT
    a.product_id AS product_a,
    b.product_id AS product_b,
    COUNT(*) AS co_customers
FROM customer_product_presence a
JOIN customer_product_presence b
    ON a.user_id = b.user_id
   AND a.product_id < b.product_id
GROUP BY
    a.product_id,
    b.product_id
""")

pair_count = con.execute("""
SELECT COUNT(*)
FROM product_pair_counts
""").fetchone()[0]

print(f"Unique product pairs: {pair_count:,}")
print(
    f"Relationship generation time: "
    f"{time.time() - relationship_start:.2f} seconds"
)

# ================================================================
# 7. Calculate directional association metrics
# ================================================================

print("\n[7/11] Calculating association metrics...")

con.execute("""
CREATE OR REPLACE TEMP TABLE product_customer_counts AS
SELECT
    product_id,
    COUNT(*) AS product_customers
FROM customer_product_presence
GROUP BY product_id
""")

# Convert undirected pairs into directional relationships:
#
# A -> B
# B -> A
#
# confidence = P(B | A)
#
# lift = P(B | A) / P(B)
#
# We use confidence + lift to avoid simply recommending globally
# popular products.

con.execute("""
CREATE OR REPLACE TEMP TABLE copurchase_relationships AS

SELECT
    ppc.product_a,
    ppc.product_b,

    ppc.co_customers,

    a.product_customers AS customers_a,
    b.product_customers AS customers_b,

    ppc.co_customers * 1.0
        / NULLIF(a.product_customers, 0)
        AS confidence_a_to_b,

    ppc.co_customers * 1.0
        / NULLIF(b.product_customers, 0)
        AS confidence_b_to_a

FROM product_pair_counts ppc

JOIN product_customer_counts a
    ON ppc.product_a = a.product_id

JOIN product_customer_counts b
    ON ppc.product_b = b.product_id
""")

# ================================================================
# 8. Create directional relationships
# ================================================================

print("[8/11] Creating directional product relationships...")

con.execute("""
CREATE OR REPLACE TEMP TABLE directional_copurchase AS

SELECT
    product_a AS source_product,
    product_b AS candidate_product,
    co_customers,
    confidence_a_to_b AS confidence
FROM copurchase_relationships

UNION ALL

SELECT
    product_b AS source_product,
    product_a AS candidate_product,
    co_customers,
    confidence_b_to_a AS confidence
FROM copurchase_relationships
""")

# ================================================================
# 9. Keep strong relationships
# ================================================================

print("[9/11] Filtering co-purchase relationships...")

# We require at least 3 shared customers.
#
# We also rank relationships using:
#   co_customers
#   confidence
#
# Confidence is especially useful because:
#
#  A -> B = "Among customers who bought A,
#             how many also bought B?"
#
# We will later rank candidate products for each customer.

con.execute("""
CREATE OR REPLACE TEMP TABLE ranked_copurchase AS
SELECT
    source_product,
    candidate_product,
    co_customers,
    confidence,

    ROW_NUMBER() OVER (
        PARTITION BY source_product
        ORDER BY
            confidence DESC,
            co_customers DESC,
            candidate_product
    ) AS relationship_rank

FROM directional_copurchase

WHERE co_customers >= 3
""")

relationship_count = con.execute("""
SELECT COUNT(*)
FROM ranked_copurchase
""").fetchone()[0]

print(f"Directional relationships: {relationship_count:,}")

# ================================================================
# 10. Base Strategy B
# ================================================================

print("\n[10/11] Building Strategy B baseline...")

# Historical candidates
con.execute("""
CREATE OR REPLACE TEMP TABLE candidates_historical AS
SELECT DISTINCT
    user_id,
    product_id
FROM historical_customer_products
""")

# Global popularity
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

# Customer departments
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

# Department popularity
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
JOIN strategy_b sb
    ON tp.user_id = sb.user_id
   AND tp.product_id = sb.product_id
""").fetchone()[0]

base_coverage = base_covered / target_product_count * 100

print(f"Strategy B candidates: {base_rows:,}")
print(f"Strategy B coverage: {base_coverage:.2f}%")

# ================================================================
# 11. Generate co-purchase candidates
# ================================================================

print("\n[11/11] Generating co-purchase candidates...")

def run_experiment(top_n):

    print("\n" + "=" * 90)
    print(f"CO-PURCHASE TOP {top_n} EXPERIMENT")
    print("=" * 90)

    exp_start = time.time()

    # For each customer's historical product,
    # retrieve its top N associated products.

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE copurchase_candidates AS
    SELECT DISTINCT
        hcp.user_id,
        rc.candidate_product AS product_id
    FROM historical_customer_products hcp

    JOIN ranked_copurchase rc
        ON hcp.product_id = rc.source_product

    WHERE rc.relationship_rank <= {top_n}

      -- Don't generate products already purchased
      -- because those are already covered by historical candidates.
      AND NOT EXISTS (
          SELECT 1
          FROM historical_customer_products existing
          WHERE existing.user_id = hcp.user_id
            AND existing.product_id = rc.candidate_product
      )
    """)

    copurchase_rows = con.execute("""
    SELECT COUNT(*)
    FROM copurchase_candidates
    """).fetchone()[0]

    # ------------------------------------------------------------
    # Combine with Strategy B
    # ------------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE combined_candidates AS
    SELECT DISTINCT
        user_id,
        product_id
    FROM (
        SELECT user_id, product_id
        FROM strategy_b

        UNION ALL

        SELECT user_id, product_id
        FROM copurchase_candidates
    )
    """)

    combined_rows = con.execute("""
    SELECT COUNT(*)
    FROM combined_candidates
    """).fetchone()[0]

    # ------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------

    covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN combined_candidates cc
        ON tp.user_id = cc.user_id
       AND tp.product_id = cc.product_id
    """).fetchone()[0]

    coverage = covered / target_product_count * 100

    incremental_coverage = coverage - base_coverage

    # ------------------------------------------------------------
    # Fully covered customers
    # ------------------------------------------------------------

    fully_covered = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT
            tp.user_id,
            COUNT(*) AS target_count,
            COUNT(cc.product_id) AS covered_count
        FROM target_products tp
        LEFT JOIN combined_candidates cc
            ON tp.user_id = cc.user_id
           AND tp.product_id = cc.product_id
        GROUP BY tp.user_id
    )
    WHERE target_count = covered_count
    """).fetchone()[0]

    full_customer_coverage = (
        fully_covered / target_customer_count * 100
    )

    # ------------------------------------------------------------
    # Candidate statistics
    # ------------------------------------------------------------

    avg_candidates = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            user_id,
            COUNT(*) AS cnt
        FROM combined_candidates
        GROUP BY user_id
    )
    """).fetchone()[0]

    median_candidates = con.execute("""
    SELECT MEDIAN(cnt)
    FROM (
        SELECT
            user_id,
            COUNT(*) AS cnt
        FROM combined_candidates
        GROUP BY user_id
    )
    """).fetchone()[0]

    # ------------------------------------------------------------
    # Incremental coverage from co-purchase alone
    # ------------------------------------------------------------

    copurchase_covered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp
    JOIN copurchase_candidates cp
        ON tp.user_id = cp.user_id
       AND tp.product_id = cp.product_id
    """).fetchone()[0]

    copurchase_only_coverage = (
        copurchase_covered / target_product_count * 100
    )

    # ------------------------------------------------------------
    # How many missed Strategy B targets are recovered?
    # ------------------------------------------------------------

    incremental_recovered = con.execute("""
    SELECT COUNT(*)
    FROM target_products tp

    LEFT JOIN strategy_b sb
        ON tp.user_id = sb.user_id
       AND tp.product_id = sb.product_id

    JOIN copurchase_candidates cp
        ON tp.user_id = cp.user_id
       AND tp.product_id = cp.product_id

    WHERE sb.product_id IS NULL
    """).fetchone()[0]

    elapsed = time.time() - exp_start

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    print(f"\nCo-purchase candidate rows:     {copurchase_rows:,}")
    print(f"Combined candidate rows:        {combined_rows:,}")
    print(f"Average candidates/customer:    {avg_candidates:.2f}")
    print(f"Median candidates/customer:     {median_candidates:.0f}")

    print("\nCoverage")
    print("-" * 60)
    print(f"Co-purchase standalone coverage:{copurchase_only_coverage:.2f}%")
    print(f"Combined coverage:              {coverage:.2f}%")
    print(f"Incremental coverage:           +{incremental_coverage:.2f} pp")

    print(
        f"Missed Strategy B recovered:    "
        f"{incremental_recovered:,}"
    )

    print(
        f"Fully covered customers:        "
        f"{fully_covered:,} / {target_customer_count:,} "
        f"({full_customer_coverage:.2f}%)"
    )

    print(f"\nRuntime: {elapsed:.2f} seconds")

    return {
        "top_n": top_n,
        "copurchase_candidate_rows": copurchase_rows,
        "combined_candidate_rows": combined_rows,
        "avg_candidates_per_customer": avg_candidates,
        "median_candidates_per_customer": median_candidates,
        "copurchase_standalone_coverage": copurchase_only_coverage,
        "combined_coverage": coverage,
        "incremental_coverage": incremental_coverage,
        "incremental_recovered_targets": incremental_recovered,
        "fully_covered_customers": fully_covered,
        "full_customer_coverage": full_customer_coverage,
        "runtime_seconds": elapsed,
    }


# ================================================================
# Run Top 5 / 10 / 20
# ================================================================

results = []

for top_n in [5, 10, 20]:
    results.append(run_experiment(top_n))

# ================================================================
# Final comparison
# ================================================================

print("\n\n" + "=" * 110)
print("FINAL CO-PURCHASE COMPARISON")
print("=" * 110)

print(
    f"{'Top N':<10}"
    f"{'Co-Purchase':>18}"
    f"{'Combined':>18}"
    f"{'Avg/User':>12}"
    f"{'Coverage':>12}"
    f"{'Incremental':>14}"
)

print("-" * 110)

for r in results:
    print(
        f"{r['top_n']:<10}"
        f"{r['copurchase_candidate_rows']:>18,}"
        f"{r['combined_candidate_rows']:>18,}"
        f"{r['avg_candidates_per_customer']:>12.2f}"
        f"{r['combined_coverage']:>11.2f}%"
        f"{r['incremental_coverage']:>13.2f} pp"
    )

# ================================================================
# Save
# ================================================================

output_path = "data/processed/copurchase_experiment.csv"

pd.DataFrame(results).to_csv(
    output_path,
    index=False
)

print("\nSaved results to:")
print(output_path)

print(f"\nTotal runtime: {time.time() - start:.2f} seconds")

con.close()

print("\n" + "=" * 100)
print("CO-PURCHASE EXPERIMENT COMPLETE")
print("=" * 100)