import duckdb
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "processed" / "instacart.duckdb"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Connecting to DuckDB...")
con = duckdb.connect(str(DB_PATH))

# ============================================================
# 1. DEPARTMENT PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("DEPARTMENT PERFORMANCE")
print("=" * 80)

department_summary = con.execute("""
    SELECT
        p.department,
        COUNT(DISTINCT op.product_id) AS products,
        COUNT(*) AS total_purchases,
        COUNT(DISTINCT op.order_id) AS unique_orders,
        COUNT(DISTINCT o.user_id) AS unique_customers,

        ROUND(
            COUNT(*) * 100.0 /
            SUM(COUNT(*)) OVER (),
            2
        ) AS purchase_share,

        ROUND(
            AVG(op.reordered) * 100.0,
            2
        ) AS reorder_rate,

        ROUND(
            COUNT(*) * 1.0 /
            COUNT(DISTINCT o.user_id),
            2
        ) AS purchases_per_customer,

        ROUND(
            COUNT(*) * 1.0 /
            COUNT(DISTINCT op.order_id),
            2
        ) AS avg_items_per_order

    FROM order_products op
    JOIN orders o
        ON op.order_id = o.order_id
    JOIN products p
        ON op.product_id = p.product_id

    GROUP BY p.department

    ORDER BY total_purchases DESC
""").df()

print(department_summary.to_string(index=False))

department_summary.to_csv(
    OUTPUT_DIR / "department_reorder_summary.csv",
    index=False
)

# ============================================================
# 2. DEPARTMENT CUSTOMER REACH
# ============================================================

print("\n" + "=" * 80)
print("DEPARTMENT CUSTOMER REACH")
print("=" * 80)

department_reach = con.execute("""
    WITH total_customers AS (
        SELECT COUNT(DISTINCT user_id) AS customers
        FROM orders
    )

    SELECT
        p.department,

        COUNT(DISTINCT o.user_id) AS unique_customers,

        ROUND(
            COUNT(DISTINCT o.user_id) * 100.0 /
            MAX(tc.customers),
            2
        ) AS customer_penetration,

        COUNT(*) AS total_purchases,

        ROUND(
            AVG(op.reordered) * 100.0,
            2
        ) AS reorder_rate

    FROM order_products op
    JOIN orders o
        ON op.order_id = o.order_id
    JOIN products p
        ON op.product_id = p.product_id
    CROSS JOIN total_customers tc

    GROUP BY
        p.department

    ORDER BY
        customer_penetration DESC
""").df()

print(department_reach.to_string(index=False))

department_reach.to_csv(
    OUTPUT_DIR / "department_customer_reach.csv",
    index=False
)

# ============================================================
# 3. DEPARTMENT REORDER BY CUSTOMER SEGMENT
# ============================================================

print("\n" + "=" * 80)
print("DEPARTMENT REORDER BY CUSTOMER SEGMENT")
print("=" * 80)

department_segment = con.execute("""
    WITH customer_stats AS (

        SELECT
            o.user_id,

            COUNT(DISTINCT o.order_id) AS total_orders,

            AVG(op.reordered) AS reorder_rate

        FROM order_products op
        JOIN orders o
            ON op.order_id = o.order_id

        GROUP BY o.user_id
    ),

    customer_segments AS (

        SELECT
            user_id,

            CASE

                WHEN total_orders <= 5
                    THEN 'Occasional'

                WHEN total_orders <= 10
                    THEN 'Regular'

                WHEN total_orders <= 20
                    THEN 'Established'

                WHEN total_orders <= 30
                    THEN 'Frequent'

                ELSE 'Highly Loyal'

            END AS customer_segment

        FROM customer_stats
    )

    SELECT
        cs.customer_segment,
        p.department,

        COUNT(*) AS purchases,

        COUNT(DISTINCT o.user_id) AS customers,

        ROUND(
            AVG(op.reordered) * 100.0,
            2
        ) AS reorder_rate,

        ROUND(
            COUNT(*) * 1.0 /
            COUNT(DISTINCT o.user_id),
            2
        ) AS purchases_per_customer

    FROM order_products op

    JOIN orders o
        ON op.order_id = o.order_id

    JOIN products p
        ON op.product_id = p.product_id

    JOIN customer_segments cs
        ON o.user_id = cs.user_id

    GROUP BY
        cs.customer_segment,
        p.department

    ORDER BY
        cs.customer_segment,
        purchases DESC
""").df()

print(department_segment.to_string(index=False))

department_segment.to_csv(
    OUTPUT_DIR / "department_reorder_by_customer_segment.csv",
    index=False
)

# ============================================================
# 4. PRODUCT REORDER SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PRODUCT REORDER SUMMARY")
print("=" * 80)

product_reorder = con.execute("""
    SELECT
        p.product_id,
        p.product_name,
        p.department,

        COUNT(*) AS total_purchases,

        COUNT(DISTINCT o.user_id) AS unique_customers,

        ROUND(
            AVG(op.reordered) * 100.0,
            2
        ) AS reorder_rate,

        ROUND(
            COUNT(*) * 1.0 /
            COUNT(DISTINCT o.user_id),
            2
        ) AS purchases_per_customer

    FROM order_products op

    JOIN orders o
        ON op.order_id = o.order_id

    JOIN products p
        ON op.product_id = p.product_id

    GROUP BY
        p.product_id,
        p.product_name,
        p.department

    ORDER BY
        reorder_rate DESC
""").df()

print("\nTop products by reorder rate with meaningful support:")

print(
    product_reorder[
        product_reorder["total_purchases"] >= 1000
    ]
    .head(30)
    .to_string(index=False)
)

product_reorder.to_csv(
    OUTPUT_DIR / "product_reorder_summary.csv",
    index=False
)

# ============================================================
# 5. REORDER OPPORTUNITY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("REORDER OPPORTUNITY ANALYSIS")
print("=" * 80)

# Combine department performance with true customer reach.
department_opportunity = department_summary.merge(
    department_reach[
        [
            "department",
            "customer_penetration"
        ]
    ],
    on="department",
    how="left"
)

# Use department-level median thresholds.
volume_median = department_opportunity["total_purchases"].median()
reorder_median = department_opportunity["reorder_rate"].median()


def classify(row):

    high_volume = row["total_purchases"] >= volume_median
    high_reorder = row["reorder_rate"] >= reorder_median

    if high_volume and high_reorder:
        return "High Volume / High Reorder"

    elif high_volume and not high_reorder:
        return "High Volume / Lower Reorder"

    elif not high_volume and high_reorder:
        return "Lower Volume / High Reorder"

    else:
        return "Lower Volume / Lower Reorder"


department_opportunity["business_segment"] = (
    department_opportunity.apply(classify, axis=1)
)

print(
    department_opportunity[
        [
            "department",
            "total_purchases",
            "purchase_share",
            "unique_customers",
            "customer_penetration",
            "reorder_rate",
            "business_segment"
        ]
    ].to_string(index=False)
)

# ============================================================
# 6. REORDER OPPORTUNITY SCORE
# ============================================================

print("\n" + "=" * 80)
print("REORDER OPPORTUNITY SCORE")
print("=" * 80)


def min_max(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(0.5, index=series.index)

    return (series - minimum) / (maximum - minimum)


department_opportunity["volume_score"] = min_max(
    department_opportunity["total_purchases"]
)

department_opportunity["reach_score"] = min_max(
    department_opportunity["customer_penetration"]
)

department_opportunity["reorder_score"] = min_max(
    department_opportunity["reorder_rate"]
)

# Handcrafted prioritization index.
# This is NOT a predictive ML score.

department_opportunity["reorder_opportunity_score"] = (
    0.40 * department_opportunity["reach_score"]
    + 0.35 * department_opportunity["reorder_score"]
    + 0.25 * department_opportunity["volume_score"]
)

department_opportunity = department_opportunity.sort_values(
    "reorder_opportunity_score",
    ascending=False
)

print(
    department_opportunity[
        [
            "department",
            "total_purchases",
            "purchase_share",
            "customer_penetration",
            "reorder_rate",
            "reorder_opportunity_score"
        ]
    ].to_string(index=False)
)

department_opportunity.to_csv(
    OUTPUT_DIR / "reorder_opportunity_analysis.csv",
    index=False
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("DEPARTMENT & REORDER ANALYSIS COMPLETE")
print("=" * 80)

print("\nGenerated files:")

for filename in [
    "department_reorder_summary.csv",
    "department_customer_reach.csv",
    "department_reorder_by_customer_segment.csv",
    "product_reorder_summary.csv",
    "reorder_opportunity_analysis.csv"
]:

    print(OUTPUT_DIR / filename)

con.close()

print("\nDone.")

# ============================================================
# 6. REORDER OPPORTUNITY SCORE
# ============================================================

print("\n" + "=" * 80)
print("REORDER OPPORTUNITY SCORE")
print("=" * 80)

# Normalize department metrics between 0 and 1.
def min_max(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(0.5, index=series.index)

    return (series - minimum) / (maximum - minimum)


department_opportunity["volume_score"] = min_max(
    department_opportunity["total_purchases"]
)

department_opportunity["reach_score"] = min_max(
    department_opportunity["customer_penetration"]
)

department_opportunity["reorder_score"] = min_max(
    department_opportunity["reorder_rate"]
)

# Handcrafted prioritization index.
#
# This is NOT a predictive ML score.
department_opportunity["reorder_opportunity_score"] = (
    0.40 * department_opportunity["reach_score"]
    + 0.35 * department_opportunity["reorder_score"]
    + 0.25 * department_opportunity["volume_score"]
)

department_opportunity = department_opportunity.sort_values(
    "reorder_opportunity_score",
    ascending=False
)

print(
    department_opportunity[
        [
            "department",
            "total_purchases",
            "purchase_share",
            "customer_penetration",
            "reorder_rate",
            "reorder_opportunity_score"
        ]
    ].to_string(index=False)
)

department_opportunity.to_csv(
    OUTPUT_DIR / "reorder_opportunity_analysis.csv",
    index=False
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("DEPARTMENT & REORDER ANALYSIS COMPLETE")
print("=" * 80)

print("\nGenerated files:")

for filename in [
    "department_reorder_summary.csv",
    "department_customer_reach.csv",
    "department_reorder_by_customer_segment.csv",
    "product_reorder_summary.csv",
    "reorder_opportunity_analysis.csv"
]:

    print(OUTPUT_DIR / filename)

con.close()

print("\nDone.")