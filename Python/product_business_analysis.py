"""
Step 17: Product Business Analysis

Purpose:
Create business-ready product analytics for the Tableau layer.

Questions:
- Which products drive the most demand?
- Which products reach the most customers?
- Which products have the strongest reorder behavior?
- Which products combine scale and loyalty?
- Which products represent potential recommendation opportunities?

Important:
Reorder rate is descriptive/observational.
It does not imply that a product causes customer loyalty.
"""

import os
import duckdb
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "instacart.duckdb"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


# ============================================================
# CONNECT
# ============================================================

print("Connecting to DuckDB...")

con = duckdb.connect(DB_PATH)


# ============================================================
# CHECK TABLE
# ============================================================

print("Checking product_features table...")

product_count = con.execute(
    """
    SELECT COUNT(*)
    FROM product_features
    """
).fetchone()[0]

print(
    f"Products available: {product_count:,}"
)


# ============================================================
# LOAD PRODUCT DATA
# ============================================================

print("Loading product-level data...")

product_df = con.execute(
    """
    SELECT
        product_id,
        product_name,
        aisle,
        department,
        total_purchases,
        unique_orders,
        unique_customers,
        reorder_rate,
        avg_cart_position
    FROM product_features
    """
).fetchdf()

print(
    f"Products loaded: {len(product_df):,}"
)


# ============================================================
# 1. PRODUCT PERFORMANCE SUMMARY
# ============================================================

print()
print("=" * 80)
print("PRODUCT PERFORMANCE SUMMARY")
print("=" * 80)

summary = pd.DataFrame({
    "metric": [
        "Products",
        "Total purchases",
        "Average purchases per product",
        "Median purchases per product",
        "Average unique customers",
        "Average reorder rate",
        "Average cart position"
    ],

    "value": [
        len(product_df),

        product_df[
            "total_purchases"
        ].sum(),

        product_df[
            "total_purchases"
        ].mean(),

        product_df[
            "total_purchases"
        ].median(),

        product_df[
            "unique_customers"
        ].mean(),

        product_df[
            "reorder_rate"
        ].mean(),

        product_df[
            "avg_cart_position"
        ].mean()
    ]
})

print(
    summary.to_string(
        index=False
    )
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "product_business_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 2. TOP PRODUCTS BY DEMAND
# ============================================================

print()
print("=" * 80)
print("TOP PRODUCTS BY PURCHASE VOLUME")
print("=" * 80)

top_volume = (
    product_df
    .sort_values(
        "total_purchases",
        ascending=False
    )
    .head(50)
    .copy()
)

print(
    top_volume[
        [
            "product_id",
            "product_name",
            "department",
            "total_purchases",
            "unique_customers",
            "reorder_rate"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


top_volume_path = os.path.join(
    OUTPUT_DIR,
    "top_products_by_volume.csv"
)

top_volume.to_csv(
    top_volume_path,
    index=False
)


# ============================================================
# 3. TOP PRODUCTS BY CUSTOMER REACH
# ============================================================

print()
print("=" * 80)
print("TOP PRODUCTS BY CUSTOMER REACH")
print("=" * 80)

top_reach = (
    product_df
    .sort_values(
        "unique_customers",
        ascending=False
    )
    .head(50)
    .copy()
)

print(
    top_reach[
        [
            "product_id",
            "product_name",
            "department",
            "unique_customers",
            "total_purchases",
            "reorder_rate"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


top_reach_path = os.path.join(
    OUTPUT_DIR,
    "top_products_by_customer_reach.csv"
)

top_reach.to_csv(
    top_reach_path,
    index=False
)


# ============================================================
# 4. TOP PRODUCTS BY REORDER RATE
# ============================================================

print()
print("=" * 80)
print("TOP PRODUCTS BY REORDER RATE")
print("=" * 80)

# Avoid products with extremely small sample sizes.
# A product with 1 purchase and 100% reorder rate is not
# as meaningful as a product with thousands of purchases.

reorder_candidates = product_df[
    product_df["total_purchases"] >= 1000
].copy()

top_reorder = (
    reorder_candidates
    .sort_values(
        "reorder_rate",
        ascending=False
    )
    .head(50)
    .copy()
)

print(
    top_reorder[
        [
            "product_id",
            "product_name",
            "department",
            "total_purchases",
            "unique_customers",
            "reorder_rate"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


top_reorder_path = os.path.join(
    OUTPUT_DIR,
    "top_products_by_reorder_rate.csv"
)

top_reorder.to_csv(
    top_reorder_path,
    index=False
)


# ============================================================
# 5. PRODUCT BUSINESS SEGMENTS
# ============================================================

print()
print("=" * 80)
print("PRODUCT BUSINESS SEGMENTS")
print("=" * 80)

# Use dataset medians to avoid arbitrary thresholds.

volume_median = product_df[
    "total_purchases"
].median()

reorder_median = product_df[
    "reorder_rate"
].median()

product_df[
    "business_segment"
] = "Lower Volume / Lower Reorder"

product_df.loc[
    (
        product_df["total_purchases"]
        >= volume_median
    )
    &
    (
        product_df["reorder_rate"]
        < reorder_median
    ),
    "business_segment"
] = "High Volume / Lower Reorder"

product_df.loc[
    (
        product_df["total_purchases"]
        < volume_median
    )
    &
    (
        product_df["reorder_rate"]
        >= reorder_median
    ),
    "business_segment"
] = "Lower Volume / High Reorder"

product_df.loc[
    (
        product_df["total_purchases"]
        >= volume_median
    )
    &
    (
        product_df["reorder_rate"]
        >= reorder_median
    ),
    "business_segment"
] = "High Volume / High Reorder"


segment_summary = (
    product_df
    .groupby("business_segment")
    .agg(
        products=(
            "product_id",
            "count"
        ),

        avg_purchases=(
            "total_purchases",
            "mean"
        ),

        avg_customers=(
            "unique_customers",
            "mean"
        ),

        avg_reorder_rate=(
            "reorder_rate",
            "mean"
        )
    )
    .reset_index()
)


segment_summary[
    "product_share"
] = (
    segment_summary["products"]
    / segment_summary["products"].sum()
)


print(
    segment_summary.to_string(
        index=False
    )
)


segment_path = os.path.join(
    OUTPUT_DIR,
    "product_business_segments.csv"
)

segment_summary.to_csv(
    segment_path,
    index=False
)


# ============================================================
# 6. HIGH-VOLUME / HIGH-REORDER PRODUCTS
# ============================================================

print()
print("=" * 80)
print("HIGH-VOLUME / HIGH-REORDER PRODUCTS")
print("=" * 80)

high_value_products = product_df[
    product_df[
        "business_segment"
    ] == "High Volume / High Reorder"
].copy()


high_value_products = (
    high_value_products
    .sort_values(
        [
            "total_purchases",
            "reorder_rate"
        ],
        ascending=False
    )
)


print(
    high_value_products[
        [
            "product_id",
            "product_name",
            "department",
            "total_purchases",
            "unique_customers",
            "reorder_rate"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


high_value_path = os.path.join(
    OUTPUT_DIR,
    "high_volume_high_reorder_products.csv"
)

high_value_products.to_csv(
    high_value_path,
    index=False
)


# ============================================================
# 7. HIGH-REORDER OPPORTUNITY PRODUCTS
# ============================================================

print()
print("=" * 80)
print("HIGH-REORDER PRODUCTS WITH LOWER VOLUME")
print("=" * 80)

opportunity_products = product_df[
    (
        product_df["business_segment"]
        == "Lower Volume / High Reorder"
    )
].copy()


opportunity_products = (
    opportunity_products
    .sort_values(
        [
            "reorder_rate",
            "unique_customers"
        ],
        ascending=False
    )
)


print(
    opportunity_products[
        [
            "product_id",
            "product_name",
            "department",
            "total_purchases",
            "unique_customers",
            "reorder_rate"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


opportunity_path = os.path.join(
    OUTPUT_DIR,
    "high_reorder_opportunity_products.csv"
)

opportunity_products.to_csv(
    opportunity_path,
    index=False
)


# ============================================================
# 8. DEPARTMENT PRODUCT PERFORMANCE
# ============================================================

print()
print("=" * 80)
print("DEPARTMENT PRODUCT PERFORMANCE")
print("=" * 80)

department_summary = (
    product_df
    .groupby("department")
    .agg(
        products=(
            "product_id",
            "nunique"
        ),

        total_purchases=(
            "total_purchases",
            "sum"
        ),

        avg_purchases_per_product=(
            "total_purchases",
            "mean"
        ),

        unique_customers=(
            "unique_customers",
            "sum"
        ),

        avg_reorder_rate=(
            "reorder_rate",
            "mean"
        )
    )
    .reset_index()
)


department_summary[
    "purchase_share"
] = (
    department_summary[
        "total_purchases"
    ]
    / department_summary[
        "total_purchases"
    ].sum()
)


department_summary = (
    department_summary
    .sort_values(
        "total_purchases",
        ascending=False
    )
)


print(
    department_summary.to_string(
        index=False
    )
)


department_path = os.path.join(
    OUTPUT_DIR,
    "product_department_performance.csv"
)

department_summary.to_csv(
    department_path,
    index=False
)


# ============================================================
# 9. PRODUCT OPPORTUNITY SCORE
# ============================================================

print()
print("=" * 80)
print("PRODUCT OPPORTUNITY SCORE")
print("=" * 80)

# Normalize:
# - customer reach
# - reorder rate
# - purchase volume
#
# This is a prioritization score, NOT a predictive model.

product_df[
    "reach_score"
] = (
    product_df["unique_customers"]
    / product_df["unique_customers"].max()
)

product_df[
    "volume_score"
] = (
    product_df["total_purchases"]
    / product_df["total_purchases"].max()
)

product_df[
    "reorder_score"
] = (
    product_df["reorder_rate"]
    / product_df["reorder_rate"].max()
)


product_df[
    "opportunity_score"
] = (
    0.40 * product_df["reach_score"]
    +
    0.30 * product_df["reorder_score"]
    +
    0.30 * product_df["volume_score"]
)


opportunity_score_df = (
    product_df
    .sort_values(
        "opportunity_score",
        ascending=False
    )
    .head(100)
    .copy()
)


print(
    opportunity_score_df[
        [
            "product_id",
            "product_name",
            "department",
            "total_purchases",
            "unique_customers",
            "reorder_rate",
            "opportunity_score"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


opportunity_score_path = os.path.join(
    OUTPUT_DIR,
    "product_opportunity_score.csv"
)

opportunity_score_df.to_csv(
    opportunity_score_path,
    index=False
)


# ============================================================
# FINAL FILE LIST
# ============================================================

print()
print("=" * 80)
print("PRODUCT BUSINESS ANALYSIS COMPLETE")
print("=" * 80)

files_created = [
    summary_path,
    top_volume_path,
    top_reach_path,
    top_reorder_path,
    segment_path,
    high_value_path,
    opportunity_path,
    department_path,
    opportunity_score_path
]

for path in files_created:
    print(path)

print("=" * 80)


# ============================================================
# CLOSE
# ============================================================

con.close()

print("Done.")