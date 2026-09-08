from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "recommendation_results.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Tableau"
    / "data"
)

DETAIL_OUTPUT = OUTPUT_DIR / "tableau_recommendation_detail.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "tableau_recommendation_summary.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("PREPARING TABLEAU RECOMMENDATION DATA")
print("=" * 80)

print(f"\nInput: {INPUT_PATH}")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows loaded: {len(df):,}")
print(f"Customers: {df['user_id'].nunique():,}")


# ============================================================
# CREATE RECOMMENDATION TYPE
# ============================================================

df["recommendation_type"] = (
    df["previous_purchases"]
    .gt(0)
    .map({
        True: "Previously Purchased",
        False: "New-to-Customer"
    })
)


# ============================================================
# RENAME MODEL SCORE
# ============================================================

df = df.rename(
    columns={
        "predicted_probability": "recommendation_score"
    }
)


# ============================================================
# SELECT DETAIL COLUMNS
# ============================================================

detail_columns = [
    "user_id",
    "recommendation_rank",
    "product_id",
    "product_name",
    "department",
    "recommendation_score",
    "previous_purchases",
    "previous_reorders",
    "customer_product_reorder_rate",
    "purchase_recency",
    "previous_orders",
    "avg_basket_size",
    "customer_reorder_rate",
    "product_purchase_count",
    "product_unique_customers",
    "product_reorder_rate",
    "customer_department_purchases",
    "customer_department_share",
    "product_department_popularity",
    "recommendation_type"
]

detail = df[detail_columns].copy()


# ============================================================
# SUMMARY DATASET
# ============================================================

summary = (
    df.groupby("recommendation_type", as_index=False)
      .agg(
          recommendation_count=("product_id", "size"),
          unique_customers=("user_id", "nunique"),
          unique_products=("product_id", "nunique"),
          average_score=("recommendation_score", "mean"),
          average_previous_purchases=("previous_purchases", "mean")
      )
)

summary["recommendation_share_percent"] = (
    summary["recommendation_count"]
    / summary["recommendation_count"].sum()
    * 100
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WRITE FILES
# ============================================================

detail.to_csv(
    DETAIL_OUTPUT,
    index=False
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("OUTPUT VALIDATION")
print("-" * 80)

print(f"\nDetail rows:              {len(detail):,}")
print(f"Detail customers:         {detail['user_id'].nunique():,}")
print(f"Detail products:          {detail['product_id'].nunique():,}")

print("\nRecommendations per customer:")

print(
    detail.groupby("user_id")
          .size()
          .describe()
)

print("\nRecommendation type:")

print(
    detail["recommendation_type"]
    .value_counts()
    .to_string()
)

print("\nSummary dataset:")

print(
    summary.to_string(index=False)
)

print("\nFiles created:")

print(DETAIL_OUTPUT)
print(SUMMARY_OUTPUT)

print("\n" + "=" * 80)
print("TABLEAU RECOMMENDATION DATA PREPARATION COMPLETE")
print("=" * 80)