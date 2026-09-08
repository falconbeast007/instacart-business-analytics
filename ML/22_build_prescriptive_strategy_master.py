from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

RECOMMENDATION_FILE = PROCESSED_DIR / "recommendation_results.csv"
FORECAST_FILE = PROCESSED_DIR / "customer_forecast_master.csv"

OUTPUT_FILE = PROCESSED_DIR / "prescriptive_strategy_master.csv"

TOP_K = 10


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PRESCRIPTIVE STRATEGY MASTER")
print("=" * 70)

print("\nLoading recommendation data...")
recommendations = pd.read_csv(RECOMMENDATION_FILE)

print(f"Recommendation rows: {len(recommendations):,}")
print(f"Recommendation customers: {recommendations['user_id'].nunique():,}")


print("\nLoading customer forecast data...")
forecast = pd.read_csv(FORECAST_FILE)

print(f"Forecast rows: {len(forecast):,}")
print(f"Forecast customers: {forecast['user_id'].nunique():,}")


# ============================================================
# VALIDATE INPUTS
# ============================================================

required_recommendation_columns = [
    "user_id",
    "recommendation_rank",
    "product_id",
    "product_name",
    "department",
    "predicted_probability",
]

required_forecast_columns = [
    "user_id",
    "future_order_likelihood",
    "expected_reorder_rate",
    "expected_basket_size",
    "expected_reorder_items",
    "expected_new_items",
    "historical_reorder_rate",
    "forecast_segment",
]


missing_rec = [
    col for col in required_recommendation_columns
    if col not in recommendations.columns
]

missing_forecast = [
    col for col in required_forecast_columns
    if col not in forecast.columns
]

if missing_rec:
    raise ValueError(
        f"Missing recommendation columns: {missing_rec}"
    )

if missing_forecast:
    raise ValueError(
        f"Missing forecast columns: {missing_forecast}"
    )


# ============================================================
# KEEP TOP-K RECOMMENDATIONS
# ============================================================

print(f"\nKeeping top {TOP_K} recommendations per customer...")

recommendations = recommendations.sort_values(
    ["user_id", "recommendation_rank"]
)

top_recommendations = (
    recommendations
    .groupby("user_id", as_index=False)
    .head(TOP_K)
    .copy()
)

print(
    f"Rows after Top-{TOP_K}: "
    f"{len(top_recommendations):,}"
)


# ============================================================
# CUSTOMER-LEVEL RECOMMENDATION SUMMARY
# ============================================================

print("\nBuilding recommendation summary...")

# Number of available recommendations
rec_count = (
    top_recommendations
    .groupby("user_id")
    .size()
    .rename("recommendation_count")
    .reset_index()
)

# Top recommendation
top_product = (
    top_recommendations[
        top_recommendations["recommendation_rank"] == 1
    ][
        [
            "user_id",
            "product_id",
            "product_name",
            "department",
            "predicted_probability",
        ]
    ]
    .rename(
        columns={
            "product_id": "top_recommended_product_id",
            "product_name": "top_recommended_product",
            "department": "top_recommended_department",
            "predicted_probability": "top_recommendation_probability",
        }
    )
)

# Average recommendation probability
avg_probability = (
    top_recommendations
    .groupby("user_id")["predicted_probability"]
    .mean()
    .rename("avg_recommendation_probability")
    .reset_index()
)

# Number of unique departments represented in recommendations
department_count = (
    top_recommendations
    .groupby("user_id")["department"]
    .nunique()
    .rename("recommended_department_count")
    .reset_index()
)


# ============================================================
# TOP RECOMMENDED DEPARTMENT
# ============================================================

department_summary = (
    top_recommendations
    .groupby(["user_id", "department"])
    .agg(
        department_recommendations=("product_id", "count"),
        department_probability=("predicted_probability", "mean")
    )
    .reset_index()
)

department_summary = department_summary.sort_values(
    [
        "user_id",
        "department_recommendations",
        "department_probability"
    ],
    ascending=[True, False, False]
)

top_department = (
    department_summary
    .drop_duplicates("user_id")
    [
        [
            "user_id",
            "department",
            "department_recommendations",
            "department_probability",
        ]
    ]
    .rename(
        columns={
            "department": "top_recommended_department_by_depth",
            "department_recommendations":
                "top_department_recommendation_count",
            "department_probability":
                "top_department_avg_probability",
        }
    )
)


# ============================================================
# MERGE RECOMMENDATIONS
# ============================================================

recommendation_summary = (
    rec_count
    .merge(top_product, on="user_id", how="left")
    .merge(avg_probability, on="user_id", how="left")
    .merge(department_count, on="user_id", how="left")
    .merge(top_department, on="user_id", how="left")
)


# ============================================================
# MERGE WITH FORECAST
# ============================================================

print("\nMerging forecast + recommendation signals...")

df = forecast.merge(
    recommendation_summary,
    on="user_id",
    how="left"
)


# ============================================================
# HANDLE CUSTOMERS WITHOUT RECOMMENDATIONS
# ============================================================

df["recommendation_count"] = (
    df["recommendation_count"]
    .fillna(0)
    .astype(int)
)

df["recommended_department_count"] = (
    df["recommended_department_count"]
    .fillna(0)
    .astype(int)
)

df["avg_recommendation_probability"] = (
    df["avg_recommendation_probability"]
    .fillna(0)
)

df["top_recommendation_probability"] = (
    df["top_recommendation_probability"]
    .fillna(0)
)


# ============================================================
# PRESCRIPTIVE STRATEGY
# ============================================================

print("\nGenerating business actions...")


def determine_priority(row):
    likelihood = row["future_order_likelihood"]
    reorder_rate = row["expected_reorder_rate"]

    # High likelihood + high loyalty
    if likelihood >= 0.70 and reorder_rate >= 0.60:
        return "High"

    # Low likelihood = retention risk
    if likelihood < 0.40:
        return "High"

    # Moderate likelihood + moderate loyalty
    if likelihood >= 0.50 and reorder_rate >= 0.50:
        return "Medium"

    return "Low"


def determine_action(row):
    likelihood = row["future_order_likelihood"]
    reorder_rate = row["expected_reorder_rate"]
    new_items = row["expected_new_items"]
    recommendation_probability = row["top_recommendation_probability"]

    # --------------------------------------------------------
    # 1. High-value loyal customers
    # --------------------------------------------------------
    if likelihood >= 0.70 and reorder_rate >= 0.60:
        return "Retain & Reward"

    # --------------------------------------------------------
    # 2. Customers unlikely to order soon
    # --------------------------------------------------------
    if likelihood < 0.40:
        return "Re-engage Customer"

    # --------------------------------------------------------
    # 3. Strong reorder behavior + discovery opportunity
    # --------------------------------------------------------
    if reorder_rate >= 0.60 and new_items >= 3:
        return "Cross-sell & Discover"

    # --------------------------------------------------------
    # 4. Strong recommendation signal
    # --------------------------------------------------------
    if recommendation_probability >= 0.60 and new_items >= 2:
        return "Promote Recommendations"

    # --------------------------------------------------------
    # 5. Stable customers
    # --------------------------------------------------------
    if likelihood >= 0.50 and reorder_rate >= 0.40:
        return "Maintain Engagement"

    # --------------------------------------------------------
    # 6. Default
    # --------------------------------------------------------
    return "Monitor"


def determine_reason(row):
    likelihood = row["future_order_likelihood"]
    reorder_rate = row["expected_reorder_rate"]
    basket = row["expected_basket_size"]
    new_items = row["expected_new_items"]
    rec_probability = row["top_recommendation_probability"]

    likelihood_pct = likelihood * 100
    reorder_pct = reorder_rate * 100

    if likelihood >= 0.70 and reorder_rate >= 0.60:
        return (
            f"High likelihood of ordering ({likelihood_pct:.0f}%) "
            f"and strong repeat behavior ({reorder_pct:.0f}% expected reorder rate)."
        )

    if likelihood < 0.40:
        return (
            f"Low near-term order likelihood ({likelihood_pct:.0f}%), "
            f"indicating a potential retention opportunity."
        )

    if reorder_rate >= 0.60 and new_items >= 3:
        return (
            f"Strong repeat behavior ({reorder_pct:.0f}%) "
            f"with approximately {new_items:.1f} expected new items "
            f"in the next basket."
        )

    if rec_probability >= 0.60 and new_items >= 2:
        return (
            f"Recommendations show strong predicted relevance "
            f"({rec_probability * 100:.0f}%) with discovery potential."
        )

    if likelihood >= 0.50 and reorder_rate >= 0.40:
        return (
            f"Stable expected ordering behavior ({likelihood_pct:.0f}% likelihood) "
            f"with an estimated basket of {basket:.1f} items."
        )

    return (
        f"Moderate behavioral signals with an estimated "
        f"{basket:.1f}-item next basket."
    )


df["action_priority"] = df.apply(
    determine_priority,
    axis=1
)

df["recommended_action"] = df.apply(
    determine_action,
    axis=1
)

df["action_reason"] = df.apply(
    determine_reason,
    axis=1
)


# ============================================================
# BUSINESS OPPORTUNITY SCORE
# ============================================================

# This is NOT revenue or ROI.
# It is an interpretable prioritization score based on
# model signals.

df["business_opportunity_score"] = (
    0.40 * df["future_order_likelihood"]
    + 0.30 * df["expected_reorder_rate"]
    + 0.20 * df["top_recommendation_probability"]
    + 0.10 * np.clip(
        df["expected_new_items"] / 10,
        0,
        1
    )
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

final_columns = [
    # Customer
    "user_id",

    # Prescriptive output
    "action_priority",
    "recommended_action",
    "action_reason",
    "business_opportunity_score",

    # Forecast signals
    "forecast_segment",
    "future_order_likelihood",
    "expected_reorder_rate",
    "expected_basket_size",
    "expected_reorder_items",
    "expected_new_items",

    # Recommendation signals
    "recommendation_count",
    "top_recommended_product_id",
    "top_recommended_product",
    "top_recommended_department",
    "top_recommendation_probability",
    "avg_recommendation_probability",
    "recommended_department_count",
    "top_recommended_department_by_depth",

    # Historical context
    "historical_reorder_rate",
    "previous_orders",
    "current_basket_size",
    "previous_basket_size",
    "historical_avg_basket_size",
    "recent_3_order_reorder_rate",
    "previous_order_reorder_rate",
    "historical_avg_days_between_orders",
    "current_order_days_since_prior",
    "recent_3_order_basket",
    "recent_3_order_interval",

    # Metadata
    "forecast_horizon",
    "forecast_target",
]

# Keep only columns that actually exist
final_columns = [
    col for col in final_columns
    if col in df.columns
]

df = df[final_columns]


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "action_priority",
        "business_opportunity_score"
    ],
    ascending=[True, False]
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(f"\nRows: {len(df):,}")
print(f"Customers: {df['user_id'].nunique():,}")

duplicate_customers = df["user_id"].duplicated().sum()

print(f"Duplicate customers: {duplicate_customers:,}")

missing_predictions = df[
    [
        "future_order_likelihood",
        "expected_reorder_rate",
        "expected_basket_size",
        "recommended_action",
    ]
].isna().any(axis=1).sum()

print(f"Rows with missing core outputs: {missing_predictions:,}")


print("\nAction distribution:")
print(
    df["recommended_action"]
    .value_counts()
    .to_string()
)


print("\nPriority distribution:")
print(
    df["action_priority"]
    .value_counts()
    .to_string()
)


print("\nForecast segment distribution:")
print(
    df["forecast_segment"]
    .value_counts()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT CREATED")
print("=" * 70)

print(f"\n{OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nSample:")
print(
    df[
        [
            "user_id",
            "forecast_segment",
            "future_order_likelihood",
            "expected_reorder_rate",
            "expected_basket_size",
            "top_recommended_product",
            "recommended_action",
            "action_priority",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nDone.")