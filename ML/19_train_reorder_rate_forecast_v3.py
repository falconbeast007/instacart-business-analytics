import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "Data" / "processed" / "instacart.duckdb"
FORECAST_PATH = BASE_DIR / "Data" / "processed" / "forecasting_dataset.parquet"

RESULTS_PATH = (
    BASE_DIR
    / "Data"
    / "processed"
    / "reorder_rate_forecast_v3_results.csv"
)

IMPORTANCE_PATH = (
    BASE_DIR
    / "Data"
    / "processed"
    / "reorder_rate_forecast_v3_feature_importance.csv"
)

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MODEL 2 V3 — FUTURE REORDER RATE")
print("=" * 70)


# ============================================================
# 1. LOAD FORECASTING DATASET
# ============================================================

print("\n" + "=" * 70)
print("1. Loading forecasting dataset")
print("=" * 70)

df = pd.read_parquet(FORECAST_PATH)

print(f"Rows      : {len(df):,}")
print(f"Customers : {df['user_id'].nunique():,}")
print("Target    : next_reorder_rate")


# ============================================================
# 2. SORT PREDICTION POINTS
# ============================================================

print("\n" + "=" * 70)
print("2. Sorting prediction points")
print("=" * 70)

df = df.sort_values(
    ["user_id", "prediction_order_number"]
).reset_index(drop=True)


# ============================================================
# 3. CREATE THREE-WAY TEMPORAL SPLIT
# ============================================================

print("\n" + "=" * 70)
print("3. Creating three-way temporal split")
print("=" * 70)

customer_counts = (
    df.groupby("user_id")
      .size()
)

eligible_customers = customer_counts[
    customer_counts >= 3
].index

df = df[
    df["user_id"].isin(eligible_customers)
].copy()

df["reverse_rank"] = (
    df.groupby("user_id")["prediction_order_number"]
      .rank(
          method="first",
          ascending=False
      )
      .astype(int)
      - 1
)

print(
    f"Customers with >=3 prediction points: "
    f"{df['user_id'].nunique():,}"
)

print("\nTemporal split:")
print("  Training      : reverse_rank >= 2")
print("  Selection     : reverse_rank == 1")
print("  Final holdout : reverse_rank == 0")


# ============================================================
# 4. BASE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("4. Preparing base features")
print("=" * 70)

base_features = [
    "previous_orders",
    "current_basket_size",
    "historical_avg_basket_size",
    "previous_basket_size",
    "recent_3_order_basket",
    "historical_reorder_rate",
    "previous_order_reorder_rate",
    "recent_3_order_reorder_rate",
    "cumulative_reorder_rate",
    "historical_avg_days_between_orders",
    "current_order_days_since_prior",
    "recent_3_order_interval",
    "previous_total_items",
    "previous_total_reorder_items",
    "order_dow",
    "order_hour_of_day"
]


# ============================================================
# 5. CREATE MODEL 2 BEHAVIORAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("5. Creating Model 2 behavioral features")
print("=" * 70)


# ------------------------------------------------------------
# Recent-5 reorder rate
# ------------------------------------------------------------

df["recent_5_order_reorder_rate"] = (
    df.groupby("user_id")["previous_order_reorder_rate"]
      .transform(
          lambda x:
          x.rolling(
              5,
              min_periods=1
          ).mean()
      )
)


# ------------------------------------------------------------
# Recent-5 basket size
# ------------------------------------------------------------

df["recent_5_order_basket"] = (
    df.groupby("user_id")["previous_basket_size"]
      .transform(
          lambda x:
          x.rolling(
              5,
              min_periods=1
          ).mean()
      )
)


# ------------------------------------------------------------
# Recent-5 interval
# ------------------------------------------------------------

df["recent_5_order_interval"] = (
    df.groupby("user_id")["current_order_days_since_prior"]
      .transform(
          lambda x:
          x.rolling(
              5,
              min_periods=1
          ).mean()
      )
)


# ------------------------------------------------------------
# Reorder-rate trend — recent 3 vs historical
# ------------------------------------------------------------

df["reorder_rate_trend_3"] = (
    df["recent_3_order_reorder_rate"]
    - df["historical_reorder_rate"]
)


# ------------------------------------------------------------
# Reorder-rate trend — previous vs historical
# ------------------------------------------------------------

df["reorder_rate_trend_previous"] = (
    df["previous_order_reorder_rate"]
    - df["historical_reorder_rate"]
)


# ------------------------------------------------------------
# Basket-size trend — recent 3 vs historical
# ------------------------------------------------------------

df["basket_size_trend_3"] = (
    df["recent_3_order_basket"]
    - df["historical_avg_basket_size"]
)


# ------------------------------------------------------------
# Basket-size trend — previous vs historical
# ------------------------------------------------------------

df["basket_size_trend_previous"] = (
    df["previous_basket_size"]
    - df["historical_avg_basket_size"]
)


# ------------------------------------------------------------
# Reorder-rate volatility — recent 5
# ------------------------------------------------------------

df["reorder_rate_std_5"] = (
    df.groupby("user_id")["previous_order_reorder_rate"]
      .transform(
          lambda x:
          x.rolling(
              5,
              min_periods=2
          ).std()
      )
)


# ------------------------------------------------------------
# Reorder-rate volatility — recent 3
# ------------------------------------------------------------

df["reorder_rate_std_3"] = (
    df.groupby("user_id")["previous_order_reorder_rate"]
      .transform(
          lambda x:
          x.rolling(
              3,
              min_periods=2
          ).std()
      )
)


# ------------------------------------------------------------
# Interval trend
# ------------------------------------------------------------

df["interval_trend"] = (
    df["recent_3_order_interval"]
    - df["historical_avg_days_between_orders"]
)


# ============================================================
# 6. V3 CADENCE / MATURITY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("6. Creating Model 2 V3 cadence features")
print("=" * 70)


# ------------------------------------------------------------
# Order frequency rate
# ------------------------------------------------------------

df["order_frequency_rate"] = (
    df["previous_orders"]
    /
    df["historical_avg_days_between_orders"]
    .replace(0, np.nan)
)

df["order_frequency_rate"] = (
    df["order_frequency_rate"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ------------------------------------------------------------
# Recent-3 order frequency
# ------------------------------------------------------------

df["recent_3_order_frequency"] = (
    1
    /
    df["recent_3_order_interval"]
    .replace(0, np.nan)
)

df["recent_3_order_frequency"] = (
    df["recent_3_order_frequency"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ------------------------------------------------------------
# Recent-5 order frequency
# ------------------------------------------------------------

df["recent_5_order_frequency"] = (
    1
    /
    df["recent_5_order_interval"]
    .replace(0, np.nan)
)

df["recent_5_order_frequency"] = (
    df["recent_5_order_frequency"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ------------------------------------------------------------
# Interval deviation
# ------------------------------------------------------------

df["interval_deviation"] = (
    df["current_order_days_since_prior"]
    -
    df["historical_avg_days_between_orders"]
)


# ------------------------------------------------------------
# Interval ratio
# ------------------------------------------------------------

df["interval_ratio"] = (
    df["current_order_days_since_prior"]
    /
    df["historical_avg_days_between_orders"]
    .replace(0, np.nan)
)

df["interval_ratio"] = (
    df["interval_ratio"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ------------------------------------------------------------
# Recent interval deviation
# ------------------------------------------------------------

df["recent_interval_deviation"] = (
    df["recent_3_order_interval"]
    -
    df["historical_avg_days_between_orders"]
)


# ------------------------------------------------------------
# Interval trend
# ------------------------------------------------------------

df["interval_trend_v3"] = (
    df["recent_3_order_interval"]
    -
    df["recent_5_order_interval"]
)


# ------------------------------------------------------------
# Recent vs historical frequency
# ------------------------------------------------------------

df["recent_vs_historical_frequency"] = (
    df["recent_3_order_frequency"]
    -
    df["order_frequency_rate"]
)


# ------------------------------------------------------------
# Ordering consistency
# ------------------------------------------------------------

df["ordering_consistency"] = (
    1
    /
    (
        df["reorder_rate_std_3"]
        + 1e-6
    )
)


# ------------------------------------------------------------
# Customer maturity
# ------------------------------------------------------------

df["customer_maturity"] = (
    np.log1p(
        df["previous_orders"]
    )
)


# ------------------------------------------------------------
# Recent activity maturity
# ------------------------------------------------------------

df["recent_activity_maturity"] = (
    df["previous_orders"]
    /
    (
        df["current_order_days_since_prior"]
        + 1
    )
)


# ============================================================
# 7. FEATURE LIST
# ============================================================

print("\n" + "=" * 70)
print("7. Selecting Model 2 V3 features")
print("=" * 70)

v3_behavioral_features = [
    "order_frequency_rate",
    "recent_3_order_frequency",
    "recent_5_order_frequency",
    "interval_deviation",
    "interval_ratio",
    "recent_interval_deviation",
    "interval_trend_v3",
    "recent_vs_historical_frequency",
    "ordering_consistency",
    "customer_maturity",
    "recent_activity_maturity"
]

features = (
    base_features
    +
    v3_behavioral_features
)

features = list(
    dict.fromkeys(features)
)

print(
    f"Base features       : "
    f"{len(base_features)}"
)

print(
    f"V3 behavioral       : "
    f"{len(v3_behavioral_features)}"
)

print(
    f"Total features      : "
    f"{len(features)}"
)


# ============================================================
# 8. FEATURE LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("8. Feature leakage check")
print("=" * 70)

if "next_reorder_rate" in features:
    raise ValueError(
        "LEAKAGE DETECTED: "
        "next_reorder_rate is present in features."
    )

print("Feature leakage check: PASSED")


# ============================================================
# 9. CLEAN FEATURE TYPES
# ============================================================

print("\n" + "=" * 70)
print("9. Cleaning feature types")
print("=" * 70)

# XGBoost requires numerical / boolean / categorical
# feature columns. In this dataset order_hour_of_day
# may be loaded as string, so explicitly convert every
# model feature to numeric.

df[features] = (
    df[features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

for feature in features:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

df[features] = (
    df[features]
    .fillna(0)
)

# Final dtype verification
non_numeric_features = [
    feature
    for feature in features
    if not pd.api.types.is_numeric_dtype(
        df[feature]
    )
]

if non_numeric_features:

    raise ValueError(
        "Non-numeric features detected: "
        +
        str(non_numeric_features)
    )

print(
    "All model features are numeric."
)


# ============================================================
# 10. RECREATE TEMPORAL SPLITS
# ============================================================

print("\n" + "=" * 70)
print("10. Preparing temporal splits")
print("=" * 70)

train = df[
    df["reverse_rank"] >= 2
].copy()

selection = df[
    df["reverse_rank"] == 1
].copy()

final = df[
    df["reverse_rank"] == 0
].copy()

print(
    f"Training rows          : "
    f"{len(train):,}"
)

print(
    f"Model-selection rows   : "
    f"{len(selection):,}"
)

print(
    f"Final holdout rows     : "
    f"{len(final):,}"
)


# ============================================================
# 11. MODEL MATRICES
# ============================================================

print("\n" + "=" * 70)
print("11. Preparing model matrices")
print("=" * 70)

X_train = train[features]
y_train = train[
    "next_reorder_rate"
]

X_selection = selection[features]
y_selection = selection[
    "next_reorder_rate"
]

X_final = final[features]
y_final = final[
    "next_reorder_rate"
]

print(
    f"X_train shape     : "
    f"{X_train.shape}"
)

print(
    f"X_selection shape : "
    f"{X_selection.shape}"
)

print(
    f"X_final shape     : "
    f"{X_final.shape}"
)


# ============================================================
# 12. METRIC FUNCTION
# ============================================================

def evaluate_predictions(
    y_true,
    predictions
):

    predictions = np.asarray(
        predictions
    )

    return {
        "MAE": mean_absolute_error(
            y_true,
            predictions
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                predictions
            )
        ),

        "R2": r2_score(
            y_true,
            predictions
        )
    }


# ============================================================
# 13. BASELINE MODELS
# ============================================================

print("\n" + "=" * 70)
print("12. Baseline models")
print("=" * 70)


# ------------------------------------------------------------
# Historical reorder rate
# ------------------------------------------------------------

historical_pred_selection = (
    selection[
        "historical_reorder_rate"
    ]
)

historical_metrics = (
    evaluate_predictions(
        y_selection,
        historical_pred_selection
    )
)

print(
    "\nHistorical Reorder Rate:"
)

print(
    f"MAE  : "
    f"{historical_metrics['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{historical_metrics['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{historical_metrics['R2']:.6f}"
)


# ------------------------------------------------------------
# Recent-3 reorder rate
# ------------------------------------------------------------

recent3_pred_selection = (
    selection[
        "recent_3_order_reorder_rate"
    ]
)

recent3_metrics = (
    evaluate_predictions(
        y_selection,
        recent3_pred_selection
    )
)

print(
    "\nRecent-3 Reorder Rate:"
)

print(
    f"MAE  : "
    f"{recent3_metrics['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{recent3_metrics['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{recent3_metrics['R2']:.6f}"
)


# ============================================================
# 14. CANDIDATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("13. Training Model 2 V3 candidates")
print("=" * 70)

candidates = {

    "XGB V2 Baseline": {
        "max_depth": 8,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "objective": "reg:squarederror"
    },

    "XGB V3 Deep": {
        "max_depth": 10,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "objective": "reg:squarederror"
    },

    "XGB V3 Shallow": {
        "max_depth": 6,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "objective": "reg:squarederror"
    },

    "XGB V3 Regularized": {
        "max_depth": 8,
        "n_estimators": 600,
        "learning_rate": 0.04,
        "min_child_weight": 10,
        "objective": "reg:squarederror"
    }
}


selection_results = []

models = {}


# ============================================================
# 15. TRAIN CANDIDATES
# ============================================================

for name, params in candidates.items():

    print(
        f"\nTraining: {name}"
    )

    model = XGBRegressor(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    predictions = model.predict(
        X_selection
    )

    metrics = evaluate_predictions(
        y_selection,
        predictions
    )

    print(
        f"MAE  : "
        f"{metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : "
        f"{metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : "
        f"{metrics['R2']:.6f}"
    )

    selection_results.append({
        "model": name,
        **metrics
    })

    models[name] = model


# ============================================================
# 16. MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("14. MODEL-SELECTION COMPARISON")
print("=" * 70)

selection_df = (
    pd.DataFrame(
        selection_results
    )
    .sort_values(
        "MAE"
    )
)

print(
    selection_df.to_string(
        index=False
    )
)

selected_name = (
    selection_df.iloc[0]["model"]
)

selected_model = (
    models[selected_name]
)

print("\n" + "=" * 70)
print("15. SELECTING MODEL 2 V3")
print("=" * 70)

print(
    f"Selected model: "
    f"{selected_name}"
)


# ============================================================
# 17. FINAL UNTOUCHED HOLDOUT
# ============================================================

print("\n" + "=" * 70)
print("16. FINAL UNTOUCHED HOLDOUT")
print("=" * 70)


# ------------------------------------------------------------
# Historical baseline
# ------------------------------------------------------------

historical_pred_final = (
    final[
        "historical_reorder_rate"
    ]
)

historical_final_metrics = (
    evaluate_predictions(
        y_final,
        historical_pred_final
    )
)

print(
    "\nHistorical Reorder Rate"
)

print(
    f"MAE  : "
    f"{historical_final_metrics['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{historical_final_metrics['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{historical_final_metrics['R2']:.6f}"
)


# ------------------------------------------------------------
# Recent-3 baseline
# ------------------------------------------------------------

recent3_pred_final = (
    final[
        "recent_3_order_reorder_rate"
    ]
)

recent3_final_metrics = (
    evaluate_predictions(
        y_final,
        recent3_pred_final
    )
)

print(
    "\nRecent-3 Reorder Rate"
)

print(
    f"MAE  : "
    f"{recent3_final_metrics['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{recent3_final_metrics['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{recent3_final_metrics['R2']:.6f}"
)


# ------------------------------------------------------------
# Selected XGBoost
# ------------------------------------------------------------

selected_pred_final = (
    selected_model.predict(
        X_final
    )
)

selected_final_metrics = (
    evaluate_predictions(
        y_final,
        selected_pred_final
    )
)

print(
    "\nSelected Model 2 V3"
)

print(
    f"MAE  : "
    f"{selected_final_metrics['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{selected_final_metrics['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{selected_final_metrics['R2']:.6f}"
)


# ============================================================
# 18. XGBOOST + RECENT-3 BLEND
# ============================================================

print("\n" + "=" * 70)
print("17. XGBoost + Recent-3 blend")
print("=" * 70)

BLEND_XGB = 0.92
BLEND_RECENT3 = 0.08

blend_pred_final = (
    BLEND_XGB
    * selected_pred_final
    +
    BLEND_RECENT3
    * recent3_pred_final
)

blend_final_metrics = (
    evaluate_predictions(
        y_final,
        blend_pred_final
    )
)

print(
    f"Blend weights: "
    f"{BLEND_XGB:.0%} XGB + "
    f"{BLEND_RECENT3:.0%} Recent-3"
)

print(
    f"MAE  : "
    f"{blend_final_metrics['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{blend_final_metrics['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{blend_final_metrics['R2']:.6f}"
)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("18. FINAL FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({

    "feature": features,

    "importance":
        selected_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("19. Saving results")
print("=" * 70)

results = pd.DataFrame([

    {
        "model":
            "Historical Reorder Rate",

        "MAE":
            historical_final_metrics[
                "MAE"
            ],

        "RMSE":
            historical_final_metrics[
                "RMSE"
            ],

        "R2":
            historical_final_metrics[
                "R2"
            ]
    },

    {
        "model":
            "Recent-3 Reorder Rate",

        "MAE":
            recent3_final_metrics[
                "MAE"
            ],

        "RMSE":
            recent3_final_metrics[
                "RMSE"
            ],

        "R2":
            recent3_final_metrics[
                "R2"
            ]
    },

    {
        "model":
            selected_name,

        "MAE":
            selected_final_metrics[
                "MAE"
            ],

        "RMSE":
            selected_final_metrics[
                "RMSE"
            ],

        "R2":
            selected_final_metrics[
                "R2"
            ]
    },

    {
        "model":
            "XGB + Recent-3 Blend",

        "MAE":
            blend_final_metrics[
                "MAE"
            ],

        "RMSE":
            blend_final_metrics[
                "RMSE"
            ],

        "R2":
            blend_final_metrics[
                "R2"
            ]
    }

])


results.to_csv(
    RESULTS_PATH,
    index=False
)

importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False
)


print("\nResults saved:")

print(
    f"  {RESULTS_PATH}"
)

print(
    f"  {IMPORTANCE_PATH}"
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL 2 V3 SUMMARY")
print("=" * 70)

print(
    f"Selected model: "
    f"{selected_name}"
)

print(
    f"Final XGB MAE: "
    f"{selected_final_metrics['MAE']:.6f}"
)

print(
    f"Final XGB RMSE: "
    f"{selected_final_metrics['RMSE']:.6f}"
)

print(
    f"Final XGB R2: "
    f"{selected_final_metrics['R2']:.6f}"
)

print(
    f"\nBlend MAE: "
    f"{blend_final_metrics['MAE']:.6f}"
)

print(
    f"Blend RMSE: "
    f"{blend_final_metrics['RMSE']:.6f}"
)

print(
    f"Blend R2: "
    f"{blend_final_metrics['R2']:.6f}"
)

print(
    "\nLeakage check: PASSED"
)

print(
    "Final holdout was not used for model selection."
)

print("\nDone.")