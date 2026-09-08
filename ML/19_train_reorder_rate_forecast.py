from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

INPUT_PATH = (
    PROCESSED_DIR /
    "forecasting_dataset.parquet"
)

RESULTS_PATH = (
    PROCESSED_DIR /
    "reorder_rate_forecast_v4_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "reorder_rate_forecast_v4_feature_importance.csv"
)

RANDOM_STATE = 42

# Small smoothing value for 0/1 target values.
EPSILON = 0.01


# ============================================================
# Helpers
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(actual, predicted):

    predicted = np.clip(
        predicted,
        0,
        1,
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    return mae, rmse, r2


def make_xgb(
    objective="reg:squarederror",
    max_depth=8,
):

    return XGBRegressor(

        n_estimators=500,

        max_depth=max_depth,

        learning_rate=0.05,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        objective=objective,

        eval_metric="mae",

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )


def target_to_logit(values):

    values = np.clip(
        values,
        EPSILON,
        1 - EPSILON,
    )

    return np.log(
        values /
        (1 - values)
    )


def logit_to_target(values):

    values = np.clip(
        values,
        -20,
        20,
    )

    probabilities = (
        1 /
        (
            1 +
            np.exp(-values)
        )
    )

    return np.clip(
        probabilities,
        0,
        1,
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "MODEL 2 V4 — FINAL EXPERIMENT"
    )

    # ========================================================
    # 1. Load dataset
    # ========================================================

    section(
        "1. Loading forecasting dataset"
    )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Rows      : {len(df):,}"
    )

    print(
        f"Customers : "
        f"{df['user_id'].nunique():,}"
    )

    target = "next_reorder_rate"

    # ========================================================
    # 2. Sort chronologically
    # ========================================================

    section(
        "2. Sorting prediction points"
    )

    df = df.sort_values(
        [
            "user_id",
            "prediction_order_number",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 3. Three-way temporal split
    # ========================================================

    section(
        "3. Creating three-way temporal split"
    )

    prediction_counts = (
        df.groupby("user_id")
        ["prediction_order_id"]
        .transform("count")
    )

    df["customer_prediction_count"] = (
        prediction_counts
    )

    eligible_customers = (
        df.loc[
            df["customer_prediction_count"] >= 3,
            "user_id",
        ]
        .unique()
    )

    df = df[
        df["user_id"].isin(
            eligible_customers
        )
    ].copy()

    print(
        f"Customers with >=3 prediction points: "
        f"{len(eligible_customers):,}"
    )

    reverse_rank = (
        df.groupby("user_id")
        ["prediction_order_number"]
        .rank(
            method="first",
            ascending=False,
        )
        - 1
    )

    df["reverse_rank"] = (
        reverse_rank.astype(int)
    )

    final_holdout = df[
        df["reverse_rank"] == 0
    ].copy()

    selection_holdout = df[
        df["reverse_rank"] == 1
    ].copy()

    train = df[
        df["reverse_rank"] >= 2
    ].copy()

    print(
        f"\nTraining rows          : "
        f"{len(train):,}"
    )

    print(
        f"Model-selection rows   : "
        f"{len(selection_holdout):,}"
    )

    print(
        f"Final holdout rows     : "
        f"{len(final_holdout):,}"
    )

    print(
        f"\nTraining customers     : "
        f"{train['user_id'].nunique():,}"
    )

    print(
        f"Selection customers    : "
        f"{selection_holdout['user_id'].nunique():,}"
    )

    print(
        f"Final holdout customers: "
        f"{final_holdout['user_id'].nunique():,}"
    )

    # ========================================================
    # 4. V3 behavioral features
    # ========================================================

    section(
        "4. Engineering behavioral features"
    )

    df["recent_5_order_reorder_rate"] = (
        df.groupby("user_id")
        ["previous_order_reorder_rate"]
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=1,
                ).mean()
        )
    )

    df["recent_5_order_basket"] = (
        df.groupby("user_id")
        ["previous_basket_size"]
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=1,
                ).mean()
        )
    )

    df["recent_5_order_interval"] = (
        df.groupby("user_id")
        ["current_order_days_since_prior"]
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=1,
                ).mean()
        )
    )

    df["reorder_rate_trend_3"] = (
        df["recent_3_order_reorder_rate"]
        -
        df["historical_reorder_rate"]
    )

    df["reorder_rate_trend_previous"] = (
        df["previous_order_reorder_rate"]
        -
        df["historical_reorder_rate"]
    )

    df["basket_size_trend_3"] = (
        df["recent_3_order_basket"]
        -
        df["historical_avg_basket_size"]
    )

    df["basket_size_trend_previous"] = (
        df["previous_basket_size"]
        -
        df["historical_avg_basket_size"]
    )

    df["reorder_rate_std_5"] = (
        df.groupby("user_id")
        ["previous_order_reorder_rate"]
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=2,
                ).std()
        )
    )

    df["reorder_rate_std_3"] = (
        df.groupby("user_id")
        ["previous_order_reorder_rate"]
        .transform(
            lambda x:
                x.rolling(
                    window=3,
                    min_periods=2,
                ).std()
        )
    )

    df["basket_size_std_5"] = (
        df.groupby("user_id")
        ["previous_basket_size"]
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=2,
                ).std()
        )
    )

    df["interval_trend"] = (
        df["recent_3_order_interval"]
        -
        df["historical_avg_days_between_orders"]
    )

    v3_features = [

        "recent_5_order_reorder_rate",

        "recent_5_order_basket",

        "recent_5_order_interval",

        "reorder_rate_trend_3",

        "reorder_rate_trend_previous",

        "basket_size_trend_3",

        "basket_size_trend_previous",

        "reorder_rate_std_5",

        "reorder_rate_std_3",

        "basket_size_std_5",

        "interval_trend",
    ]

    for column in v3_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # Recreate splits.

    train = df[
        df["reverse_rank"] >= 2
    ].copy()

    selection_holdout = df[
        df["reverse_rank"] == 1
    ].copy()

    final_holdout = df[
        df["reverse_rank"] == 0
    ].copy()

    # ========================================================
    # 5. Features
    # ========================================================

    section(
        "5. Selecting features"
    )

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

        "order_hour_of_day",
    ]

    feature_columns = (
        base_features +
        v3_features
    )

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # 6. Prepare matrices
    # ========================================================

    section(
        "6. Preparing datasets"
    )

    X_train = train[
        feature_columns
    ].copy()

    X_selection = selection_holdout[
        feature_columns
    ].copy()

    X_final = final_holdout[
        feature_columns
    ].copy()

    y_train = train[
        target
    ].astype(float)

    y_selection = selection_holdout[
        target
    ].astype(float)

    y_final = final_holdout[
        target
    ].astype(float)

    for column in feature_columns:

        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )

        X_selection[column] = pd.to_numeric(
            X_selection[column],
            errors="coerce",
        )

        X_final[column] = pd.to_numeric(
            X_final[column],
            errors="coerce",
        )

    X_train = (
        X_train
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    X_selection = (
        X_selection
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    X_final = (
        X_final
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    # ========================================================
    # 7. Basket-size-aware sample weights
    # ========================================================

    section(
        "7. Creating basket-size-aware training weights"
    )

    # Larger baskets provide a more stable estimate of
    # reorder rate than very small baskets.
    #
    # Weight is deliberately capped so large baskets
    # cannot dominate training.

    basket_weights = np.sqrt(
        np.maximum(
            train["next_basket_size"].astype(float),
            1,
        )
    )

    basket_weights = np.clip(
        basket_weights,
        1.0,
        3.0,
    )

    basket_weights = (
        basket_weights /
        basket_weights.mean()
    )

    print(
        f"Weight minimum: "
        f"{basket_weights.min():.4f}"
    )

    print(
        f"Weight maximum: "
        f"{basket_weights.max():.4f}"
    )

    print(
        f"Weight mean: "
        f"{basket_weights.mean():.4f}"
    )

    # ========================================================
    # 8. Train candidate models
    # ========================================================

    section(
        "8. Training V4 candidate models"
    )

    candidates = []

    # --------------------------------------------------------
    # Candidate A
    # Standard V3 Deep
    # --------------------------------------------------------

    print(
        "\nCandidate A: "
        "Standard XGBoost V3 Deep"
    )

    model_standard = make_xgb(
        objective="reg:absoluteerror",
        max_depth=8,
    )

    model_standard.fit(
        X_train,
        y_train,
        verbose=False,
    )

    selection_standard = np.clip(
        model_standard.predict(
            X_selection
        ),
        0,
        1,
    )

    candidates.append(
        (
            "Standard XGBoost V3 Deep",
            model_standard,
            selection_standard,
        )
    )

    # --------------------------------------------------------
    # Candidate B
    # Logit target
    # --------------------------------------------------------

    print(
        "\nCandidate B: "
        "Logit-Target XGBoost"
    )

    y_train_logit = (
        target_to_logit(
            y_train.to_numpy()
        )
    )

    model_logit = make_xgb(
        objective="reg:squarederror",
        max_depth=8,
    )

    model_logit.fit(
        X_train,
        y_train_logit,
        verbose=False,
    )

    selection_logit_raw = (
        model_logit.predict(
            X_selection
        )
    )

    selection_logit = (
        logit_to_target(
            selection_logit_raw
        )
    )

    candidates.append(
        (
            "Logit-Target XGBoost",
            model_logit,
            selection_logit,
        )
    )

    # --------------------------------------------------------
    # Candidate C
    # Logit + basket weights
    # --------------------------------------------------------

    print(
        "\nCandidate C: "
        "Logit-Target + Basket Weights"
    )

    model_logit_weighted = make_xgb(
        objective="reg:squarederror",
        max_depth=8,
    )

    model_logit_weighted.fit(
        X_train,
        y_train_logit,
        sample_weight=basket_weights,
        verbose=False,
    )

    selection_logit_weighted_raw = (
        model_logit_weighted.predict(
            X_selection
        )
    )

    selection_logit_weighted = (
        logit_to_target(
            selection_logit_weighted_raw
        )
    )

    candidates.append(
        (
            "Logit + Basket-Weighted XGBoost",
            model_logit_weighted,
            selection_logit_weighted,
        )
    )

    # --------------------------------------------------------
    # Candidate D
    # Absolute-error + basket weights
    # --------------------------------------------------------

    print(
        "\nCandidate D: "
        "MAE XGBoost + Basket Weights"
    )

    model_weighted = make_xgb(
        objective="reg:absoluteerror",
        max_depth=8,
    )

    model_weighted.fit(
        X_train,
        y_train,
        sample_weight=basket_weights,
        verbose=False,
    )

    selection_weighted = np.clip(
        model_weighted.predict(
            X_selection
        ),
        0,
        1,
    )

    candidates.append(
        (
            "MAE XGBoost + Basket Weights",
            model_weighted,
            selection_weighted,
        )
    )

    # ========================================================
    # 9. Selection evaluation
    # ========================================================

    section(
        "9. Candidate evaluation on selection holdout"
    )

    actual_selection = (
        y_selection.to_numpy()
    )

    selection_results = []

    for name, model, prediction in candidates:

        mae, rmse, r2 = calculate_metrics(
            actual_selection,
            prediction,
        )

        selection_results.append({

            "model":
                name,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,
        })

        print(
            f"\n{name}"
        )

        print(
            f"MAE : {mae:.6f}"
        )

        print(
            f"RMSE: {rmse:.6f}"
        )

        print(
            f"R²  : {r2:.6f}"
        )

    selection_df = (
        pd.DataFrame(
            selection_results
        )
        .sort_values(
            "MAE",
            ascending=True,
        )
        .reset_index(
            drop=True,
        )
    )

    # ========================================================
    # 10. Select candidate
    # ========================================================

    section(
        "10. Selecting V4 candidate"
    )

    best_name = (
        selection_df.iloc[0]["model"]
    )

    print(
        f"Selected candidate: "
        f"{best_name}"
    )

    # ========================================================
    # 11. Find selected candidate object
    # ========================================================

    selected_tuple = next(
        item
        for item in candidates
        if item[0] == best_name
    )

    selected_model = (
        selected_tuple[1]
    )

    # ========================================================
    # 12. Final prediction
    # ========================================================

    section(
        "11. Generating final holdout prediction"
    )

    # Important:
    # The final holdout has never been used to select
    # the model.

    if best_name == (
        "Logit-Target XGBoost"
    ):

        final_raw = (
            selected_model.predict(
                X_final
            )
        )

        final_prediction = (
            logit_to_target(
                final_raw
            )
        )

    elif best_name == (
        "Logit + Basket-Weighted XGBoost"
    ):

        final_raw = (
            selected_model.predict(
                X_final
            )
        )

        final_prediction = (
            logit_to_target(
                final_raw
            )
        )

    else:

        final_prediction = np.clip(
            selected_model.predict(
                X_final
            ),
            0,
            1,
        )

    # ========================================================
    # 13. Final evaluation
    # ========================================================

    section(
        "12. FINAL UNTOUCHED HOLDOUT RESULTS"
    )

    actual_final = (
        y_final.to_numpy()
    )

    # Baselines.

    historical_final = (
        final_holdout[
            "historical_reorder_rate"
        ]
        .to_numpy()
    )

    recent_3_final = (
        final_holdout[
            "recent_3_order_reorder_rate"
        ]
        .to_numpy()
    )

    final_candidates = {

        "Historical Reorder Rate":
            historical_final,

        "Recent 3-Order Reorder Rate":
            recent_3_final,

        "Selected V4 Model":
            final_prediction,
    }

    final_results = []

    for name, prediction in (
        final_candidates.items()
    ):

        mae, rmse, r2 = calculate_metrics(
            actual_final,
            prediction,
        )

        final_results.append({

            "evaluation":
                "Final Untouched Holdout",

            "model":
                name,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,
        })

        print(
            f"\n{name}"
        )

        print(
            f"MAE : {mae:.6f}"
        )

        print(
            f"RMSE: {rmse:.6f}"
        )

        print(
            f"R²  : {r2:.6f}"
        )

    # ========================================================
    # 14. Final comparison with V3
    # ========================================================

    section(
        "13. FINAL COMPARISON WITH LOCKED V3"
    )

    final_df = pd.DataFrame(
        final_results
    )

    v4_row = final_df[
        final_df["model"]
        ==
        "Selected V4 Model"
    ].iloc[0]

    v4_mae = v4_row["MAE"]
    v4_rmse = v4_row["RMSE"]
    v4_r2 = v4_row["R2"]

    # Locked V3 final holdout results.

    v3_mae = 0.203614
    v3_rmse = 0.263450
    v3_r2 = 0.253438

    mae_change = (
        v4_mae -
        v3_mae
    )

    rmse_change = (
        v4_rmse -
        v3_rmse
    )

    r2_change = (
        v4_r2 -
        v3_r2
    )

    mae_improvement = (
        (
            v3_mae -
            v4_mae
        )
        /
        v3_mae
        *
        100
    )

    print(
        f"Locked V3 MAE : "
        f"{v3_mae:.6f}"
    )

    print(
        f"V4 MAE        : "
        f"{v4_mae:.6f}"
    )

    print(
        f"\nMAE change:"
        f" {mae_change:+.6f}"
    )

    print(
        f"MAE improvement:"
        f" {mae_improvement:+.3f}%"
    )

    print(
        f"\nLocked V3 RMSE:"
        f" {v3_rmse:.6f}"
    )

    print(
        f"V4 RMSE:"
        f" {v4_rmse:.6f}"
    )

    print(
        f"RMSE change:"
        f" {rmse_change:+.6f}"
    )

    print(
        f"\nLocked V3 R²:"
        f" {v3_r2:.6f}"
    )

    print(
        f"V4 R²:"
        f" {v4_r2:.6f}"
    )

    print(
        f"R² change:"
        f" {r2_change:+.6f}"
    )

    # ========================================================
    # 15. Prediction distribution
    # ========================================================

    section(
        "14. V4 prediction distribution"
    )

    print(
        pd.Series(
            final_prediction,
            name="predicted_reorder_rate",
        )
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 16. Feature importance
    # ========================================================

    section(
        "15. Feature importance"
    )

    importance_df = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            selected_model.feature_importances_,
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )

    print(
        importance_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # 17. Save
    # ========================================================

    section(
        "16. Saving results"
    )

    final_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    print(
        f"Results saved:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {FEATURE_IMPORTANCE_PATH}"
    )

    # ========================================================
    # 18. Final decision
    # ========================================================

    section(
        "FINAL MODEL 2 V4 DECISION"
    )

    print(
        f"V4 selected candidate:"
        f" {best_name}"
    )

    print(
        f"V4 final MAE:"
        f" {v4_mae:.6f}"
    )

    print(
        f"V4 final RMSE:"
        f" {v4_rmse:.6f}"
    )

    print(
        f"V4 final R²:"
        f" {v4_r2:.6f}"
    )

    if v4_mae < v3_mae:

        print(
            "\nV4 improves MAE over locked V3."
        )

    else:

        print(
            "\nV4 does NOT improve MAE over "
            "locked V3."
        )

    print(
        "\nFinal holdout remained untouched "
        "during model selection."
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()