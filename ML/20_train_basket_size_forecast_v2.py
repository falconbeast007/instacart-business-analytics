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
# MODEL 3 V2 — FUTURE BASKET SIZE
# Basket-Dynamics Improvement
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

INPUT_PATH = (
    PROCESSED_DIR /
    "forecasting_dataset.parquet"
)

RESULTS_PATH = (
    PROCESSED_DIR /
    "basket_size_forecast_v2_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "basket_size_forecast_v2_feature_importance.csv"
)

RANDOM_STATE = 42


# ============================================================
# Helpers
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(actual, predicted):

    predicted = np.maximum(
        predicted,
        0,
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

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def make_xgb(
    objective="reg:squarederror",
    max_depth=8,
    n_estimators=500,
    learning_rate=0.05,
    min_child_weight=5,
):

    return XGBRegressor(

        n_estimators=n_estimators,

        max_depth=max_depth,

        learning_rate=learning_rate,

        min_child_weight=min_child_weight,

        subsample=0.8,

        colsample_bytree=0.8,

        objective=objective,

        eval_metric="mae",

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )


def target_to_log(values):

    values = np.maximum(
        values,
        0,
    )

    return np.log1p(
        values
    )


def log_to_target(values):

    return np.maximum(
        np.expm1(values),
        0,
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "MODEL 3 V2 — FUTURE BASKET SIZE"
    )

    # ========================================================
    # 1. Load dataset
    # ========================================================

    section(
        "1. Loading forecasting dataset"
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Forecasting dataset not found:\n"
            f"{INPUT_PATH}"
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

    target = "next_basket_size"

    print(
        f"Target    : {target}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df[target]
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 2. Sort chronologically
    # ========================================================

    section(
        "2. Sorting prediction points"
    )

    df = (
        df.sort_values(
            [
                "user_id",
                "prediction_order_number",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 3. Three-way temporal split
    # ========================================================

    section(
        "3. Creating three-way temporal split"
    )

    prediction_counts = (
        df.groupby(
            "user_id"
        )[
            "prediction_order_id"
        ]
        .transform("count")
    )

    df[
        "customer_prediction_count"
    ] = prediction_counts

    eligible_customers = (
        df.loc[
            df[
                "customer_prediction_count"
            ] >= 3,
            "user_id",
        ]
        .unique()
    )

    df = df[
        df[
            "user_id"
        ].isin(
            eligible_customers
        )
    ].copy()

    print(
        f"Customers with >=3 prediction points: "
        f"{len(eligible_customers):,}"
    )

    reverse_rank = (
        df.groupby(
            "user_id"
        )[
            "prediction_order_number"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        - 1
    )

    df[
        "reverse_rank"
    ] = reverse_rank.astype(int)

    print(
        "\nTemporal split:"
    )

    print(
        "  Training      : reverse_rank >= 2"
    )

    print(
        "  Selection     : reverse_rank == 1"
    )

    print(
        "  Final holdout : reverse_rank == 0"
    )

    # ========================================================
    # 4. V1 behavioral features
    # ========================================================

    section(
        "4. Creating V1 behavioral features"
    )

    df[
        "recent_5_order_reorder_rate"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).mean()
        )
    )

    df[
        "recent_5_order_basket"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).mean()
        )
    )

    df[
        "recent_5_order_interval"
    ] = (
        df.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).mean()
        )
    )

    df[
        "reorder_rate_trend_3"
    ] = (
        df[
            "recent_3_order_reorder_rate"
        ]
        -
        df[
            "historical_reorder_rate"
        ]
    )

    df[
        "reorder_rate_trend_previous"
    ] = (
        df[
            "previous_order_reorder_rate"
        ]
        -
        df[
            "historical_reorder_rate"
        ]
    )

    df[
        "basket_size_trend_3"
    ] = (
        df[
            "recent_3_order_basket"
        ]
        -
        df[
            "historical_avg_basket_size"
        ]
    )

    df[
        "basket_size_trend_previous"
    ] = (
        df[
            "previous_basket_size"
        ]
        -
        df[
            "historical_avg_basket_size"
        ]
    )

    df[
        "reorder_rate_std_5"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=2,
            ).std()
        )
    )

    df[
        "reorder_rate_std_3"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=3,
                min_periods=2,
            ).std()
        )
    )

    df[
        "basket_size_std_5"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=2,
            ).std()
        )
    )

    df[
        "interval_trend"
    ] = (
        df[
            "recent_3_order_interval"
        ]
        -
        df[
            "historical_avg_days_between_orders"
        ]
    )

    v1_behavioral_features = [

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

    for column in v1_behavioral_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # ========================================================
    # 5. V2 basket-dynamics features
    # ========================================================

    section(
        "5. Creating V2 basket-dynamics features"
    )

    # --------------------------------------------------------
    # Recent 2-order basket average
    # --------------------------------------------------------

    df[
        "recent_2_order_basket"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=2,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent 10-order basket average
    # --------------------------------------------------------

    df[
        "recent_10_order_basket"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=10,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent basket standard deviation
    # --------------------------------------------------------

    df[
        "basket_size_std_3"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=3,
                min_periods=2,
            ).std()
        )
    )

    # --------------------------------------------------------
    # Recent 10-order basket standard deviation
    # --------------------------------------------------------

    df[
        "basket_size_std_10"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=10,
                min_periods=2,
            ).std()
        )
    )

    # --------------------------------------------------------
    # Recent vs historical ratio
    # --------------------------------------------------------

    historical_safe = (
        df[
            "historical_avg_basket_size"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    df[
        "recent_3_basket_ratio"
    ] = (
        df[
            "recent_3_order_basket"
        ]
        /
        historical_safe
    )

    df[
        "recent_5_basket_ratio"
    ] = (
        df[
            "recent_5_order_basket"
        ]
        /
        historical_safe
    )

    df[
        "recent_10_basket_ratio"
    ] = (
        df[
            "recent_10_order_basket"
        ]
        /
        historical_safe
    )

    # --------------------------------------------------------
    # Recent trend
    #
    # Positive = baskets getting larger.
    # Negative = baskets getting smaller.
    # --------------------------------------------------------

    df[
        "basket_recent_trend_2_vs_5"
    ] = (
        df[
            "recent_2_order_basket"
        ]
        -
        df[
            "recent_5_order_basket"
        ]
    )

    df[
        "basket_recent_trend_3_vs_10"
    ] = (
        df[
            "recent_3_order_basket"
        ]
        -
        df[
            "recent_10_order_basket"
        ]
    )

    # --------------------------------------------------------
    # Previous basket deviation from recent average
    # --------------------------------------------------------

    df[
        "previous_basket_vs_recent3"
    ] = (
        df[
            "previous_basket_size"
        ]
        -
        df[
            "recent_3_order_basket"
        ]
    )

    # --------------------------------------------------------
    # Basket consistency
    #
    # Lower coefficient of variation = more consistent basket.
    # --------------------------------------------------------

    recent3_std_safe = (
        df[
            "basket_size_std_3"
        ]
        .fillna(0)
    )

    recent3_mean_safe = (
        df[
            "recent_3_order_basket"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    df[
        "basket_cv_3"
    ] = (
        recent3_std_safe
        /
        recent3_mean_safe
    )

    # --------------------------------------------------------
    # Historical basket range proxy
    # --------------------------------------------------------

    df[
        "recent_5_basket_range"
    ] = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).max()
        )
        -
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).min()
        )
    )

    v2_basket_features = [

        "recent_2_order_basket",

        "recent_10_order_basket",

        "basket_size_std_3",

        "basket_size_std_10",

        "recent_3_basket_ratio",

        "recent_5_basket_ratio",

        "recent_10_basket_ratio",

        "basket_recent_trend_2_vs_5",

        "basket_recent_trend_3_vs_10",

        "previous_basket_vs_recent3",

        "basket_cv_3",

        "recent_5_basket_range",
    ]

    for column in v2_basket_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    print(
        f"Created {len(v2_basket_features)} "
        f"new V2 basket features."
    )

    # ========================================================
    # 6. Recreate temporal splits
    # ========================================================

    section(
        "6. Recreating temporal splits"
    )

    train = df[
        df["reverse_rank"] >= 2
    ].copy()

    selection = df[
        df["reverse_rank"] == 1
    ].copy()

    final_holdout = df[
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
        f"{len(final_holdout):,}"
    )

    # ========================================================
    # 7. Feature set
    # ========================================================

    section(
        "7. Selecting Model 3 V2 features"
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
        base_features
        +
        v1_behavioral_features
        +
        v2_basket_features
    )

    print(
        f"Base features       : "
        f"{len(base_features)}"
    )

    print(
        f"V1 behavioral       : "
        f"{len(v1_behavioral_features)}"
    )

    print(
        f"V2 basket dynamics  : "
        f"{len(v2_basket_features)}"
    )

    print(
        f"Total features      : "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    forbidden_targets = [

        "next_basket_size",

        "next_reorder_rate",

        "next_order_interval",

        "next_order_within_horizon",
    ]

    leakage_columns = [
        column
        for column in feature_columns
        if column in forbidden_targets
    ]

    if leakage_columns:

        raise ValueError(
            "LEAKAGE DETECTED:\n"
            +
            "\n".join(
                leakage_columns
            )
        )

    print(
        "Feature leakage check: PASSED"
    )

    # ========================================================
    # 8. Prepare matrices
    # ========================================================

    section(
        "8. Preparing model matrices"
    )

    X_train = train[
        feature_columns
    ].copy()

    X_selection = selection[
        feature_columns
    ].copy()

    X_final = final_holdout[
        feature_columns
    ].copy()

    y_train = (
        train[
            target
        ]
        .astype(float)
    )

    y_selection = (
        selection[
            target
        ]
        .astype(float)
    )

    y_final = (
        final_holdout[
            target
        ]
        .astype(float)
    )

    for X in [
        X_train,
        X_selection,
        X_final,
    ]:

        for column in feature_columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        X.fillna(
            0,
            inplace=True,
        )

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

    # ========================================================
    # 9. Baselines
    # ========================================================

    section(
        "9. Baseline models"
    )

    actual_selection = (
        y_selection.to_numpy()
    )

    # --------------------------------------------------------
    # Historical average
    # --------------------------------------------------------

    historical_selection = (
        selection[
            "historical_avg_basket_size"
        ]
        .fillna(0)
        .to_numpy()
    )

    historical_metrics = calculate_metrics(
        actual_selection,
        historical_selection,
    )

    print(
        "\nHistorical Average Basket:"
    )

    print(
        f"MAE  : {historical_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {historical_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {historical_metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Recent-3
    # --------------------------------------------------------

    recent3_selection = (
        selection[
            "recent_3_order_basket"
        ]
        .fillna(0)
        .to_numpy()
    )

    recent3_metrics = calculate_metrics(
        actual_selection,
        recent3_selection,
    )

    print(
        "\nRecent-3 Average Basket:"
    )

    print(
        f"MAE  : {recent3_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {recent3_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {recent3_metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Previous basket
    # --------------------------------------------------------

    previous_selection = (
        selection[
            "previous_basket_size"
        ]
        .fillna(0)
        .to_numpy()
    )

    previous_metrics = calculate_metrics(
        actual_selection,
        previous_selection,
    )

    print(
        "\nPrevious Basket:"
    )

    print(
        f"MAE  : {previous_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {previous_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {previous_metrics['R2']:.6f}"
    )

    # ========================================================
    # 10. Candidate models
    # ========================================================

    section(
        "10. Training Model 3 V2 candidates"
    )

    candidates = {}

    # --------------------------------------------------------
    # Candidate A — Standard
    # --------------------------------------------------------

    print(
        "\nTraining: V2 Standard"
    )

    model_standard = make_xgb(
        objective="reg:squarederror",
        max_depth=8,
    )

    model_standard.fit(
        X_train,
        y_train,
        verbose=False,
    )

    pred_standard = np.maximum(
        model_standard.predict(
            X_selection
        ),
        0,
    )

    candidates[
        "V2 Standard"
    ] = (
        model_standard,
        pred_standard,
    )

    metrics = calculate_metrics(
        actual_selection,
        pred_standard,
    )

    print(
        f"MAE  : {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Candidate B — MAE objective
    # --------------------------------------------------------

    print(
        "\nTraining: V2 MAE Objective"
    )

    model_mae = make_xgb(
        objective="reg:absoluteerror",
        max_depth=8,
    )

    model_mae.fit(
        X_train,
        y_train,
        verbose=False,
    )

    pred_mae = np.maximum(
        model_mae.predict(
            X_selection
        ),
        0,
    )

    candidates[
        "V2 MAE Objective"
    ] = (
        model_mae,
        pred_mae,
    )

    metrics = calculate_metrics(
        actual_selection,
        pred_mae,
    )

    print(
        f"MAE  : {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Candidate C — Deeper
    # --------------------------------------------------------

    print(
        "\nTraining: V2 Deeper"
    )

    model_deeper = make_xgb(
        objective="reg:squarederror",
        max_depth=10,
    )

    model_deeper.fit(
        X_train,
        y_train,
        verbose=False,
    )

    pred_deeper = np.maximum(
        model_deeper.predict(
            X_selection
        ),
        0,
    )

    candidates[
        "V2 Deeper"
    ] = (
        model_deeper,
        pred_deeper,
    )

    metrics = calculate_metrics(
        actual_selection,
        pred_deeper,
    )

    print(
        f"MAE  : {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Candidate D — Regularized
    # --------------------------------------------------------

    print(
        "\nTraining: V2 Regularized"
    )

    model_regularized = make_xgb(
        objective="reg:squarederror",
        max_depth=8,
        n_estimators=600,
        learning_rate=0.04,
        min_child_weight=10,
    )

    model_regularized.fit(
        X_train,
        y_train,
        verbose=False,
    )

    pred_regularized = np.maximum(
        model_regularized.predict(
            X_selection
        ),
        0,
    )

    candidates[
        "V2 Regularized"
    ] = (
        model_regularized,
        pred_regularized,
    )

    metrics = calculate_metrics(
        actual_selection,
        pred_regularized,
    )

    print(
        f"MAE  : {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Candidate E — Log target
    # --------------------------------------------------------

    print(
        "\nTraining: V2 Log-Target"
    )

    model_log = make_xgb(
        objective="reg:squarederror",
        max_depth=8,
    )

    y_train_log = target_to_log(
        y_train.to_numpy()
    )

    model_log.fit(
        X_train,
        y_train_log,
        verbose=False,
    )

    pred_log = log_to_target(
        model_log.predict(
            X_selection
        )
    )

    candidates[
        "V2 Log-Target"
    ] = (
        model_log,
        pred_log,
    )

    metrics = calculate_metrics(
        actual_selection,
        pred_log,
    )

    print(
        f"MAE  : {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : {metrics['R2']:.6f}"
    )

    # ========================================================
    # 11. Candidate comparison
    # ========================================================

    section(
        "11. MODEL-SELECTION COMPARISON"
    )

    selection_results = [

        {
            "model":
                "Historical Average Basket",

            "MAE":
                historical_metrics["MAE"],

            "RMSE":
                historical_metrics["RMSE"],

            "R2":
                historical_metrics["R2"],
        },

        {
            "model":
                "Recent-3 Average Basket",

            "MAE":
                recent3_metrics["MAE"],

            "RMSE":
                recent3_metrics["RMSE"],

            "R2":
                recent3_metrics["R2"],
        },

        {
            "model":
                "Previous Basket",

            "MAE":
                previous_metrics["MAE"],

            "RMSE":
                previous_metrics["RMSE"],

            "R2":
                previous_metrics["R2"],
        },
    ]

    for name, (
        model,
        prediction,
    ) in candidates.items():

        metrics = calculate_metrics(
            actual_selection,
            prediction,
        )

        selection_results.append({

            "model":
                name,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"],
        })

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

    print(
        selection_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # 12. Select model
    # ========================================================

    section(
        "12. SELECTING MODEL 3 V2"
    )

    ml_selection_df = (
        selection_df[
            selection_df[
                "model"
            ].isin(
                candidates.keys()
            )
        ]
        .sort_values(
            "MAE",
            ascending=True,
        )
        .reset_index(
            drop=True,
        )
    )

    best_name = (
        ml_selection_df.iloc[
            0
        ]["model"]
    )

    best_model = (
        candidates[
            best_name
        ][0]
    )

    print(
        f"Selected model: "
        f"{best_name}"
    )

    # ========================================================
    # 13. Final untouched holdout
    # ========================================================

    section(
        "13. FINAL UNTOUCHED HOLDOUT"
    )

    actual_final = (
        y_final.to_numpy()
    )

    # --------------------------------------------------------
    # Baseline predictions
    # --------------------------------------------------------

    historical_final = (
        final_holdout[
            "historical_avg_basket_size"
        ]
        .fillna(0)
        .to_numpy()
    )

    recent3_final = (
        final_holdout[
            "recent_3_order_basket"
        ]
        .fillna(0)
        .to_numpy()
    )

    previous_final = (
        final_holdout[
            "previous_basket_size"
        ]
        .fillna(0)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Selected model
    # --------------------------------------------------------

    if best_name == (
        "V2 Log-Target"
    ):

        final_prediction = (
            log_to_target(
                best_model.predict(
                    X_final
                )
            )
        )

    else:

        final_prediction = np.maximum(
            best_model.predict(
                X_final
            ),
            0,
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    final_predictions = {

        "Historical Average Basket":
            historical_final,

        "Recent-3 Average Basket":
            recent3_final,

        "Previous Basket":
            previous_final,

        "Selected Model 3 V2":
            final_prediction,
    }

    final_results = []

    for name, prediction in (
        final_predictions.items()
    ):

        metrics = calculate_metrics(
            actual_final,
            prediction,
        )

        final_results.append({

            "evaluation":
                "Final Untouched Holdout",

            "model":
                name,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"],
        })

        print(
            f"\n{name}"
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

    final_df = pd.DataFrame(
        final_results
    )

    # ========================================================
    # 14. Compare V1 vs V2
    # ========================================================

    section(
        "14. V1 VS V2 COMPARISON"
    )

    # V1 final result.
    v1_mae = 3.914726
    v1_rmse = 5.668096
    v1_r2 = 0.485664

    v2_row = final_df[
        final_df[
            "model"
        ]
        ==
        "Selected Model 3 V2"
    ].iloc[0]

    v2_mae = (
        v2_row["MAE"]
    )

    v2_rmse = (
        v2_row["RMSE"]
    )

    v2_r2 = (
        v2_row["R2"]
    )

    mae_improvement = (
        (
            v1_mae
            -
            v2_mae
        )
        /
        v1_mae
        *
        100
    )

    rmse_improvement = (
        (
            v1_rmse
            -
            v2_rmse
        )
        /
        v1_rmse
        *
        100
    )

    r2_change = (
        v2_r2
        -
        v1_r2
    )

    print(
        f"V1 MAE:"
        f" {v1_mae:.6f}"
    )

    print(
        f"V2 MAE:"
        f" {v2_mae:.6f}"
    )

    print(
        f"\nMAE improvement:"
        f" {mae_improvement:+.3f}%"
    )

    print(
        f"\nV1 RMSE:"
        f" {v1_rmse:.6f}"
    )

    print(
        f"V2 RMSE:"
        f" {v2_rmse:.6f}"
    )

    print(
        f"\nRMSE improvement:"
        f" {rmse_improvement:+.3f}%"
    )

    print(
        f"\nV1 R2:"
        f" {v1_r2:.6f}"
    )

    print(
        f"V2 R2:"
        f" {v2_r2:.6f}"
    )

    print(
        f"\nR2 change:"
        f" {r2_change:+.6f}"
    )

    # ========================================================
    # 15. Prediction distribution
    # ========================================================

    section(
        "15. FINAL PREDICTION DISTRIBUTION"
    )

    print(
        pd.Series(
            final_prediction,
            name="predicted_basket_size",
        )
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 16. Feature importance
    # ========================================================

    section(
        "16. FINAL FEATURE IMPORTANCE"
    )

    importance_df = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            best_model.feature_importances_,
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
    # 17. Save results
    # ========================================================

    section(
        "17. Saving results"
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
        "\nResults saved:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {FEATURE_IMPORTANCE_PATH}"
    )

    # ========================================================
    # 18. Final summary
    # ========================================================

    section(
        "FINAL MODEL 3 V2 SUMMARY"
    )

    print(
        f"Selected model:"
        f" {best_name}"
    )

    print(
        f"\nFinal MAE:"
        f" {v2_mae:.6f}"
    )

    print(
        f"Final RMSE:"
        f" {v2_rmse:.6f}"
    )

    print(
        f"Final R2:"
        f" {v2_r2:.6f}"
    )

    print(
        f"\nMAE improvement vs V1:"
        f" {mae_improvement:+.3f}%"
    )

    print(
        f"RMSE improvement vs V1:"
        f" {rmse_improvement:+.3f}%"
    )

    print(
        f"R2 change vs V1:"
        f" {r2_change:+.6f}"
    )

    print(
        "\nLeakage check: PASSED"
    )

    print(
        "No future target information "
        "was used in the features."
    )

    print(
        "\nFinal holdout was not used "
        "for model selection."
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":

    main()