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
# MODEL 3 V1 — FUTURE BASKET SIZE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

INPUT_PATH = (
    PROCESSED_DIR /
    "forecasting_dataset.parquet"
)

RESULTS_PATH = (
    PROCESSED_DIR /
    "basket_size_forecast_v1_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "basket_size_forecast_v1_feature_importance.csv"
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

    # Basket size cannot be negative.
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

    predictions = np.expm1(
        values
    )

    return np.maximum(
        predictions,
        0,
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "MODEL 3 V1 — FUTURE BASKET SIZE"
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

    if target not in df.columns:

        raise ValueError(
            f"Target '{target}' "
            f"not found."
        )

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
    # 4. Historical behavioral features
    # ========================================================

    section(
        "4. Engineering leakage-safe behavioral features"
    )

    # --------------------------------------------------------
    # Same valid 11-feature behavioral set used by Model 2.
    #
    # These are constructed from historical information only.
    # --------------------------------------------------------

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

    behavioral_features = [

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

    for column in behavioral_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    print(
        f"\nCreated {len(behavioral_features)} "
        f"historical behavioral features."
    )

    # ========================================================
    # 5. Recreate splits after feature engineering
    # ========================================================

    section(
        "5. Recreating temporal splits"
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

    print(
        f"\nTraining customers     : "
        f"{train['user_id'].nunique():,}"
    )

    print(
        f"Selection customers    : "
        f"{selection['user_id'].nunique():,}"
    )

    print(
        f"Final holdout customers: "
        f"{final_holdout['user_id'].nunique():,}"
    )

    # ========================================================
    # 6. Feature selection
    # ========================================================

    section(
        "6. Selecting Model 3 features"
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
        behavioral_features
    )

    print(
        f"Base features : "
        f"{len(base_features)}"
    )

    print(
        f"Behavioral    : "
        f"{len(behavioral_features)}"
    )

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Leakage validation
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
    # 7. Prepare matrices
    # ========================================================

    section(
        "7. Preparing model matrices"
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
    # 8. Baselines
    # ========================================================

    section(
        "8. Baseline models"
    )

    # --------------------------------------------------------
    # Baseline A: Historical average basket
    # --------------------------------------------------------

    historical_selection = (
        selection[
            "historical_avg_basket_size"
        ]
        .fillna(0)
        .to_numpy()
    )

    historical_metrics = (
        calculate_metrics(
            y_selection.to_numpy(),
            historical_selection,
        )
    )

    print(
        "\nHistorical Average Basket:"
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

    # --------------------------------------------------------
    # Baseline B: Recent-3 average basket
    # --------------------------------------------------------

    recent3_selection = (
        selection[
            "recent_3_order_basket"
        ]
        .fillna(0)
        .to_numpy()
    )

    recent3_metrics = (
        calculate_metrics(
            y_selection.to_numpy(),
            recent3_selection,
        )
    )

    print(
        "\nRecent-3 Average Basket:"
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

    # --------------------------------------------------------
    # Baseline C: Previous basket
    # --------------------------------------------------------

    previous_selection = (
        selection[
            "previous_basket_size"
        ]
        .fillna(0)
        .to_numpy()
    )

    previous_metrics = (
        calculate_metrics(
            y_selection.to_numpy(),
            previous_selection,
        )
    )

    print(
        "\nPrevious Basket:"
    )

    print(
        f"MAE  : "
        f"{previous_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : "
        f"{previous_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : "
        f"{previous_metrics['R2']:.6f}"
    )

    # ========================================================
    # 9. Train candidate models
    # ========================================================

    section(
        "9. Training Model 3 candidate models"
    )

    candidates = {}

    # --------------------------------------------------------
    # Candidate A — Square Error
    # --------------------------------------------------------

    print(
        "\nTraining: Model 3 Standard"
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

    pred_standard = (
        model_standard.predict(
            X_selection
        )
    )

    pred_standard = np.maximum(
        pred_standard,
        0,
    )

    candidates[
        "Model 3 Standard"
    ] = (
        model_standard,
        pred_standard,
    )

    metrics = calculate_metrics(
        y_selection.to_numpy(),
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
        "\nTraining: Model 3 MAE Objective"
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

    pred_mae = (
        model_mae.predict(
            X_selection
        )
    )

    pred_mae = np.maximum(
        pred_mae,
        0,
    )

    candidates[
        "Model 3 MAE Objective"
    ] = (
        model_mae,
        pred_mae,
    )

    metrics = calculate_metrics(
        y_selection.to_numpy(),
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
    # Candidate C — Log target
    # --------------------------------------------------------

    print(
        "\nTraining: Model 3 Log-Target"
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

    pred_log_raw = (
        model_log.predict(
            X_selection
        )
    )

    pred_log = log_to_target(
        pred_log_raw
    )

    candidates[
        "Model 3 Log-Target"
    ] = (
        model_log,
        pred_log,
    )

    metrics = calculate_metrics(
        y_selection.to_numpy(),
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
    # 10. Candidate evaluation
    # ========================================================

    section(
        "10. MODEL-SELECTION COMPARISON"
    )

    actual_selection = (
        y_selection.to_numpy()
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
    # 11. Select best model
    # ========================================================

    section(
        "11. SELECTING MODEL 3"
    )

    # Only ML candidates can be selected as the final
    # production model. Baselines are benchmarks.

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
    # 12. Final untouched holdout
    # ========================================================

    section(
        "12. FINAL UNTOUCHED HOLDOUT"
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
    # Selected model prediction
    # --------------------------------------------------------

    if best_name == (
        "Model 3 Log-Target"
    ):

        final_raw = (
            best_model.predict(
                X_final
            )
        )

        final_prediction = (
            log_to_target(
                final_raw
            )
        )

    else:

        final_prediction = (
            best_model.predict(
                X_final
            )
        )

        final_prediction = np.maximum(
            final_prediction,
            0,
        )

    # --------------------------------------------------------
    # Evaluate all
    # --------------------------------------------------------

    final_predictions = {

        "Historical Average Basket":
            historical_final,

        "Recent-3 Average Basket":
            recent3_final,

        "Previous Basket":
            previous_final,

        "Selected Model 3":
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
    # 13. Model improvement
    # ========================================================

    section(
        "13. MODEL 3 IMPROVEMENT"
    )

    selected_row = final_df[
        final_df[
            "model"
        ] ==
        "Selected Model 3"
    ].iloc[0]

    selected_mae = (
        selected_row["MAE"]
    )

    selected_rmse = (
        selected_row["RMSE"]
    )

    selected_r2 = (
        selected_row["R2"]
    )

    recent3_mae = (
        final_df[
            final_df[
                "model"
            ]
            ==
            "Recent-3 Average Basket"
        ]
        .iloc[0]["MAE"]
    )

    recent3_rmse = (
        final_df[
            final_df[
                "model"
            ]
            ==
            "Recent-3 Average Basket"
        ]
        .iloc[0]["RMSE"]
    )

    mae_improvement = (
        (
            recent3_mae
            -
            selected_mae
        )
        /
        recent3_mae
        *
        100
    )

    rmse_improvement = (
        (
            recent3_rmse
            -
            selected_rmse
        )
        /
        recent3_rmse
        *
        100
    )

    print(
        f"Selected Model 3 MAE:"
        f" {selected_mae:.6f}"
    )

    print(
        f"Recent-3 baseline MAE:"
        f" {recent3_mae:.6f}"
    )

    print(
        f"\nMAE improvement vs Recent-3:"
        f" {mae_improvement:+.3f}%"
    )

    print(
        f"\nSelected Model 3 RMSE:"
        f" {selected_rmse:.6f}"
    )

    print(
        f"Recent-3 baseline RMSE:"
        f" {recent3_rmse:.6f}"
    )

    print(
        f"\nRMSE improvement vs Recent-3:"
        f" {rmse_improvement:+.3f}%"
    )

    # ========================================================
    # 14. Prediction distribution
    # ========================================================

    section(
        "14. FINAL PREDICTION DISTRIBUTION"
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
    # 15. Feature importance
    # ========================================================

    section(
        "15. FINAL FEATURE IMPORTANCE"
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
    # 16. Save results
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
        "\nResults saved:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {FEATURE_IMPORTANCE_PATH}"
    )

    # ========================================================
    # 17. Final summary
    # ========================================================

    section(
        "FINAL MODEL 3 SUMMARY"
    )

    print(
        f"Selected model:"
        f" {best_name}"
    )

    print(
        f"\nFinal MAE:"
        f" {selected_mae:.6f}"
    )

    print(
        f"Final RMSE:"
        f" {selected_rmse:.6f}"
    )

    print(
        f"Final R2:"
        f" {selected_r2:.6f}"
    )

    print(
        "\nLeakage check: PASSED"
    )

    print(
        "next_basket_size was not used "
        "as a feature."
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