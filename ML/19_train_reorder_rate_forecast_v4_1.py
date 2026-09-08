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
# MODEL 2 V4.1 — LEAKAGE-SAFE V4 IMPROVEMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

INPUT_PATH = (
    PROCESSED_DIR /
    "forecasting_dataset.parquet"
)

RESULTS_PATH = (
    PROCESSED_DIR /
    "reorder_rate_forecast_v4_1_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "reorder_rate_forecast_v4_1_feature_importance.csv"
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

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def make_xgb(
    objective="reg:absoluteerror",
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


def normalize_weights(weights):

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = np.nan_to_num(
        weights,
        nan=1.0,
        posinf=3.0,
        neginf=1.0,
    )

    weights = np.maximum(
        weights,
        1.0,
    )

    # Prevent any group from dominating the objective.
    weights = np.clip(
        weights,
        1.0,
        3.0,
    )

    # Normalize around 1.
    weights = (
        weights /
        weights.mean()
    )

    return weights


# ============================================================
# Main
# ============================================================

def main():

    section(
        "MODEL 2 V4.1 — LEAKAGE-SAFE V4 IMPROVEMENT"
    )

    # ========================================================
    # 1. Load forecasting dataset
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

    target = (
        "next_reorder_rate"
    )

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' "
            f"not found."
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

    # --------------------------------------------------------
    # Latest prediction point:
    #     reverse_rank = 0
    #
    # Selection:
    #     reverse_rank = 1
    #
    # Training:
    #     reverse_rank >= 2
    # --------------------------------------------------------

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
    # These features are based only on historical behavior.
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

    v4_features = [

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

    for column in v4_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    print(
        f"\nCreated {len(v4_features)} "
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
        "6. Selecting V4.1 features"
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
        v4_features
    )

    print(
        f"Base features : "
        f"{len(base_features)}"
    )

    print(
        f"V4 features   : "
        f"{len(v4_features)}"
    )

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Validate no future targets appear as features.
    # --------------------------------------------------------

    forbidden_columns = [

        "next_reorder_rate",

        "next_basket_size",

        "next_order_interval",

        "next_order_within_horizon",
    ]

    leakage_columns = [
        column
        for column in feature_columns
        if column in forbidden_columns
    ]

    if leakage_columns:

        raise ValueError(
            "LEAKAGE DETECTED IN FEATURES:\n"
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
    # Historical reorder-rate baseline
    # --------------------------------------------------------

    historical_selection_prediction = (
        selection[
            "historical_reorder_rate"
        ]
        .fillna(0)
        .to_numpy()
    )

    historical_selection_metrics = (
        calculate_metrics(
            y_selection.to_numpy(),
            historical_selection_prediction,
        )
    )

    print(
        "\nHistorical baseline:"
    )

    print(
        f"MAE  : "
        f"{historical_selection_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : "
        f"{historical_selection_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : "
        f"{historical_selection_metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Recent-3 baseline
    # --------------------------------------------------------

    recent3_selection_prediction = (
        selection[
            "recent_3_order_reorder_rate"
        ]
        .fillna(0)
        .to_numpy()
    )

    recent3_selection_metrics = (
        calculate_metrics(
            y_selection.to_numpy(),
            recent3_selection_prediction,
        )
    )

    print(
        "\nRecent-3 baseline:"
    )

    print(
        f"MAE  : "
        f"{recent3_selection_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : "
        f"{recent3_selection_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : "
        f"{recent3_selection_metrics['R2']:.6f}"
    )

    # ========================================================
    # 9. Create VALID sample weights
    # ========================================================

    section(
        "9. Creating leakage-safe sample weights"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # V4 used:
    #
    #     next_basket_size
    #
    # which is future information.
    #
    # V4.1 DOES NOT use next_basket_size.
    #
    # Every weighting strategy below uses only information
    # available at prediction time.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # A. No weighting
    # --------------------------------------------------------

    weight_none = None

    # --------------------------------------------------------
    # B. Historical average basket
    # --------------------------------------------------------

    weight_historical_basket = normalize_weights(

        np.sqrt(
            np.maximum(
                train[
                    "historical_avg_basket_size"
                ]
                .fillna(1)
                .to_numpy(),
                1,
            )
        )
    )

    # --------------------------------------------------------
    # C. Recent-3 basket
    # --------------------------------------------------------

    weight_recent_basket = normalize_weights(

        np.sqrt(
            np.maximum(
                train[
                    "recent_3_order_basket"
                ]
                .fillna(1)
                .to_numpy(),
                1,
            )
        )
    )

    # --------------------------------------------------------
    # D. Previous basket
    # --------------------------------------------------------

    weight_previous_basket = normalize_weights(

        np.sqrt(
            np.maximum(
                train[
                    "previous_basket_size"
                ]
                .fillna(1)
                .to_numpy(),
                1,
            )
        )
    )

    # --------------------------------------------------------
    # E. Customer-history confidence
    #
    # More historical observations = more reliable estimate.
    #
    # We use sqrt(previous_orders) so mature customers receive
    # somewhat more weight without dominating the model.
    # --------------------------------------------------------

    weight_history_confidence = normalize_weights(

        np.sqrt(
            np.maximum(
                train[
                    "previous_orders"
                ]
                .fillna(1)
                .to_numpy(),
                1,
            )
        )
    )

    # --------------------------------------------------------
    # F. Combined historical basket + confidence
    # --------------------------------------------------------

    combined_raw = (

        np.sqrt(
            np.maximum(
                train[
                    "historical_avg_basket_size"
                ]
                .fillna(1)
                .to_numpy(),
                1,
            )
        )

        *

        np.sqrt(
            np.maximum(
                train[
                    "previous_orders"
                ]
                .fillna(1)
                .to_numpy(),
                1,
            )
        )
    )

    weight_combined = normalize_weights(
        combined_raw
    )

    print(
        "\nWeight strategies:"
    )

    print(
        "  A. None"
    )

    print(
        "  B. Historical average basket"
    )

    print(
        "  C. Recent-3 basket"
    )

    print(
        "  D. Previous basket"
    )

    print(
        "  E. Customer-history confidence"
    )

    print(
        "  F. Historical basket + history confidence"
    )

    # ========================================================
    # 10. Candidate configurations
    # ========================================================

    section(
        "10. Training V4.1 candidate models"
    )

    candidates = {

        "V4.1 Standard":

        {
            "objective":
                "reg:absoluteerror",

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "weight":
                weight_none,
        },


        "V4.1 HistoricalBasketWeighted":

        {
            "objective":
                "reg:absoluteerror",

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "weight":
                weight_historical_basket,
        },


        "V4.1 RecentBasketWeighted":

        {
            "objective":
                "reg:absoluteerror",

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "weight":
                weight_recent_basket,
        },


        "V4.1 PreviousBasketWeighted":

        {
            "objective":
                "reg:absoluteerror",

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "weight":
                weight_previous_basket,
        },


        "V4.1 HistoryConfidenceWeighted":

        {
            "objective":
                "reg:absoluteerror",

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "weight":
                weight_history_confidence,
        },


        "V4.1 CombinedWeighted":

        {
            "objective":
                "reg:absoluteerror",

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "weight":
                weight_combined,
        },


        "V4.1 SquaredError":

        {
            "objective":
                "reg:squarederror",

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "weight":
                weight_none,
        },
    }

    trained_models = {}

    selection_predictions = {}

    selection_results = []

    # ========================================================
    # 11. Train models
    # ========================================================

    for name, config in candidates.items():

        print(
            f"\nTraining: {name}"
        )

        model = make_xgb(

            objective=config[
                "objective"
            ],

            max_depth=config[
                "max_depth"
            ],

            n_estimators=config[
                "n_estimators"
            ],

            learning_rate=config[
                "learning_rate"
            ],

            min_child_weight=config[
                "min_child_weight"
            ],
        )

        model.fit(

            X_train,

            y_train,

            sample_weight=config[
                "weight"
            ],

            verbose=False,
        )

        prediction = model.predict(
            X_selection
        )

        prediction = np.clip(
            prediction,
            0,
            1,
        )

        metrics = calculate_metrics(

            y_selection.to_numpy(),

            prediction,
        )

        trained_models[
            name
        ] = model

        selection_predictions[
            name
        ] = prediction

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

    # ========================================================
    # 12. Selection comparison
    # ========================================================

    section(
        "12. MODEL-SELECTION COMPARISON"
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

    print(
        selection_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # 13. Select best candidate
    # ========================================================

    section(
        "13. SELECTING V4.1 CANDIDATE"
    )

    best_name = (
        selection_df.iloc[
            0
        ]["model"]
    )

    best_model = (
        trained_models[
            best_name
        ]
    )

    best_selection_prediction = (
        selection_predictions[
            best_name
        ]
    )

    print(
        f"Selected candidate: "
        f"{best_name}"
    )

    # ========================================================
    # 14. Blend with Recent-3
    # ========================================================

    section(
        "14. SELECTING XGBOOST + RECENT-3 BLEND"
    )

    actual_selection = (
        y_selection.to_numpy()
    )

    blend_results = []

    for xgb_weight in np.arange(
        0.00,
        1.01,
        0.05,
    ):

        recent_weight = (
            1.0 -
            xgb_weight
        )

        blended_prediction = (

            xgb_weight
            *
            best_selection_prediction

            +

            recent_weight
            *
            recent3_selection_prediction
        )

        metrics = calculate_metrics(

            actual_selection,

            blended_prediction,
        )

        blend_results.append({

            "xgb_weight":
                round(
                    float(
                        xgb_weight
                    ),
                    2,
                ),

            "recent3_weight":
                round(
                    float(
                        recent_weight
                    ),
                    2,
                ),

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"],
        })

    blend_df = (
        pd.DataFrame(
            blend_results
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
        "\nTop blend candidates:"
    )

    print(
        blend_df.head(
            10
        ).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    best_blend = (
        blend_df.iloc[
            0
        ]
    )

    best_xgb_weight = float(
        best_blend[
            "xgb_weight"
        ]
    )

    best_recent_weight = float(
        best_blend[
            "recent3_weight"
        ]
    )

    print(
        f"\nSelected blend:"
    )

    print(
        f"XGBoost : "
        f"{best_xgb_weight:.0%}"
    )

    print(
        f"Recent-3: "
        f"{best_recent_weight:.0%}"
    )

    print(
        f"Selection MAE: "
        f"{best_blend['MAE']:.6f}"
    )

    # ========================================================
    # 15. Final untouched holdout
    # ========================================================

    section(
        "15. FINAL UNTOUCHED HOLDOUT"
    )

    actual_final = (
        y_final.to_numpy()
    )

    # --------------------------------------------------------
    # Historical baseline
    # --------------------------------------------------------

    historical_final_prediction = (
        final_holdout[
            "historical_reorder_rate"
        ]
        .fillna(0)
        .to_numpy()
    )

    historical_final_metrics = (
        calculate_metrics(
            actual_final,
            historical_final_prediction,
        )
    )

    print(
        "\nHistorical baseline:"
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

    # --------------------------------------------------------
    # Recent-3 baseline
    # --------------------------------------------------------

    recent3_final_prediction = (
        final_holdout[
            "recent_3_order_reorder_rate"
        ]
        .fillna(0)
        .to_numpy()
    )

    recent3_final_metrics = (
        calculate_metrics(
            actual_final,
            recent3_final_prediction,
        )
    )

    print(
        "\nRecent-3 baseline:"
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

    # --------------------------------------------------------
    # Selected V4.1 model
    # --------------------------------------------------------

    xgb_final_prediction = (
        best_model.predict(
            X_final
        )
    )

    xgb_final_prediction = np.clip(
        xgb_final_prediction,
        0,
        1,
    )

    xgb_final_metrics = (
        calculate_metrics(
            actual_final,
            xgb_final_prediction,
        )
    )

    print(
        f"\nSelected V4.1 model:"
    )

    print(
        f"{best_name}"
    )

    print(
        f"MAE  : "
        f"{xgb_final_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : "
        f"{xgb_final_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : "
        f"{xgb_final_metrics['R2']:.6f}"
    )

    # --------------------------------------------------------
    # Selected blend
    # --------------------------------------------------------

    final_blended_prediction = (

        best_xgb_weight
        *
        xgb_final_prediction

        +

        best_recent_weight
        *
        recent3_final_prediction
    )

    final_blended_prediction = np.clip(
        final_blended_prediction,
        0,
        1,
    )

    blended_final_metrics = (
        calculate_metrics(
            actual_final,
            final_blended_prediction,
        )
    )

    print(
        "\nSelected V4.1 + Recent-3 blend:"
    )

    print(
        f"MAE  : "
        f"{blended_final_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : "
        f"{blended_final_metrics['RMSE']:.6f}"
    )

    print(
        f"R2   : "
        f"{blended_final_metrics['R2']:.6f}"
    )

    # ========================================================
    # 16. Compare against V3 / V4 / V5
    # ========================================================

    section(
        "16. COMPARISON WITH PREVIOUS MODEL VERSIONS"
    )

    comparison = pd.DataFrame([

        {
            "model":
                "V3 Valid",

            "MAE":
                0.203614,

            "RMSE":
                0.263450,

            "R2":
                0.253438,

            "status":
                "Valid",
        },

        {
            "model":
                "V4 Original",

            "MAE":
                0.203357,

            "RMSE":
                0.261591,

            "R2":
                0.263938,

            "status":
                "LEAKAGE — not usable",
        },

        {
            "model":
                "V5 Valid",

            "MAE":
                0.203665,

            "RMSE":
                0.263543,

            "R2":
                0.252912,

            "status":
                "Valid",
        },

        {
            "model":
                "V4.1 XGBoost",

            "MAE":
                xgb_final_metrics[
                    "MAE"
                ],

            "RMSE":
                xgb_final_metrics[
                    "RMSE"
                ],

            "R2":
                xgb_final_metrics[
                    "R2"
                ],

            "status":
                "Valid",
        },

        {
            "model":
                "V4.1 + Recent-3",

            "MAE":
                blended_final_metrics[
                    "MAE"
                ],

            "RMSE":
                blended_final_metrics[
                    "RMSE"
                ],

            "R2":
                blended_final_metrics[
                    "R2"
                ],

            "status":
                "Valid",
        },
    ])

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # 17. Improvement vs V3
    # ========================================================

    section(
        "17. V4.1 IMPROVEMENT VS VALID V3"
    )

    v3_mae = 0.203614
    v3_rmse = 0.263450
    v3_r2 = 0.253438

    mae_improvement = (
        (
            v3_mae
            -
            blended_final_metrics[
                "MAE"
            ]
        )
        /
        v3_mae
        *
        100
    )

    rmse_improvement = (
        (
            v3_rmse
            -
            blended_final_metrics[
                "RMSE"
            ]
        )
        /
        v3_rmse
        *
        100
    )

    r2_change = (
        blended_final_metrics[
            "R2"
        ]
        -
        v3_r2
    )

    print(
        f"V3 MAE:"
        f" {v3_mae:.6f}"
    )

    print(
        f"V4.1 MAE:"
        f" {blended_final_metrics['MAE']:.6f}"
    )

    print(
        f"\nMAE improvement:"
        f" {mae_improvement:+.3f}%"
    )

    print(
        f"\nV3 RMSE:"
        f" {v3_rmse:.6f}"
    )

    print(
        f"V4.1 RMSE:"
        f" {blended_final_metrics['RMSE']:.6f}"
    )

    print(
        f"\nRMSE improvement:"
        f" {rmse_improvement:+.3f}%"
    )

    print(
        f"\nV3 R2:"
        f" {v3_r2:.6f}"
    )

    print(
        f"V4.1 R2:"
        f" {blended_final_metrics['R2']:.6f}"
    )

    print(
        f"\nR2 change:"
        f" {r2_change:+.6f}"
    )

    # ========================================================
    # 18. Prediction distribution
    # ========================================================

    section(
        "18. FINAL PREDICTION DISTRIBUTION"
    )

    print(
        pd.Series(
            final_blended_prediction,
            name="predicted_reorder_rate",
        )
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 19. Feature importance
    # ========================================================

    section(
        "19. FINAL FEATURE IMPORTANCE"
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
    # 20. Save results
    # ========================================================

    section(
        "20. Saving results"
    )

    result = {

        "selected_model":
            best_name,

        "xgb_weight":
            best_xgb_weight,

        "recent3_weight":
            best_recent_weight,

        "final_xgb_mae":
            xgb_final_metrics[
                "MAE"
            ],

        "final_xgb_rmse":
            xgb_final_metrics[
                "RMSE"
            ],

        "final_xgb_r2":
            xgb_final_metrics[
                "R2"
            ],

        "final_blend_mae":
            blended_final_metrics[
                "MAE"
            ],

        "final_blend_rmse":
            blended_final_metrics[
                "RMSE"
            ],

        "final_blend_r2":
            blended_final_metrics[
                "R2"
            ],

        "training_rows":
            len(train),

        "selection_rows":
            len(selection),

        "final_holdout_rows":
            len(final_holdout),

        "feature_count":
            len(feature_columns),

        "mae_improvement_vs_v3_pct":
            mae_improvement,

        "rmse_improvement_vs_v3_pct":
            rmse_improvement,

        "r2_change_vs_v3":
            r2_change,
    }

    pd.DataFrame(
        [result]
    ).to_csv(
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
    # 21. Final decision
    # ========================================================

    section(
        "FINAL V4.1 DECISION"
    )

    print(
        f"Selected model:"
        f" {best_name}"
    )

    print(
        f"Blend:"
        f" {best_xgb_weight:.0%} XGBoost + "
        f"{best_recent_weight:.0%} Recent-3"
    )

    print(
        f"\nFinal MAE:"
        f" {blended_final_metrics['MAE']:.6f}"
    )

    print(
        f"Final RMSE:"
        f" {blended_final_metrics['RMSE']:.6f}"
    )

    print(
        f"Final R2:"
        f" {blended_final_metrics['R2']:.6f}"
    )

    print(
        "\nLeakage check: PASSED"
    )

    print(
        "No next_basket_size was used "
        "for features or sample weights."
    )

    print(
        "\nFinal holdout was not used "
        "for candidate selection."
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":

    main()