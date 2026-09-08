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
# MODEL 2 V5 — FUTURE REORDER RATE
# ============================================================

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT /
    "Data" /
    "processed"
)

INPUT_PATH = (
    PROCESSED_DIR /
    "forecasting_dataset.parquet"
)

RESULTS_PATH = (
    PROCESSED_DIR /
    "reorder_rate_forecast_v5_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "reorder_rate_forecast_v5_feature_importance.csv"
)

RANDOM_STATE = 42


# ============================================================
# Utility functions
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(
    actual,
    predicted,
):

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


def create_model(
    max_depth,
    n_estimators,
    learning_rate,
    min_child_weight,
    objective,
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


# ============================================================
# Main
# ============================================================

def main():

    section(
        "MODEL 2 V5 — FUTURE REORDER RATE"
    )

    # ========================================================
    # 1. Loading forecasting dataset
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
            f"does not exist."
        )

    # ========================================================
    # 2. Sorting prediction points
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
    # Temporal rank
    #
    # reverse_rank = 0
    #     latest prediction point
    #
    # reverse_rank = 1
    #     second latest prediction point
    #
    # reverse_rank >= 2
    #     historical training prediction points
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
        "\nTemporal split definition:"
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
    # 4. Leakage-safe V5 feature engineering
    # ========================================================

    section(
        "4. Creating leakage-safe historical behavioral features"
    )

    print(
        "\nThese features are recreated inside V5."
    )

    print(
        "No future target information is used."
    )

    # --------------------------------------------------------
    # Historical series
    #
    # SHIFT(1) means the current prediction point is excluded
    # from these rolling calculations.
    # --------------------------------------------------------

    historical_interval = (
        df.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .shift(1)
    )

    historical_basket = (
        df.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .shift(1)
    )

    historical_reorder = (
        df.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .shift(1)
    )

    # --------------------------------------------------------
    # Recent 5 reorder rate
    # --------------------------------------------------------

    df[
        "recent_5_order_reorder_rate"
    ] = (
        historical_reorder
        .groupby(
            df["user_id"]
        )
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent 5 basket size
    # --------------------------------------------------------

    df[
        "recent_5_order_basket"
    ] = (
        historical_basket
        .groupby(
            df["user_id"]
        )
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent 5 order interval
    # --------------------------------------------------------

    df[
        "recent_5_order_interval"
    ] = (
        historical_interval
        .groupby(
            df["user_id"]
        )
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Reorder-rate trends
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Basket-size trends
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reorder-rate volatility
    # --------------------------------------------------------

    df[
        "reorder_rate_std_5"
    ] = (
        historical_reorder
        .groupby(
            df["user_id"]
        )
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
        historical_reorder
        .groupby(
            df["user_id"]
        )
        .transform(
            lambda x:
            x.rolling(
                window=3,
                min_periods=2,
            ).std()
        )
    )

    # --------------------------------------------------------
    # Interval trend
    # --------------------------------------------------------

    df[
        "interval_trend"
    ] = (
        df[
            "recent_3_order_interval"
        ]
        -
        df[
            "recent_5_order_interval"
        ]
    )

    # --------------------------------------------------------
    # Basket-size volatility
    # --------------------------------------------------------

    df[
        "basket_size_std_5"
    ] = (
        historical_basket
        .groupby(
            df["user_id"]
        )
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=2,
            ).std()
        )
    )

    derived_features = [

        "recent_5_order_reorder_rate",

        "recent_5_order_basket",

        "recent_5_order_interval",

        "reorder_rate_trend_3",

        "reorder_rate_trend_previous",

        "basket_size_trend_3",

        "basket_size_trend_previous",

        "reorder_rate_std_5",

        "reorder_rate_std_3",

        "interval_trend",

        "basket_size_std_5",
    ]

    # --------------------------------------------------------
    # Clean derived features
    # --------------------------------------------------------

    for column in derived_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    print(
        f"\nCreated {len(derived_features)} "
        f"historical behavioral features:"
    )

    for feature in derived_features:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # 5. Recreate temporal splits AFTER feature engineering
    # ========================================================

    section(
        "5. Recreating temporal splits after feature engineering"
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
    # 6. V5 feature list
    # ========================================================

    section(
        "6. Preparing V5 feature set"
    )

    features = [

        # ----------------------------------------------------
        # Customer behavior
        # ----------------------------------------------------

        "previous_orders",

        "current_basket_size",

        "historical_avg_basket_size",

        "previous_basket_size",

        "recent_3_order_basket",

        # ----------------------------------------------------
        # Reorder behavior
        # ----------------------------------------------------

        "historical_reorder_rate",

        "previous_order_reorder_rate",

        "recent_3_order_reorder_rate",

        "cumulative_reorder_rate",

        # ----------------------------------------------------
        # Ordering cadence
        # ----------------------------------------------------

        "historical_avg_days_between_orders",

        "current_order_days_since_prior",

        "recent_3_order_interval",

        # ----------------------------------------------------
        # Historical volume
        # ----------------------------------------------------

        "previous_total_items",

        "previous_total_reorder_items",

        # ----------------------------------------------------
        # Order timing
        # ----------------------------------------------------

        "order_dow",

        "order_hour_of_day",

        # ----------------------------------------------------
        # V5 behavioral features
        # ----------------------------------------------------

        "recent_5_order_reorder_rate",

        "recent_5_order_basket",

        "recent_5_order_interval",

        "reorder_rate_trend_3",

        "reorder_rate_trend_previous",

        "basket_size_trend_3",

        "basket_size_trend_previous",

        "reorder_rate_std_5",

        "reorder_rate_std_3",

        "interval_trend",

        "basket_size_std_5",
    ]

    print(
        f"Total features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "The following V5 features "
            "are missing:\n"
            +
            "\n".join(
                missing_features
            )
        )

    print(
        "\nAll required V5 features exist."
    )

    # --------------------------------------------------------
    # Explicit leakage check
    # --------------------------------------------------------

    forbidden_features = [

        "next_reorder_rate",

        "next_basket_size",

        "next_order_within_horizon",

        "next_order_interval",
    ]

    leakage_features = [
        feature
        for feature in features
        if feature in forbidden_features
    ]

    if leakage_features:

        raise ValueError(
            "LEAKAGE DETECTED! "
            "Future target columns found "
            "in feature set:\n"
            +
            "\n".join(
                leakage_features
            )
        )

    print(
        "Leakage check passed."
    )

    # ========================================================
    # 7. Prepare matrices
    # ========================================================

    section(
        "7. Preparing model matrices"
    )

    X_train = train[
        features
    ].copy()

    X_selection = selection[
        features
    ].copy()

    X_final = final_holdout[
        features
    ].copy()

    y_train = (
        train[target]
        .astype(float)
    )

    y_selection = (
        selection[target]
        .astype(float)
    )

    y_final = (
        final_holdout[target]
        .astype(float)
    )

    for X in [

        X_train,

        X_selection,

        X_final,

    ]:

        for column in features:

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
    # Historical baseline
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
    # 9. Candidate models
    # ========================================================

    section(
        "9. Training V5 candidate models"
    )

    candidate_configs = {

        # ----------------------------------------------------
        # Standard
        # ----------------------------------------------------

        "V5 Standard": {

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "objective":
                "reg:squarederror",

            "weight_type":
                None,
        },

        # ----------------------------------------------------
        # Absolute-error objective
        # ----------------------------------------------------

        "V5 MAE Objective": {

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "objective":
                "reg:absoluteerror",

            "weight_type":
                None,
        },

        # ----------------------------------------------------
        # Deeper
        # ----------------------------------------------------

        "V5 Deep": {

            "max_depth":
                10,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "objective":
                "reg:squarederror",

            "weight_type":
                None,
        },

        # ----------------------------------------------------
        # Regularized
        # ----------------------------------------------------

        "V5 Regularized": {

            "max_depth":
                8,

            "n_estimators":
                600,

            "learning_rate":
                0.04,

            "min_child_weight":
                10,

            "objective":
                "reg:squarederror",

            "weight_type":
                None,
        },

        # ----------------------------------------------------
        # Historical basket weighted
        #
        # Uses only historical_avg_basket_size.
        # ----------------------------------------------------

        "V5 HistoricalBasketWeighted": {

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "objective":
                "reg:squarederror",

            "weight_type":
                "historical_basket",
        },

        # ----------------------------------------------------
        # Recent basket weighted
        #
        # Uses only recent_3_order_basket.
        # ----------------------------------------------------

        "V5 RecentBasketWeighted": {

            "max_depth":
                8,

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "min_child_weight":
                5,

            "objective":
                "reg:squarederror",

            "weight_type":
                "recent_basket",
        },
    }

    trained_models = {}

    selection_results = []

    # ========================================================
    # 10. Train candidates
    # ========================================================

    for name, config in candidate_configs.items():

        print()
        print(
            f"Training: {name}"
        )

        model = create_model(

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

            objective=config[
                "objective"
            ],
        )

        sample_weight = None

        # ----------------------------------------------------
        # Historical basket weighting
        # ----------------------------------------------------

        if config[
            "weight_type"
        ] == "historical_basket":

            raw_weight = (
                train[
                    "historical_avg_basket_size"
                ]
                .fillna(1)
                .to_numpy()
            )

            sample_weight = np.sqrt(
                np.maximum(
                    raw_weight,
                    1,
                )
            )

            sample_weight = np.clip(
                sample_weight,
                1,
                3,
            )

            sample_weight = (
                sample_weight
                /
                sample_weight.mean()
            )

            print(
                "Using historical basket "
                "sample weights."
            )

        # ----------------------------------------------------
        # Recent basket weighting
        # ----------------------------------------------------

        elif config[
            "weight_type"
        ] == "recent_basket":

            raw_weight = (
                train[
                    "recent_3_order_basket"
                ]
                .fillna(1)
                .to_numpy()
            )

            sample_weight = np.sqrt(
                np.maximum(
                    raw_weight,
                    1,
                )
            )

            sample_weight = np.clip(
                sample_weight,
                1,
                3,
            )

            sample_weight = (
                sample_weight
                /
                sample_weight.mean()
            )

            print(
                "Using recent basket "
                "sample weights."
            )

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            verbose=False,
        )

        selection_prediction = (
            model.predict(
                X_selection
            )
        )

        selection_prediction = np.clip(
            selection_prediction,
            0,
            1,
        )

        metrics = calculate_metrics(
            y_selection.to_numpy(),
            selection_prediction,
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

        trained_models[
            name
        ] = model

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
    # 11. Model-selection comparison
    # ========================================================

    section(
        "11. Model-selection comparison"
    )

    selection_df = pd.DataFrame(
        selection_results
    )

    baseline_df = pd.DataFrame([

        {
            "model":
                "Historical Baseline",

            "MAE":
                historical_selection_metrics[
                    "MAE"
                ],

            "RMSE":
                historical_selection_metrics[
                    "RMSE"
                ],

            "R2":
                historical_selection_metrics[
                    "R2"
                ],
        },

        {
            "model":
                "Recent-3 Baseline",

            "MAE":
                recent3_selection_metrics[
                    "MAE"
                ],

            "RMSE":
                recent3_selection_metrics[
                    "RMSE"
                ],

            "R2":
                recent3_selection_metrics[
                    "R2"
                ],
        },

    ])

    selection_comparison = pd.concat(
        [
            baseline_df,
            selection_df,
        ],
        ignore_index=True,
    )

    selection_comparison = (
        selection_comparison
        .sort_values(
            "MAE",
            ascending=True,
        )
        .reset_index(
            drop=True,
        )
    )

    print(
        selection_comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # 12. Select best XGBoost candidate
    # ========================================================

    section(
        "12. Selecting V5 XGBoost candidate"
    )

    xgb_selection = (
        selection_df
        .sort_values(
            "MAE",
            ascending=True,
        )
        .reset_index(
            drop=True,
        )
    )

    best_name = (
        xgb_selection.iloc[
            0
        ]["model"]
    )

    best_model = (
        trained_models[
            best_name
        ]
    )

    print(
        f"Selected V5 candidate: "
        f"{best_name}"
    )

    # ========================================================
    # 13. Select blend weight
    # ========================================================

    section(
        "13. Selecting XGBoost + Recent-3 blend weight"
    )

    xgb_selection_prediction = (
        best_model.predict(
            X_selection
        )
    )

    xgb_selection_prediction = np.clip(
        xgb_selection_prediction,
        0,
        1,
    )

    actual_selection = (
        y_selection.to_numpy()
    )

    blend_results = []

    # --------------------------------------------------------
    # Search weights in 5% increments.
    #
    # This uses ONLY the model-selection set.
    # --------------------------------------------------------

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
            xgb_selection_prediction

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
    # 14. Final untouched holdout
    # ========================================================

    section(
        "14. FINAL UNTOUCHED HOLDOUT"
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
    # Selected XGBoost
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
        f"\nSelected XGBoost: "
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
        "\nSelected XGBoost + Recent-3 blend:"
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
    # 15. Final comparison
    # ========================================================

    section(
        "15. FINAL MODEL COMPARISON"
    )

    final_comparison = pd.DataFrame([

        {
            "model":
                "Historical Baseline",

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
                ],
        },

        {
            "model":
                "Recent-3 Baseline",

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
                ],
        },

        {
            "model":
                best_name,

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
        },

        {
            "model":
                f"{best_name} + Recent-3",

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
        },

    ])

    print(
        final_comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # 16. Improvement vs Recent-3
    # ========================================================

    section(
        "16. Improvement vs Recent-3 baseline"
    )

    mae_improvement = (

        (
            recent3_final_metrics[
                "MAE"
            ]

            -

            blended_final_metrics[
                "MAE"
            ]
        )

        /

        recent3_final_metrics[
            "MAE"
        ]

        *

        100
    )

    rmse_improvement = (

        (
            recent3_final_metrics[
                "RMSE"
            ]

            -

            blended_final_metrics[
                "RMSE"
            ]
        )

        /

        recent3_final_metrics[
            "RMSE"
        ]

        *

        100
    )

    print(
        f"MAE improvement : "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_improvement:.2f}%"
    )

    # ========================================================
    # 17. Prediction distribution
    # ========================================================

    section(
        "17. Final prediction distribution"
    )

    prediction_series = pd.Series(
        final_blended_prediction,
        name="predicted_reorder_rate",
    )

    print(
        prediction_series
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 18. Feature importance
    # ========================================================

    section(
        "18. Final feature importance"
    )

    importance_df = pd.DataFrame({

        "feature":
            features,

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
    # 19. Save results
    # ========================================================

    section(
        "19. Saving results"
    )

    result = {

        "selected_xgb_model":
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

        "recent3_final_mae":
            recent3_final_metrics[
                "MAE"
            ],

        "recent3_final_rmse":
            recent3_final_metrics[
                "RMSE"
            ],

        "recent3_final_r2":
            recent3_final_metrics[
                "R2"
            ],

        "historical_final_mae":
            historical_final_metrics[
                "MAE"
            ],

        "historical_final_rmse":
            historical_final_metrics[
                "RMSE"
            ],

        "historical_final_r2":
            historical_final_metrics[
                "R2"
            ],

        "training_rows":
            len(train),

        "selection_rows":
            len(selection),

        "final_holdout_rows":
            len(final_holdout),

        "final_holdout_customers":
            final_holdout[
                "user_id"
            ].nunique(),

        "feature_count":
            len(features),

        "mae_improvement_vs_recent3_pct":
            mae_improvement,

        "rmse_improvement_vs_recent3_pct":
            rmse_improvement,
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
    # 20. Final summary
    # ========================================================

    section(
        "FINAL SUMMARY"
    )

    print(
        f"Selected XGBoost: "
        f"{best_name}"
    )

    print(
        f"Selected blend: "
        f"{best_xgb_weight:.0%} XGBoost + "
        f"{best_recent_weight:.0%} Recent-3"
    )

    print(
        f"\nFinal blended MAE: "
        f"{blended_final_metrics['MAE']:.6f}"
    )

    print(
        f"Final blended RMSE: "
        f"{blended_final_metrics['RMSE']:.6f}"
    )

    print(
        f"Final blended R2: "
        f"{blended_final_metrics['R2']:.6f}"
    )

    print(
        "\nNo future target information was used "
        "for features or sample weights."
    )

    print(
        "The final holdout remained untouched "
        "during model selection."
    )

    print(
        "\nDone."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()