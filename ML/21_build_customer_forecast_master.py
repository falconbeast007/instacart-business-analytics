from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBClassifier, XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

INPUT_PATH = (
    PROCESSED_DIR /
    "forecasting_dataset.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR /
    "customer_forecast_master.csv"
)

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def numeric_clean(df, columns):

    for column in columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[columns] = (
        df[columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return df


def regression_metrics(actual, predicted):

    predicted = np.maximum(
        predicted,
        0,
    )

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted,
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),

        "R2": r2_score(
            actual,
            predicted,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "CUSTOMER FORECAST MASTER — PRODUCTION PIPELINE"
    )

    # ========================================================
    # 1. LOAD DATA
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

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

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
    # 2. LATEST PREDICTION POINT
    # ========================================================

    section(
        "2. Identifying latest customer prediction point"
    )

    df["is_latest_prediction"] = (
        df.groupby(
            "user_id"
        )[
            "prediction_order_number"
        ]
        .transform("max")
        ==
        df[
            "prediction_order_number"
        ]
    )

    latest = df[
        df[
            "is_latest_prediction"
        ]
    ].copy()

    print(
        f"Latest customer prediction points: "
        f"{len(latest):,}"
    )

    print(
        f"Latest customers: "
        f"{latest['user_id'].nunique():,}"
    )

    # ========================================================
    # 3. MODEL 1 — FUTURE ORDER LIKELIHOOD
    # ========================================================

    section(
        "3. MODEL 1 — FUTURE ORDER LIKELIHOOD"
    )

    model1 = df.copy()

    # --------------------------------------------------------
    # V2 behavioral features
    #
    # IMPORTANT:
    # We do NOT filter customers here.
    #
    # Customers with <3 prediction points are excluded only
    # from temporal model selection, not from production
    # forecasting.
    # --------------------------------------------------------

    model1[
        "recent_5_order_interval"
    ] = (
        model1.groupby(
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

    model1[
        "recent_5_order_basket"
    ] = (
        model1.groupby(
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

    model1[
        "recent_5_order_reorder_rate"
    ] = (
        model1.groupby(
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

    model1[
        "interval_deviation"
    ] = (
        model1[
            "current_order_days_since_prior"
        ]
        -
        model1[
            "historical_avg_days_between_orders"
        ]
    )

    denominator = (
        model1[
            "historical_avg_days_between_orders"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    model1[
        "interval_ratio"
    ] = (
        model1[
            "current_order_days_since_prior"
        ]
        /
        denominator
    )

    model1[
        "recent_interval_deviation"
    ] = (
        model1[
            "recent_3_order_interval"
        ]
        -
        model1[
            "historical_avg_days_between_orders"
        ]
    )

    model1[
        "reorder_rate_trend"
    ] = (
        model1[
            "recent_3_order_reorder_rate"
        ]
        -
        model1[
            "historical_reorder_rate"
        ]
    )

    model1[
        "basket_size_trend"
    ] = (
        model1[
            "recent_3_order_basket"
        ]
        -
        model1[
            "historical_avg_basket_size"
        ]
    )

    model1[
        "interval_std_5"
    ] = (
        model1.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=2,
            ).std()
        )
    )

    model1[
        "interval_std_3"
    ] = (
        model1.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=3,
                min_periods=2,
            ).std()
        )
    )

    model1[
        "reorder_rate_std_5"
    ] = (
        model1.groupby(
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

    model1_features = [

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

        "recent_5_order_interval",

        "recent_5_order_basket",

        "recent_5_order_reorder_rate",

        "interval_deviation",

        "interval_ratio",

        "recent_interval_deviation",

        "reorder_rate_trend",

        "basket_size_trend",

        "interval_std_5",

        "interval_std_3",

        "reorder_rate_std_5",
    ]

    model1 = numeric_clean(
        model1,
        model1_features,
    )

    # --------------------------------------------------------
    # Temporal rank
    # --------------------------------------------------------

    model1["reverse_rank"] = (
        model1.groupby(
            "user_id"
        )[
            "prediction_order_number"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
        - 1
    )

    # --------------------------------------------------------
    # Production training data
    #
    # ALL non-latest historical prediction points.
    # --------------------------------------------------------

    train1 = model1[
        ~model1[
            "is_latest_prediction"
        ]
    ].copy()

    latest1 = model1[
        model1[
            "is_latest_prediction"
        ]
    ].copy()

    print(
        f"Model 1 training rows: "
        f"{len(train1):,}"
    )

    print(
        f"Model 1 forecast rows: "
        f"{len(latest1):,}"
    )

    # --------------------------------------------------------
    # Model-selection data
    #
    # Only customers with >=3 prediction points can have:
    #
    #   reverse_rank >= 2 -> training
    #   reverse_rank == 1 -> selection
    #   reverse_rank == 0 -> final
    #
    # This restriction is ONLY for model selection.
    # --------------------------------------------------------

    selection1 = model1[
        model1[
            "reverse_rank"
        ] == 1
    ].copy()

    train1_selection = model1[
        model1[
            "reverse_rank"
        ] >= 2
    ].copy()

    print(
        f"Model 1 selection rows: "
        f"{len(selection1):,}"
    )

    # --------------------------------------------------------
    # Training matrices
    # --------------------------------------------------------

    X1_train = train1[
        model1_features
    ]

    y1_train = (
        train1[
            "next_order_within_horizon"
        ]
        .astype(int)
    )

    X1_selection_train = (
        train1_selection[
            model1_features
        ]
    )

    y1_selection_train = (
        train1_selection[
            "next_order_within_horizon"
        ]
        .astype(int)
    )

    X1_selection = (
        selection1[
            model1_features
        ]
    )

    y1_selection = (
        selection1[
            "next_order_within_horizon"
        ]
        .astype(int)
    )

    X1_latest = (
        latest1[
            model1_features
        ]
    )

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    positive = (
        y1_train.sum()
    )

    negative = (
        len(y1_train)
        -
        positive
    )

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    print(
        f"Model 1 positive rows: "
        f"{positive:,}"
    )

    print(
        f"Model 1 negative rows: "
        f"{negative:,}"
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.6f}"
    )

    # ========================================================
    # MODEL 1 CANDIDATES
    # ========================================================

    model1_candidates = {

        "XGBoost V1 Configuration": {

            "max_depth": 6,

            "n_estimators": 500,

            "learning_rate": 0.05,

            "min_child_weight": 5,
        },

        "XGBoost V2 Deep": {

            "max_depth": 8,

            "n_estimators": 500,

            "learning_rate": 0.05,

            "min_child_weight": 5,
        },

        "XGBoost V2 MoreTrees": {

            "max_depth": 6,

            "n_estimators": 700,

            "learning_rate": 0.04,

            "min_child_weight": 5,
        },

        "XGBoost V2 Regularized": {

            "max_depth": 6,

            "n_estimators": 600,

            "learning_rate": 0.04,

            "min_child_weight": 10,
        },
    }

    # --------------------------------------------------------
    # Selection class weight
    # --------------------------------------------------------

    positive_selection = (
        y1_selection_train.sum()
    )

    negative_selection = (
        len(
            y1_selection_train
        )
        -
        positive_selection
    )

    selection_weight = (
        negative_selection
        /
        positive_selection
        if positive_selection > 0
        else 1.0
    )

    selected_model1_name = None

    selected_model1_params = None

    selected_pr_auc = -np.inf

    section(
        "MODEL 1 CANDIDATE SELECTION"
    )

    for name, params in (
        model1_candidates.items()
    ):

        print(
            f"\nTraining: {name}"
        )

        candidate = XGBClassifier(

            n_estimators=params[
                "n_estimators"
            ],

            max_depth=params[
                "max_depth"
            ],

            learning_rate=params[
                "learning_rate"
            ],

            min_child_weight=params[
                "min_child_weight"
            ],

            subsample=0.8,

            colsample_bytree=0.8,

            objective="binary:logistic",

            eval_metric="aucpr",

            tree_method="hist",

            n_jobs=-1,

            random_state=RANDOM_STATE,

            scale_pos_weight=selection_weight,
        )

        candidate.fit(
            X1_selection_train,
            y1_selection_train,
            verbose=False,
        )

        probabilities = (
            candidate.predict_proba(
                X1_selection
            )[:, 1]
        )

        pr_auc = (
            average_precision_score(
                y1_selection,
                probabilities,
            )
        )

        print(
            f"PR-AUC: "
            f"{pr_auc:.6f}"
        )

        if (
            pr_auc
            >
            selected_pr_auc
        ):

            selected_pr_auc = pr_auc

            selected_model1_name = name

            selected_model1_params = params

    print(
        f"\nSelected Model 1: "
        f"{selected_model1_name}"
    )

    # ========================================================
    # TRAIN FINAL MODEL 1
    # ========================================================

    section(
        "MODEL 1 FINAL PRODUCTION TRAINING"
    )

    final_model1 = XGBClassifier(

        n_estimators=selected_model1_params[
            "n_estimators"
        ],

        max_depth=selected_model1_params[
            "max_depth"
        ],

        learning_rate=selected_model1_params[
            "learning_rate"
        ],

        min_child_weight=selected_model1_params[
            "min_child_weight"
        ],

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        scale_pos_weight=scale_pos_weight,
    )

    final_model1.fit(
        X1_train,
        y1_train,
        verbose=False,
    )

    model1_probability = (
        final_model1.predict_proba(
            X1_latest
        )[:, 1]
    )

    model1_output = latest1[
        [
            "user_id",
            "prediction_order_number",
        ]
    ].copy()

    model1_output[
        "future_order_likelihood"
    ] = np.clip(
        model1_probability,
        0,
        1,
    )

    print(
        "\nModel 1 prediction distribution:"
    )

    print(
        pd.Series(
            model1_probability
        )
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 4. MODEL 2 — FUTURE REORDER RATE
    # ========================================================

    section(
        "4. MODEL 2 — FUTURE REORDER RATE"
    )

    model2 = df.copy()

    # --------------------------------------------------------
    # Model 2 V3 behavioral features
    # --------------------------------------------------------

    model2[
        "recent_5_order_reorder_rate"
    ] = (
        model2.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).mean()
        )
    )

    model2[
        "recent_5_order_basket"
    ] = (
        model2.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).mean()
        )
    )

    model2[
        "recent_5_order_interval"
    ] = (
        model2.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).mean()
        )
    )

    model2[
        "reorder_rate_trend_3"
    ] = (
        model2[
            "recent_3_order_reorder_rate"
        ]
        -
        model2[
            "historical_reorder_rate"
        ]
    )

    model2[
        "reorder_rate_trend_previous"
    ] = (
        model2[
            "previous_order_reorder_rate"
        ]
        -
        model2[
            "historical_reorder_rate"
        ]
    )

    model2[
        "basket_size_trend_3"
    ] = (
        model2[
            "recent_3_order_basket"
        ]
        -
        model2[
            "historical_avg_basket_size"
        ]
    )

    model2[
        "basket_size_trend_previous"
    ] = (
        model2[
            "previous_basket_size"
        ]
        -
        model2[
            "historical_avg_basket_size"
        ]
    )

    model2[
        "reorder_rate_std_5"
    ] = (
        model2.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=2,
            ).std()
        )
    )

    model2[
        "reorder_rate_std_3"
    ] = (
        model2.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=2,
            ).std()
        )
    )

    model2[
        "interval_trend"
    ] = (
        model2[
            "recent_3_order_interval"
        ]
        -
        model2[
            "historical_avg_days_between_orders"
        ]
    )

    # --------------------------------------------------------
    # V3 cadence / maturity features
    # --------------------------------------------------------

    model2[
        "order_frequency_rate"
    ] = (
        model2[
            "previous_orders"
        ]
        /
        model2[
            "historical_avg_days_between_orders"
        ].replace(
            0,
            np.nan,
        )
    )

    model2[
        "recent_3_order_frequency"
    ] = (
        1
        /
        model2[
            "recent_3_order_interval"
        ].replace(
            0,
            np.nan,
        )
    )

    model2[
        "recent_5_order_frequency"
    ] = (
        1
        /
        model2[
            "recent_5_order_interval"
        ].replace(
            0,
            np.nan,
        )
    )

    model2[
        "interval_deviation"
    ] = (
        model2[
            "current_order_days_since_prior"
        ]
        -
        model2[
            "historical_avg_days_between_orders"
        ]
    )

    model2[
        "interval_ratio"
    ] = (
        model2[
            "current_order_days_since_prior"
        ]
        /
        model2[
            "historical_avg_days_between_orders"
        ].replace(
            0,
            np.nan,
        )
    )

    model2[
        "recent_interval_deviation"
    ] = (
        model2[
            "recent_3_order_interval"
        ]
        -
        model2[
            "historical_avg_days_between_orders"
        ]
    )

    model2[
        "interval_trend_v3"
    ] = (
        model2[
            "recent_3_order_interval"
        ]
        -
        model2[
            "recent_5_order_interval"
        ]
    )

    model2[
        "recent_vs_historical_frequency"
    ] = (
        model2[
            "recent_3_order_frequency"
        ]
        -
        model2[
            "order_frequency_rate"
        ]
    )

    model2[
        "ordering_consistency"
    ] = (
        1
        /
        (
            model2[
                "reorder_rate_std_3"
            ]
            + 1e-6
        )
    )

    model2[
        "customer_maturity"
    ] = np.log1p(
        model2[
            "previous_orders"
        ]
    )

    model2[
        "recent_activity_maturity"
    ] = (
        model2[
            "previous_orders"
        ]
        /
        (
            model2[
                "current_order_days_since_prior"
            ]
            + 1
        )
    )

    model2_features = [

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

        "recent_activity_maturity",
    ]

    model2 = numeric_clean(
        model2,
        model2_features,
    )

    train2 = model2[
        ~model2[
            "is_latest_prediction"
        ]
    ].copy()

    latest2 = model2[
        model2[
            "is_latest_prediction"
        ]
    ].copy()

    X2_train = train2[
        model2_features
    ]

    y2_train = (
        train2[
            "next_reorder_rate"
        ]
        .astype(float)
    )

    X2_latest = latest2[
        model2_features
    ]

    print(
        f"Model 2 training rows: "
        f"{len(train2):,}"
    )

    print(
        f"Model 2 forecast rows: "
        f"{len(latest2):,}"
    )

    # --------------------------------------------------------
    # Current accepted Model 2 V3 production configuration
    #
    # XGB V3 Regularized
    # 92% XGB + 8% Recent-3 blend
    # --------------------------------------------------------

    model2_xgb = XGBRegressor(

        n_estimators=600,

        max_depth=6,

        learning_rate=0.04,

        min_child_weight=10,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        eval_metric="mae",

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )

    model2_xgb.fit(
        X2_train,
        y2_train,
        verbose=False,
    )

    model2_xgb_prediction = (
        model2_xgb.predict(
            X2_latest
        )
    )

    model2_recent3 = (
        latest2[
            "recent_3_order_reorder_rate"
        ]
        .to_numpy()
    )

    model2_prediction = (
        0.92
        * model2_xgb_prediction
        +
        0.08
        * model2_recent3
    )

    model2_prediction = np.clip(
        model2_prediction,
        0,
        1,
    )

    model2_output = latest2[
        [
            "user_id",
            "prediction_order_number",
        ]
    ].copy()

    model2_output[
        "expected_reorder_rate"
    ] = model2_prediction

    print(
        "\nModel 2 prediction distribution:"
    )

    print(
        pd.Series(
            model2_prediction
        )
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 5. MODEL 3 — FUTURE BASKET SIZE
    # ========================================================

    section(
        "5. MODEL 3 — FUTURE BASKET SIZE"
    )

    model3 = df.copy()

    # --------------------------------------------------------
    # V1 behavioral features
    # --------------------------------------------------------

    model3[
        "recent_5_order_reorder_rate"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).mean()
        )
    )

    model3[
        "recent_5_order_basket"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).mean()
        )
    )

    model3[
        "recent_5_order_interval"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).mean()
        )
    )

    model3[
        "interval_deviation"
    ] = (
        model3[
            "current_order_days_since_prior"
        ]
        -
        model3[
            "historical_avg_days_between_orders"
        ]
    )

    model3[
        "interval_ratio"
    ] = (
        model3[
            "current_order_days_since_prior"
        ]
        /
        model3[
            "historical_avg_days_between_orders"
        ].replace(
            0,
            np.nan,
        )
    )

    model3[
        "recent_interval_deviation"
    ] = (
        model3[
            "recent_3_order_interval"
        ]
        -
        model3[
            "historical_avg_days_between_orders"
        ]
    )

    model3[
        "reorder_rate_trend"
    ] = (
        model3[
            "recent_3_order_reorder_rate"
        ]
        -
        model3[
            "historical_reorder_rate"
        ]
    )

    model3[
        "basket_size_trend"
    ] = (
        model3[
            "recent_3_order_basket"
        ]
        -
        model3[
            "historical_avg_basket_size"
        ]
    )

    model3[
        "interval_std_5"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=2,
            ).std()
        )
    )

    model3[
        "interval_std_3"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "current_order_days_since_prior"
        ]
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=2,
            ).std()
        )
    )

    model3[
        "reorder_rate_std_5"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "previous_order_reorder_rate"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=2,
            ).std()
        )
    )

    v1_features = [

        "recent_5_order_reorder_rate",

        "recent_5_order_basket",

        "recent_5_order_interval",

        "interval_deviation",

        "interval_ratio",

        "recent_interval_deviation",

        "reorder_rate_trend",

        "basket_size_trend",

        "interval_std_5",

        "interval_std_3",

        "reorder_rate_std_5",
    ]

    # --------------------------------------------------------
    # V2 basket-dynamics features
    # --------------------------------------------------------

    model3[
        "recent_2_order_basket"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                2,
                min_periods=1,
            ).mean()
        )
    )

    model3[
        "recent_10_order_basket"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                10,
                min_periods=1,
            ).mean()
        )
    )

    model3[
        "basket_size_std_3"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=2,
            ).std()
        )
    )

    model3[
        "basket_size_std_10"
    ] = (
        model3.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                10,
                min_periods=2,
            ).std()
        )
    )

    model3[
        "recent_3_basket_ratio"
    ] = (
        model3[
            "recent_3_order_basket"
        ]
        /
        model3[
            "historical_avg_basket_size"
        ].replace(
            0,
            np.nan,
        )
    )

    model3[
        "recent_5_basket_ratio"
    ] = (
        model3[
            "recent_5_order_basket"
        ]
        /
        model3[
            "historical_avg_basket_size"
        ].replace(
            0,
            np.nan,
        )
    )

    model3[
        "recent_10_basket_ratio"
    ] = (
        model3[
            "recent_10_order_basket"
        ]
        /
        model3[
            "historical_avg_basket_size"
        ].replace(
            0,
            np.nan,
        )
    )

    model3[
        "basket_recent_trend_2_vs_5"
    ] = (
        model3[
            "recent_2_order_basket"
        ]
        -
        model3[
            "recent_5_order_basket"
        ]
    )

    model3[
        "basket_recent_trend_3_vs_10"
    ] = (
        model3[
            "recent_3_order_basket"
        ]
        -
        model3[
            "recent_10_order_basket"
        ]
    )

    model3[
        "previous_basket_vs_recent3"
    ] = (
        model3[
            "previous_basket_size"
        ]
        -
        model3[
            "recent_3_order_basket"
        ]
    )

    model3[
        "basket_cv_3"
    ] = (
        model3[
            "basket_size_std_3"
        ]
        /
        model3[
            "recent_3_order_basket"
        ].replace(
            0,
            np.nan,
        )
    )

    recent5_max = (
        model3.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).max()
        )
    )

    recent5_min = (
        model3.groupby(
            "user_id"
        )[
            "previous_basket_size"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=1,
            ).min()
        )
    )

    model3[
        "recent_5_basket_range"
    ] = (
        recent5_max
        -
        recent5_min
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

    model3_features = [

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

    ] + v1_features + v2_basket_features

    model3 = numeric_clean(
        model3,
        model3_features,
    )

    # --------------------------------------------------------
    # Temporal rank
    # --------------------------------------------------------

    model3["reverse_rank"] = (
        model3.groupby(
            "user_id"
        )[
            "prediction_order_number"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
        - 1
    )

    train3 = model3[
        ~model3[
            "is_latest_prediction"
        ]
    ].copy()

    latest3 = model3[
        model3[
            "is_latest_prediction"
        ]
    ].copy()

    print(
        f"Model 3 training rows: "
        f"{len(train3):,}"
    )

    print(
        f"Model 3 forecast rows: "
        f"{len(latest3):,}"
    )

    X3_train = train3[
        model3_features
    ]

    y3_train = (
        train3[
            "next_basket_size"
        ]
        .astype(float)
    )

    X3_latest = latest3[
        model3_features
    ]

    # --------------------------------------------------------
    # Model 3 selection
    # --------------------------------------------------------

    selection3 = model3[
        model3[
            "reverse_rank"
        ] == 1
    ].copy()

    train3_selection = model3[
        model3[
            "reverse_rank"
        ] >= 2
    ].copy()

    X3_selection_train = (
        train3_selection[
            model3_features
        ]
    )

    y3_selection_train = (
        train3_selection[
            "next_basket_size"
        ]
        .astype(float)
    )

    X3_selection = (
        selection3[
            model3_features
        ]
    )

    y3_selection = (
        selection3[
            "next_basket_size"
        ]
        .astype(float)
    )

    model3_candidates = {

        "V2 Standard": {

            "max_depth": 8,

            "n_estimators": 500,

            "learning_rate": 0.05,

            "min_child_weight": 5,

            "objective": "reg:squarederror",
        },

        "V2 MAE Objective": {

            "max_depth": 8,

            "n_estimators": 500,

            "learning_rate": 0.05,

            "min_child_weight": 5,

            "objective": "reg:absoluteerror",
        },

        "V2 Deeper": {

            "max_depth": 10,

            "n_estimators": 500,

            "learning_rate": 0.05,

            "min_child_weight": 5,

            "objective": "reg:squarederror",
        },

        "V2 Regularized": {

            "max_depth": 8,

            "n_estimators": 600,

            "learning_rate": 0.04,

            "min_child_weight": 10,

            "objective": "reg:squarederror",
        },
    }

    selected_model3_name = None

    selected_model3_params = None

    selected_mae = np.inf

    section(
        "MODEL 3 CANDIDATE SELECTION"
    )

    for name, params in (
        model3_candidates.items()
    ):

        print(
            f"\nTraining: {name}"
        )

        candidate = XGBRegressor(

            n_estimators=params[
                "n_estimators"
            ],

            max_depth=params[
                "max_depth"
            ],

            learning_rate=params[
                "learning_rate"
            ],

            min_child_weight=params[
                "min_child_weight"
            ],

            objective=params[
                "objective"
            ],

            eval_metric="mae",

            subsample=0.8,

            colsample_bytree=0.8,

            tree_method="hist",

            n_jobs=-1,

            random_state=RANDOM_STATE,
        )

        candidate.fit(
            X3_selection_train,
            y3_selection_train,
            verbose=False,
        )

        prediction = (
            candidate.predict(
                X3_selection
            )
        )

        metrics = regression_metrics(
            y3_selection,
            prediction,
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

        if (
            metrics["MAE"]
            <
            selected_mae
        ):

            selected_mae = (
                metrics["MAE"]
            )

            selected_model3_name = name

            selected_model3_params = params

    print(
        f"\nSelected Model 3: "
        f"{selected_model3_name}"
    )

    # ========================================================
    # MODEL 3 FINAL PRODUCTION TRAINING
    # ========================================================

    section(
        "MODEL 3 FINAL PRODUCTION TRAINING"
    )

    final_model3 = XGBRegressor(

        n_estimators=selected_model3_params[
            "n_estimators"
        ],

        max_depth=selected_model3_params[
            "max_depth"
        ],

        learning_rate=selected_model3_params[
            "learning_rate"
        ],

        min_child_weight=selected_model3_params[
            "min_child_weight"
        ],

        objective=selected_model3_params[
            "objective"
        ],

        eval_metric="mae",

        subsample=0.8,

        colsample_bytree=0.8,

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )

    final_model3.fit(
        X3_train,
        y3_train,
        verbose=False,
    )

    model3_prediction = (
        final_model3.predict(
            X3_latest
        )
    )

    model3_prediction = np.maximum(
        model3_prediction,
        1,
    )

    model3_output = latest3[
        [
            "user_id",
            "prediction_order_number",
        ]
    ].copy()

    model3_output[
        "expected_basket_size"
    ] = model3_prediction

    print(
        "\nModel 3 prediction distribution:"
    )

    print(
        pd.Series(
            model3_prediction
        )
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 6. UNIFIED FORECAST
    # ========================================================

    section(
        "6. Building unified customer forecast"
    )

    master = (
        model1_output
        .merge(
            model2_output,
            on=[
                "user_id",
                "prediction_order_number",
            ],
            how="inner",
        )
        .merge(
            model3_output,
            on=[
                "user_id",
                "prediction_order_number",
            ],
            how="inner",
        )
    )

    print(
        f"Unified rows: "
        f"{len(master):,}"
    )

    print(
        f"Unified customers: "
        f"{master['user_id'].nunique():,}"
    )

    # ========================================================
    # 7. ADD CUSTOMER CONTEXT
    # ========================================================

    section(
        "7. Adding customer forecast context"
    )

    context_columns = [

        "user_id",

        "prediction_order_number",

        "current_basket_size",

        "historical_avg_basket_size",

        "previous_basket_size",

        "historical_reorder_rate",

        "recent_3_order_reorder_rate",

        "previous_order_reorder_rate",

        "previous_orders",

        "historical_avg_days_between_orders",

        "current_order_days_since_prior",

        "recent_3_order_basket",

        "recent_3_order_interval",
    ]

    context = latest[
        context_columns
    ].copy()

    master = master.merge(
        context,
        on=[
            "user_id",
            "prediction_order_number",
        ],
        how="left",
    )

    # ========================================================
    # 8. BUSINESS FORECAST SIGNALS
    # ========================================================

    section(
        "8. Creating business forecast signals"
    )

    # --------------------------------------------------------
    # Expected number of reordered items
    # --------------------------------------------------------

    master[
        "expected_reorder_items"
    ] = (
        master[
            "expected_basket_size"
        ]
        *
        master[
            "expected_reorder_rate"
        ]
    )

    # --------------------------------------------------------
    # Expected new/discovery items
    # --------------------------------------------------------

    master[
        "expected_new_items"
    ] = (
        master[
            "expected_basket_size"
        ]
        -
        master[
            "expected_reorder_items"
        ]
    )

    master[
        "expected_new_items"
    ] = np.maximum(
        master[
            "expected_new_items"
        ],
        0,
    )

    # --------------------------------------------------------
    # Expected reorder rate change
    # --------------------------------------------------------

    master[
        "expected_reorder_rate_change"
    ] = (
        master[
            "expected_reorder_rate"
        ]
        -
        master[
            "historical_reorder_rate"
        ]
    )

    # ========================================================
    # 9. FORECAST SEGMENT
    # ========================================================

    section(
        "9. Creating forecast segments"
    )

    def assign_segment(row):

        likelihood = (
            row[
                "future_order_likelihood"
            ]
        )

        reorder_rate = (
            row[
                "expected_reorder_rate"
            ]
        )

        if (
            likelihood >= 0.70
            and
            reorder_rate >= 0.60
        ):

            return "Loyal"

        if (
            likelihood >= 0.60
            and
            reorder_rate >= 0.50
        ):

            return "Potential Loyal"

        if (
            likelihood < 0.40
            or
            (
                likelihood < 0.50
                and
                reorder_rate < 0.40
            )
        ):

            return "At-Risk"

        if (
            likelihood >= 0.40
            and
            reorder_rate >= 0.40
        ):

            return "Regular"

        return "Occasional"

    master[
        "forecast_segment"
    ] = master.apply(
        assign_segment,
        axis=1,
    )

    # ========================================================
    # 10. FORECAST METADATA
    # ========================================================

    section(
        "10. Adding forecast metadata"
    )

    master[
        "forecast_horizon"
    ] = "Next Order"

    master[
        "forecast_target"
    ] = (
        "Future customer behavior"
    )

    # ========================================================
    # 11. CLEAN PREDICTIONS
    # ========================================================

    section(
        "11. Cleaning final output"
    )

    master[
        "future_order_likelihood"
    ] = np.clip(
        master[
            "future_order_likelihood"
        ],
        0,
        1,
    )

    master[
        "expected_reorder_rate"
    ] = np.clip(
        master[
            "expected_reorder_rate"
        ],
        0,
        1,
    )

    master[
        "expected_basket_size"
    ] = np.maximum(
        master[
            "expected_basket_size"
        ],
        1,
    )

    master[
        "expected_reorder_items"
    ] = np.maximum(
        master[
            "expected_reorder_items"
        ],
        0,
    )

    master[
        "expected_new_items"
    ] = np.maximum(
        master[
            "expected_new_items"
        ],
        0,
    )

    # ========================================================
    # 12. FINAL QA
    # ========================================================

    section(
        "12. FINAL QA"
    )

    duplicate_pairs = (
        master.duplicated(
            [
                "user_id",
                "prediction_order_number",
            ]
        )
        .sum()
    )

    duplicate_customers = (
        master[
            "user_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate customer/order pairs: "
        f"{duplicate_pairs}"
    )

    print(
        f"Duplicate customers: "
        f"{duplicate_customers}"
    )

    prediction_columns = [

        "future_order_likelihood",

        "expected_reorder_rate",

        "expected_basket_size",

        "expected_reorder_items",

        "expected_new_items",
    ]

    print(
        "\nMissing predictions:"
    )

    print(
        master[
            prediction_columns
        ]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nForecast segment distribution:"
    )

    print(
        master[
            "forecast_segment"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Explicit assertions
    # --------------------------------------------------------

    assert duplicate_pairs == 0

    assert duplicate_customers == 0

    assert (
        master[
            prediction_columns
        ]
        .isna()
        .sum()
        .sum()
        ==
        0
    )

    assert (
        master[
            "future_order_likelihood"
        ]
        .between(
            0,
            1,
        )
        .all()
    )

    assert (
        master[
            "expected_reorder_rate"
        ]
        .between(
            0,
            1,
        )
        .all()
    )

    assert (
        master[
            "expected_basket_size"
        ]
        >=
        1
    ).all()

    assert (
        master[
            "expected_reorder_items"
        ]
        >=
        0
    ).all()

    assert (
        master[
            "expected_new_items"
        ]
        >=
        0
    ).all()

    # ========================================================
    # 13. FINAL COLUMN ORDER
    # ========================================================

    section(
        "13. Organizing final dataset"
    )

    final_columns = [

        # Identifiers
        "user_id",

        "prediction_order_number",

        # Main forecasts
        "future_order_likelihood",

        "expected_reorder_rate",

        "expected_basket_size",

        # Derived forecast signals
        "expected_reorder_items",

        "expected_new_items",

        "expected_reorder_rate_change",

        # Customer behavior
        "previous_orders",

        "current_basket_size",

        "previous_basket_size",

        "historical_avg_basket_size",

        "historical_reorder_rate",

        "recent_3_order_reorder_rate",

        "previous_order_reorder_rate",

        "historical_avg_days_between_orders",

        "current_order_days_since_prior",

        "recent_3_order_basket",

        "recent_3_order_interval",

        # Business interpretation
        "forecast_segment",

        "forecast_horizon",

        "forecast_target",
    ]

    master = master[
        final_columns
    ].copy()

    master = (
        master
        .sort_values(
            "user_id"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # 14. SAVE
    # ========================================================

    section(
        "14. Saving customer forecast master"
    )

    master.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"\nRows: "
        f"{len(master):,}"
    )

    print(
        f"Customers: "
        f"{master['user_id'].nunique():,}"
    )

    # ========================================================
    # 15. FINAL SUMMARY
    # ========================================================

    section(
        "FINAL CUSTOMER FORECAST SUMMARY"
    )

    print(
        "Model 1 — Future Order Likelihood       ✓"
    )

    print(
        "Model 2 — Future Reorder Rate           ✓"
    )

    print(
        "Model 3 — Future Basket Size            ✓"
    )

    print(
        "Customer Forecast Master               ✓"
    )

    print(
        f"\nFinal customers: "
        f"{master['user_id'].nunique():,}"
    )

    print(
        f"Final rows: "
        f"{len(master):,}"
    )

    print(
        "\nPrediction columns:"
    )

    for column in prediction_columns:

        print(
            f"  ✓ {column}"
        )

    print(
        "\nReady for Tableau Dashboard 4."
    )


if __name__ == "__main__":

    main()