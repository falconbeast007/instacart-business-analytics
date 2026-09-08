from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


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
    "order_likelihood_v3_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "order_likelihood_v3_feature_importance.csv"
)

RANDOM_STATE = 42


# ============================================================
# Helper
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(actual, probabilities):

    probabilities = np.clip(
        probabilities,
        0,
        1,
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {

        "PR_AUC":
            average_precision_score(
                actual,
                probabilities,
            ),

        "ROC_AUC":
            roc_auc_score(
                actual,
                probabilities,
            ),

        "Precision":
            precision_score(
                actual,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                actual,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                actual,
                predictions,
                zero_division=0,
            ),
    }


# ============================================================
# XGBoost model factory
# ============================================================

def create_model(
    max_depth,
    n_estimators,
    learning_rate,
    min_child_weight,
    scale_pos_weight,
):

    return XGBClassifier(

        n_estimators=n_estimators,

        max_depth=max_depth,

        learning_rate=learning_rate,

        min_child_weight=min_child_weight,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        scale_pos_weight=scale_pos_weight,
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "MODEL 1 V3 — FUTURE ORDER LIKELIHOOD"
    )

    # ========================================================
    # 1. Load forecasting dataset
    # ========================================================

    section(
        "1. Loading forecasting dataset"
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
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
        "next_order_within_horizon"
    )

    if target not in df.columns:

        raise ValueError(
            f"Target '{target}' "
            f"not found in forecasting dataset."
        )

    # ========================================================
    # 2. Sort prediction points
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
    # 3. Create three-way temporal split
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

    # --------------------------------------------------------
    # Reverse temporal rank
    #
    # 0 = latest prediction point
    # 1 = penultimate
    # 2+ = training history
    # --------------------------------------------------------

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
        f"\nTraining rows          : "
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
    # 4. V3 historical rolling features
    # ========================================================

    section(
        "4. Engineering V3 cadence + maturity features"
    )

    print(
        "\nCreating historical rolling features..."
    )

    # --------------------------------------------------------
    # IMPORTANT LEAKAGE CONTROL
    #
    # At each prediction point:
    #
    # Current customer history
    #          ↓
    #       features
    #          ↓
    # Predict NEXT order
    #
    # Therefore rolling features must not use the current
    # prediction row itself.
    #
    # We shift the historical series by one row first.
    # --------------------------------------------------------

    historical_interval = (
        df.groupby("user_id")[
            "current_order_days_since_prior"
        ]
        .shift(1)
    )

    historical_basket = (
        df.groupby("user_id")[
            "previous_basket_size"
        ]
        .shift(1)
    )

    historical_reorder = (
        df.groupby("user_id")[
            "previous_order_reorder_rate"
        ]
        .shift(1)
    )

    # --------------------------------------------------------
    # Recent 5 interval
    # --------------------------------------------------------

    df["recent_5_order_interval"] = (
        historical_interval
        .groupby(df["user_id"])
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=1,
                ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent 5 basket
    # --------------------------------------------------------

    df["recent_5_order_basket"] = (
        historical_basket
        .groupby(df["user_id"])
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=1,
                ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent 5 reorder rate
    # --------------------------------------------------------

    df["recent_5_order_reorder_rate"] = (
        historical_reorder
        .groupby(df["user_id"])
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=1,
                ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent 3 interval
    #
    # The original forecasting dataset already contains
    # recent_3_order_interval.
    #
    # We recreate a leakage-safe version here for V3.
    # --------------------------------------------------------

    df["recent_3_order_interval_v3"] = (
        historical_interval
        .groupby(df["user_id"])
        .transform(
            lambda x:
                x.rolling(
                    window=3,
                    min_periods=1,
                ).mean()
        )
    )

    # --------------------------------------------------------
    # Interval volatility
    # --------------------------------------------------------

    df["interval_std_5"] = (
        historical_interval
        .groupby(df["user_id"])
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=2,
                ).std()
        )
    )

    df["interval_std_3"] = (
        historical_interval
        .groupby(df["user_id"])
        .transform(
            lambda x:
                x.rolling(
                    window=3,
                    min_periods=2,
                ).std()
        )
    )

    # --------------------------------------------------------
    # Reorder volatility
    # --------------------------------------------------------

    df["reorder_rate_std_5"] = (
        historical_reorder
        .groupby(df["user_id"])
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=2,
                ).std()
        )
    )

    # ========================================================
    # 5. V3 derived cadence features
    # ========================================================

    print(
        "\nCreating cadence and maturity features..."
    )

    # --------------------------------------------------------
    # Overall historical ordering frequency
    #
    # Avoiding division by zero.
    # --------------------------------------------------------

    historical_interval_safe = (
        df["historical_avg_days_between_orders"]
        .replace(
            0,
            np.nan,
        )
    )

    df["order_frequency_rate"] = (
        1.0 /
        historical_interval_safe
    )

    # --------------------------------------------------------
    # Recent 3 frequency
    # --------------------------------------------------------

    recent_3_interval_safe = (
        df["recent_3_order_interval_v3"]
        .replace(
            0,
            np.nan,
        )
    )

    df["recent_3_order_frequency"] = (
        1.0 /
        recent_3_interval_safe
    )

    # --------------------------------------------------------
    # Recent 5 frequency
    # --------------------------------------------------------

    recent_5_interval_safe = (
        df["recent_5_order_interval"]
        .replace(
            0,
            np.nan,
        )
    )

    df["recent_5_order_frequency"] = (
        1.0 /
        recent_5_interval_safe
    )

    # --------------------------------------------------------
    # Current interval deviation
    #
    # Positive:
    # customer is taking longer than usual.
    #
    # Negative:
    # customer is ordering faster than usual.
    # --------------------------------------------------------

    df["interval_deviation"] = (
        df["current_order_days_since_prior"]
        -
        df["historical_avg_days_between_orders"]
    )

    # --------------------------------------------------------
    # Current interval ratio
    #
    # 1.0 = normal cadence
    # >1  = slower than usual
    # <1  = faster than usual
    # --------------------------------------------------------

    df["interval_ratio"] = (
        df["current_order_days_since_prior"]
        /
        historical_interval_safe
    )

    # --------------------------------------------------------
    # Recent cadence deviation
    # --------------------------------------------------------

    df["recent_interval_deviation"] = (
        df["recent_3_order_interval_v3"]
        -
        df["historical_avg_days_between_orders"]
    )

    # --------------------------------------------------------
    # Interval trend
    #
    # Negative:
    # recent 3-order interval is shorter than recent 5-order
    # interval → ordering may be speeding up.
    #
    # Positive:
    # recent 3-order interval is longer → ordering may be
    # slowing down.
    # --------------------------------------------------------

    df["interval_trend"] = (
        df["recent_3_order_interval_v3"]
        -
        df["recent_5_order_interval"]
    )

    # --------------------------------------------------------
    # Recent vs historical frequency
    # --------------------------------------------------------

    historical_frequency = (
        1.0 /
        historical_interval_safe
    )

    recent_frequency = (
        1.0 /
        recent_3_interval_safe
    )

    df["recent_vs_historical_frequency"] = (
        recent_frequency
        /
        historical_frequency.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Ordering consistency
    #
    # Lower interval variability → higher consistency.
    # --------------------------------------------------------

    df["ordering_consistency"] = (
        1.0 /
        (
            1.0
            +
            df["interval_std_5"]
        )
    )

    # --------------------------------------------------------
    # Customer maturity
    #
    # Log transform reduces the effect of very large
    # previous-order counts.
    # --------------------------------------------------------

    df["customer_maturity"] = (
        np.log1p(
            df["previous_orders"]
        )
    )

    # --------------------------------------------------------
    # Recent activity × maturity
    # --------------------------------------------------------

    df["recent_activity_maturity"] = (
        np.log1p(
            df["previous_orders"]
        )
        *
        recent_frequency
    )

    # ========================================================
    # 6. V3 feature list
    # ========================================================

    v3_features = [

        "order_frequency_rate",

        "recent_3_order_frequency",

        "recent_5_order_frequency",

        "interval_deviation",

        "interval_ratio",

        "recent_interval_deviation",

        "interval_trend",

        "recent_vs_historical_frequency",

        "ordering_consistency",

        "customer_maturity",

        "recent_activity_maturity",
    ]

    # ========================================================
    # 7. Clean derived features
    # ========================================================

    print(
        "\nCleaning V3 features..."
    )

    derived_features = [

        "recent_5_order_interval",

        "recent_5_order_basket",

        "recent_5_order_reorder_rate",

        "recent_3_order_interval_v3",

        "interval_std_5",

        "interval_std_3",

        "reorder_rate_std_5",

    ] + v3_features

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
        f"\nAdded V3 features: "
        f"{len(v3_features)}"
    )

    for feature in v3_features:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # 8. Recreate temporal splits
    # ========================================================

    train = df[
        df["reverse_rank"] >= 2
    ].copy()

    selection = df[
        df["reverse_rank"] == 1
    ].copy()

    final_holdout = df[
        df["reverse_rank"] == 0
    ].copy()

    # ========================================================
    # 9. Base feature set
    # ========================================================

    section(
        "5. Selecting model features"
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
        f"Base features : "
        f"{len(base_features)}"
    )

    print(
        f"V3 features   : "
        f"{len(v3_features)}"
    )

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # 10. Prepare matrices
    # ========================================================

    section(
        "6. Preparing model matrices"
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
        train[target]
        .astype(int)
    )

    y_selection = (
        selection[target]
        .astype(int)
    )

    y_final = (
        final_holdout[target]
        .astype(int)
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

    # ========================================================
    # 11. Class balance
    # ========================================================

    section(
        "7. Class balance"
    )

    positive_count = (
        y_train.sum()
    )

    negative_count = (
        len(y_train)
        -
        positive_count
    )

    scale_pos_weight = (
        negative_count /
        positive_count
    )

    print(
        f"Positive training rows: "
        f"{positive_count:,}"
    )

    print(
        f"Negative training rows: "
        f"{negative_count:,}"
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================
    # 12. Candidate configurations
    # ========================================================

    section(
        "8. Training V3 candidates"
    )

    candidate_configs = {

        # ----------------------------------------------------
        # Benchmark against V2
        # ----------------------------------------------------

        "V2 Deep Baseline": {

            "max_depth": 8,

            "n_estimators": 500,

            "learning_rate": 0.05,

            "min_child_weight": 5,
        },

        # ----------------------------------------------------
        # Deeper trees
        # ----------------------------------------------------

        "V3 Deeper": {

            "max_depth": 10,

            "n_estimators": 500,

            "learning_rate": 0.05,

            "min_child_weight": 5,
        },

        # ----------------------------------------------------
        # More trees
        # ----------------------------------------------------

        "V3 MoreTrees": {

            "max_depth": 8,

            "n_estimators": 700,

            "learning_rate": 0.04,

            "min_child_weight": 5,
        },

        # ----------------------------------------------------
        # More regularization
        # ----------------------------------------------------

        "V3 Regularized": {

            "max_depth": 8,

            "n_estimators": 600,

            "learning_rate": 0.04,

            "min_child_weight": 10,
        },

        # ----------------------------------------------------
        # Shallower model
        # ----------------------------------------------------

        "V3 Shallow": {

            "max_depth": 6,

            "n_estimators": 700,

            "learning_rate": 0.04,

            "min_child_weight": 5,
        },
    }

    trained_models = []

    selection_results = []

    # ========================================================
    # 13. Train candidates
    # ========================================================

    for name, config in candidate_configs.items():

        print(
            f"\nTraining: {name}"
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

            scale_pos_weight=scale_pos_weight,
        )

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        probabilities = (
            model.predict_proba(
                X_selection
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_selection.to_numpy(),
            probabilities,
        )

        selection_results.append({

            "model":
                name,

            "PR_AUC":
                metrics["PR_AUC"],

            "ROC_AUC":
                metrics["ROC_AUC"],

            "Precision":
                metrics["Precision"],

            "Recall":
                metrics["Recall"],

            "F1":
                metrics["F1"],
        })

        trained_models.append(
            (
                name,
                model,
            )
        )

        print(
            f"PR-AUC   : "
            f"{metrics['PR_AUC']:.6f}"
        )

        print(
            f"ROC-AUC  : "
            f"{metrics['ROC_AUC']:.6f}"
        )

        print(
            f"Precision: "
            f"{metrics['Precision']:.6f}"
        )

        print(
            f"Recall   : "
            f"{metrics['Recall']:.6f}"
        )

        print(
            f"F1       : "
            f"{metrics['F1']:.6f}"
        )

    selection_df = (
        pd.DataFrame(
            selection_results
        )
        .sort_values(
            "PR_AUC",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )

    # ========================================================
    # 14. Candidate selection
    # ========================================================

    section(
        "9. Selecting V3 candidate"
    )

    print(
        selection_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    best_name = (
        selection_df.iloc[0]["model"]
    )

    print(
        f"\nSelected candidate: "
        f"{best_name}"
    )

    # ========================================================
    # 15. Selected configuration
    # ========================================================

    selected_config = (
        candidate_configs[
            best_name
        ]
    )

    # ========================================================
    # 16. Retrain selected model
    # ========================================================

    section(
        "10. Retraining selected V3 model"
    )

    final_model = create_model(

        max_depth=selected_config[
            "max_depth"
        ],

        n_estimators=selected_config[
            "n_estimators"
        ],

        learning_rate=selected_config[
            "learning_rate"
        ],

        min_child_weight=selected_config[
            "min_child_weight"
        ],

        scale_pos_weight=scale_pos_weight,
    )

    final_model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    # ========================================================
    # 17. Final predictions
    # ========================================================

    final_probabilities = (
        final_model.predict_proba(
            X_final
        )[:, 1]
    )

    # ========================================================
    # 18. Final untouched holdout
    # ========================================================

    section(
        "11. FINAL UNTOUCHED HOLDOUT"
    )

    final_metrics = calculate_metrics(
        y_final.to_numpy(),
        final_probabilities,
    )

    print(
        f"Selected model: "
        f"{best_name}"
    )

    print(
        f"\nPR-AUC   : "
        f"{final_metrics['PR_AUC']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{final_metrics['ROC_AUC']:.6f}"
    )

    print(
        f"Precision: "
        f"{final_metrics['Precision']:.6f}"
    )

    print(
        f"Recall   : "
        f"{final_metrics['Recall']:.6f}"
    )

    print(
        f"F1       : "
        f"{final_metrics['F1']:.6f}"
    )

    # ========================================================
    # 19. Probability distribution
    # ========================================================

    section(
        "12. Prediction probability distribution"
    )

    probability_summary = (
        pd.Series(
            final_probabilities,
            name="order_likelihood",
        )
        .describe()
        .round(4)
    )

    print(
        probability_summary.to_string()
    )

    # ========================================================
    # 20. Feature importance
    # ========================================================

    section(
        "13. Final feature importance"
    )

    importance_df = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            final_model.feature_importances_,
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
    # 21. Save results
    # ========================================================

    section(
        "14. Saving results"
    )

    result = {

        "model":
            best_name,

        "PR_AUC":
            final_metrics["PR_AUC"],

        "ROC_AUC":
            final_metrics["ROC_AUC"],

        "Precision":
            final_metrics["Precision"],

        "Recall":
            final_metrics["Recall"],

        "F1":
            final_metrics["F1"],

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
            len(feature_columns),
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
    # 22. Final summary
    # ========================================================

    section(
        "FINAL SUMMARY"
    )

    print(
        f"Selected model: "
        f"{best_name}"
    )

    print(
        f"Final PR-AUC: "
        f"{final_metrics['PR_AUC']:.6f}"
    )

    print(
        f"Final ROC-AUC: "
        f"{final_metrics['ROC_AUC']:.6f}"
    )

    print(
        f"Final Precision: "
        f"{final_metrics['Precision']:.6f}"
    )

    print(
        f"Final Recall: "
        f"{final_metrics['Recall']:.6f}"
    )

    print(
        f"Final F1: "
        f"{final_metrics['F1']:.6f}"
    )

    print(
        "\nFinal holdout remained completely "
        "untouched during candidate selection."
    )

    print(
        "\nAll V3 rolling features use "
        "historical information only."
    )

    print(
        "\nDone."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()