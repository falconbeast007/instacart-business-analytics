from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
    "order_likelihood_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "order_likelihood_feature_importance.csv"
)

RANDOM_STATE = 42

# Business horizon:
# Predict whether the customer places their next order
# within 14 days.
HORIZON = 14


# ============================================================
# Helper
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    section("FUTURE ORDER LIKELIHOOD MODEL")

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    section("1. Loading forecasting dataset")

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    print(f"Rows:      {len(df):,}")
    print(
        f"Customers: "
        f"{df['user_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # 2. Validate target
    # --------------------------------------------------------

    target = "next_order_within_horizon"

    if target not in df.columns:

        raise ValueError(
            f"Required target '{target}' "
            f"not found in dataset."
        )

    # --------------------------------------------------------
    # 3. Sort chronologically
    # --------------------------------------------------------

    section("2. Sorting prediction points")

    df = df.sort_values(
        [
            "user_id",
            "prediction_order_number",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Per-customer temporal holdout
    # --------------------------------------------------------
    #
    # For every customer:
    #
    # Earlier prediction points -> TRAIN
    # Latest prediction point   -> VALIDATION
    #
    # This simulates using historical customer behavior
    # to predict their future behavior.
    # --------------------------------------------------------

    section(
        "3. Creating per-customer temporal holdout"
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
            df["customer_prediction_count"] >= 2,
            "user_id"
        ]
        .unique()
    )

    print(
        f"Customers with >=2 prediction points: "
        f"{len(eligible_customers):,}"
    )

    model_df = df[
        df["user_id"].isin(
            eligible_customers
        )
    ].copy()

    # Latest prediction point for each customer.
    latest_prediction_number = (
        model_df.groupby("user_id")
        ["prediction_order_number"]
        .transform("max")
    )

    validation_mask = (
        model_df["prediction_order_number"]
        == latest_prediction_number
    )

    validation = (
        model_df[
            validation_mask
        ]
        .copy()
    )

    train = (
        model_df[
            ~validation_mask
        ]
        .copy()
    )

    print(
        f"\nTraining rows   : "
        f"{len(train):,}"
    )

    print(
        f"Validation rows : "
        f"{len(validation):,}"
    )

    print(
        f"Training customers   : "
        f"{train['user_id'].nunique():,}"
    )

    print(
        f"Validation customers : "
        f"{validation['user_id'].nunique():,}"
    )

    print(
        "\nValidation represents the latest "
        "available prediction point for each customer."
    )

    # --------------------------------------------------------
    # 5. Target distribution
    # --------------------------------------------------------

    section("4. Target distribution")

    for name, data in [
        ("Training", train),
        ("Validation", validation),
    ]:

        positive_rate = (
            data[target]
            .mean()
        )

        print(
            f"{name:<12}: "
            f"Yes={positive_rate * 100:.2f}% | "
            f"No={(1 - positive_rate) * 100:.2f}%"
        )

    # --------------------------------------------------------
    # 6. Feature selection
    # --------------------------------------------------------

    section("5. Selecting model features")

    # Identifiers
    identifier_columns = {
        "user_id",
        "prediction_order_id",
        "customer_prediction_count",
    }

    # Future information
    future_columns = {
        "next_order_id",
        "next_order_number",
        "next_days_since_prior_order",
        "next_order_interval",
        "next_basket_size",
        "next_reorder_rate",
        "next_order_within_horizon",
        "next_order_within_7_days",
        "next_order_within_14_days",
        "next_order_within_30_days",
        "next_order_observed",
    }

    excluded_columns = (
        identifier_columns |
        future_columns
    )

    # --------------------------------------------------------
    # Important:
    #
    # prediction_order_number is intentionally removed.
    #
    # previous_orders already represents customer lifecycle
    # position and is more directly interpretable.
    # --------------------------------------------------------

    excluded_columns.add(
        "prediction_order_number"
    )

    feature_columns = [
        column
        for column in train.columns
        if column not in excluded_columns
    ]

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    for column in feature_columns:

        print(
            f"  - {column}"
        )

    X_train = (
        train[
            feature_columns
        ]
        .copy()
    )

    y_train = (
        train[target]
        .astype(int)
    )

    X_valid = (
        validation[
            feature_columns
        ]
        .copy()
    )

    y_valid = (
        validation[target]
        .astype(int)
    )

    # --------------------------------------------------------
    # 7. Numeric validation
    # --------------------------------------------------------

    section(
        "6. Validating model feature data types"
    )

    for column in feature_columns:

        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )

        X_valid[column] = pd.to_numeric(
            X_valid[column],
            errors="coerce",
        )

    # Replace infinities.
    X_train = (
        X_train
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    X_valid = (
        X_valid
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # Fill remaining missing values.
    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)

    # Verify numeric types.
    invalid_train_columns = [
        column
        for column in X_train.columns
        if not pd.api.types.is_numeric_dtype(
            X_train[column]
        )
    ]

    invalid_valid_columns = [
        column
        for column in X_valid.columns
        if not pd.api.types.is_numeric_dtype(
            X_valid[column]
        )
    ]

    if invalid_train_columns:

        raise TypeError(
            "Non-numeric training columns: "
            f"{invalid_train_columns}"
        )

    if invalid_valid_columns:

        raise TypeError(
            "Non-numeric validation columns: "
            f"{invalid_valid_columns}"
        )

    print(
        "All model features verified as numeric."
    )

    # --------------------------------------------------------
    # 8. Historical interval baseline
    # --------------------------------------------------------

    section(
        "7. Historical interval baseline"
    )

    baseline_predictions = (
        validation[
            "historical_avg_days_between_orders"
        ]
        <= HORIZON
    ).astype(int)

    baseline_metrics = {

        "model":
            "Historical Interval Baseline",

        "PR_AUC":
            average_precision_score(
                y_valid,
                baseline_predictions,
            ),

        "ROC_AUC":
            roc_auc_score(
                y_valid,
                baseline_predictions,
            ),

        "Precision":
            precision_score(
                y_valid,
                baseline_predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_valid,
                baseline_predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y_valid,
                baseline_predictions,
                zero_division=0,
            ),
    }

    print(
        pd.Series(
            baseline_metrics
        ).to_string()
    )

    # --------------------------------------------------------
    # 9. Train XGBoost
    # --------------------------------------------------------

    section("8. Training XGBoost")

    positive_count = int(
        y_train.sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    scale_pos_weight = (
        negative_count /
        positive_count
        if positive_count > 0
        else 1.0
    )

    print(
        f"Positive training rows : "
        f"{positive_count:,}"
    )

    print(
        f"Negative training rows : "
        f"{negative_count:,}"
    )

    print(
        f"Scale pos weight       : "
        f"{scale_pos_weight:.3f}"
    )

    model = XGBClassifier(

        n_estimators=500,

        max_depth=6,

        learning_rate=0.05,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        scale_pos_weight=scale_pos_weight,

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],

        verbose=False,
    )

    print(
        "XGBoost training completed."
    )

    # --------------------------------------------------------
    # 10. Generate probabilities
    # --------------------------------------------------------

    section(
        "9. Generating validation predictions"
    )

    probabilities = (
        model
        .predict_proba(
            X_valid
        )[:, 1]
    )

    # Keep 0.50 threshold only for classification metrics.
    # Dashboard 4 will use the continuous probability.
    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # 11. Model performance
    # --------------------------------------------------------

    section(
        "10. Model performance"
    )

    model_metrics = {

        "model":
            "XGBoost",

        "PR_AUC":
            average_precision_score(
                y_valid,
                probabilities,
            ),

        "ROC_AUC":
            roc_auc_score(
                y_valid,
                probabilities,
            ),

        "Precision":
            precision_score(
                y_valid,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_valid,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
    }

    metrics_df = pd.DataFrame(
        [
            baseline_metrics,
            model_metrics,
        ]
    )

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 12. Probability distribution
    # --------------------------------------------------------

    section(
        "11. Prediction probability distribution"
    )

    probability_series = pd.Series(
        probabilities,
        name="predicted_probability",
    )

    print(
        probability_series
        .describe()
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # 13. Feature importance
    # --------------------------------------------------------

    section(
        "12. Feature importance"
    )

    importance_df = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            model.feature_importances_,
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        importance_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # 14. Save results
    # --------------------------------------------------------

    section(
        "13. Saving results"
    )

    metrics_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    print(
        "Results saved:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {FEATURE_IMPORTANCE_PATH}"
    )

    # --------------------------------------------------------
    # 15. Compare baseline and XGBoost
    # --------------------------------------------------------

    section(
        "14. Baseline vs XGBoost"
    )

    print(
        f"PR-AUC improvement : "
        f"{model_metrics['PR_AUC'] - baseline_metrics['PR_AUC']:+.4f}"
    )

    print(
        f"Precision change   : "
        f"{model_metrics['Precision'] - baseline_metrics['Precision']:+.4f}"
    )

    print(
        f"Recall change      : "
        f"{model_metrics['Recall'] - baseline_metrics['Recall']:+.4f}"
    )

    print(
        f"F1 change          : "
        f"{model_metrics['F1'] - baseline_metrics['F1']:+.4f}"
    )

    # --------------------------------------------------------
    # 16. Final summary
    # --------------------------------------------------------

    section("FINAL SUMMARY")

    print(
        f"Forecast horizon       : "
        f"{HORIZON} days"
    )

    print(
        f"Training rows          : "
        f"{len(train):,}"
    )

    print(
        f"Validation rows        : "
        f"{len(validation):,}"
    )

    print(
        f"Validation customers   : "
        f"{validation['user_id'].nunique():,}"
    )

    print(
        f"Baseline PR-AUC        : "
        f"{baseline_metrics['PR_AUC']:.4f}"
    )

    print(
        f"XGBoost PR-AUC         : "
        f"{model_metrics['PR_AUC']:.4f}"
    )

    print(
        f"Baseline F1            : "
        f"{baseline_metrics['F1']:.4f}"
    )

    print(
        f"XGBoost F1             : "
        f"{model_metrics['F1']:.4f}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()