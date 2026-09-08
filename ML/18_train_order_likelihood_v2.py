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
    "order_likelihood_v2_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DIR /
    "order_likelihood_v2_feature_importance.csv"
)

RANDOM_STATE = 42


# ============================================================
# Helpers
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


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


def train_xgb(
    X_train,
    y_train,
    X_eval,
    objective="binary:logistic",
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    min_child_weight=5,
    scale_pos_weight=None,
):

    model = XGBClassifier(

        n_estimators=n_estimators,

        max_depth=max_depth,

        learning_rate=learning_rate,

        min_child_weight=min_child_weight,

        subsample=0.8,

        colsample_bytree=0.8,

        objective=objective,

        eval_metric="aucpr",

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    probabilities = model.predict_proba(
        X_eval
    )[:, 1]

    return model, probabilities


# ============================================================
# Main
# ============================================================

def main():

    section(
        "MODEL 1 V2 — FUTURE ORDER LIKELIHOOD"
    )

    # ========================================================
    # 1. Load data
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

    target = (
        "next_order_within_horizon"
    )

    if target not in df.columns:

        raise ValueError(
            f"Target '{target}' not found."
        )

    # ========================================================
    # 2. Sort
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

    train = df[
        df["reverse_rank"] >= 2
    ].copy()

    selection_holdout = df[
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
    # 4. V2 feature engineering
    # ========================================================

    section(
        "4. Engineering V2 behavioral features"
    )

    # --------------------------------------------------------
    # Recent 5 order interval
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Recent 5 basket
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Recent 5 reorder rate
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Interval deviation
    # --------------------------------------------------------

    df["interval_deviation"] = (
        df["current_order_days_since_prior"]
        -
        df["historical_avg_days_between_orders"]
    )

    # --------------------------------------------------------
    # Interval ratio
    # --------------------------------------------------------

    denominator = (
        df["historical_avg_days_between_orders"]
        .replace(
            0,
            np.nan,
        )
    )

    df["interval_ratio"] = (
        df["current_order_days_since_prior"]
        /
        denominator
    )

    # --------------------------------------------------------
    # Recent interval vs historical interval
    # --------------------------------------------------------

    df["recent_interval_deviation"] = (
        df["recent_3_order_interval"]
        -
        df["historical_avg_days_between_orders"]
    )

    # --------------------------------------------------------
    # Reorder behavior trend
    # --------------------------------------------------------

    df["reorder_rate_trend"] = (
        df["recent_3_order_reorder_rate"]
        -
        df["historical_reorder_rate"]
    )

    # --------------------------------------------------------
    # Basket behavior trend
    # --------------------------------------------------------

    df["basket_size_trend"] = (
        df["recent_3_order_basket"]
        -
        df["historical_avg_basket_size"]
    )

    # --------------------------------------------------------
    # Interval volatility
    # --------------------------------------------------------

    df["interval_std_5"] = (
        df.groupby("user_id")
        ["current_order_days_since_prior"]
        .transform(
            lambda x:
                x.rolling(
                    window=5,
                    min_periods=2,
                ).std()
        )
    )

    # --------------------------------------------------------
    # Recent interval volatility
    # --------------------------------------------------------

    df["interval_std_3"] = (
        df.groupby("user_id")
        ["current_order_days_since_prior"]
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

    v2_features = [

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

    for column in v2_features:

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

    print(
        f"Added V2 features: "
        f"{len(v2_features)}"
    )

    for feature in v2_features:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # 5. Feature selection
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
        v2_features
    )

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # 6. Prepare matrices
    # ========================================================

    section(
        "6. Preparing train / selection / final data"
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
    ].astype(int)

    y_selection = selection_holdout[
        target
    ].astype(int)

    y_final = final_holdout[
        target
    ].astype(int)

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
    # 7. Class balance
    # ========================================================

    section(
        "7. Class balance"
    )

    positive_count = (
        y_train.sum()
    )

    negative_count = (
        len(y_train) -
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
    # 8. Candidate models
    # ========================================================

    section(
        "8. Training candidate models"
    )

    candidates = []

    # --------------------------------------------------------
    # Candidate A
    # Current V1-style model
    # --------------------------------------------------------

    print(
        "\nCandidate A: V1 Baseline Configuration"
    )

    model_a, prediction_a = train_xgb(

        X_train,

        y_train,

        X_selection,

        max_depth=6,

        n_estimators=500,

        learning_rate=0.05,

        min_child_weight=5,

        scale_pos_weight=scale_pos_weight,
    )

    candidates.append(
        (
            "XGBoost V1 Configuration",
            model_a,
            prediction_a,
        )
    )

    # --------------------------------------------------------
    # Candidate B
    # Deeper
    # --------------------------------------------------------

    print(
        "\nCandidate B: Deeper XGBoost"
    )

    model_b, prediction_b = train_xgb(

        X_train,

        y_train,

        X_selection,

        max_depth=8,

        n_estimators=500,

        learning_rate=0.05,

        min_child_weight=5,

        scale_pos_weight=scale_pos_weight,
    )

    candidates.append(
        (
            "XGBoost V2 Deep",
            model_b,
            prediction_b,
        )
    )

    # --------------------------------------------------------
    # Candidate C
    # More trees
    # --------------------------------------------------------

    print(
        "\nCandidate C: More Trees"
    )

    model_c, prediction_c = train_xgb(

        X_train,

        y_train,

        X_selection,

        max_depth=6,

        n_estimators=700,

        learning_rate=0.04,

        min_child_weight=5,

        scale_pos_weight=scale_pos_weight,
    )

    candidates.append(
        (
            "XGBoost V2 MoreTrees",
            model_c,
            prediction_c,
        )
    )

    # --------------------------------------------------------
    # Candidate D
    # More regularized
    # --------------------------------------------------------

    print(
        "\nCandidate D: Regularized XGBoost"
    )

    model_d, prediction_d = train_xgb(

        X_train,

        y_train,

        X_selection,

        max_depth=6,

        n_estimators=600,

        learning_rate=0.04,

        min_child_weight=10,

        scale_pos_weight=scale_pos_weight,
    )

    candidates.append(
        (
            "XGBoost V2 Regularized",
            model_d,
            prediction_d,
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

    for name, model, probabilities in candidates:

        metric = calculate_metrics(
            actual_selection,
            probabilities,
        )

        selection_results.append({

            "model":
                name,

            "PR_AUC":
                metric["PR_AUC"],

            "ROC_AUC":
                metric["ROC_AUC"],

            "Precision":
                metric["Precision"],

            "Recall":
                metric["Recall"],

            "F1":
                metric["F1"],
        })

        print(
            f"\n{name}"
        )

        print(
            f"PR-AUC   : "
            f"{metric['PR_AUC']:.6f}"
        )

        print(
            f"ROC-AUC  : "
            f"{metric['ROC_AUC']:.6f}"
        )

        print(
            f"Precision: "
            f"{metric['Precision']:.6f}"
        )

        print(
            f"Recall   : "
            f"{metric['Recall']:.6f}"
        )

        print(
            f"F1       : "
            f"{metric['F1']:.6f}"
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
    # 10. Select candidate
    # ========================================================

    section(
        "10. Selecting V2 candidate"
    )

    best_name = (
        selection_df.iloc[0]["model"]
    )

    print(
        f"Selected candidate: "
        f"{best_name}"
    )

    selected_tuple = next(
        item
        for item in candidates
        if item[0] == best_name
    )

    # ========================================================
    # 11. Retrain selected model
    # ========================================================

    section(
        "11. Retraining selected model"
    )

    if best_name == (
        "XGBoost V2 Deep"
    ):

        final_model = XGBClassifier(

            n_estimators=500,

            max_depth=8,

            learning_rate=0.05,

            min_child_weight=5,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="binary:logistic",

            eval_metric="aucpr",

            tree_method="hist",

            n_jobs=-1,

            random_state=RANDOM_STATE,

            scale_pos_weight=scale_pos_weight,
        )

    elif best_name == (
        "XGBoost V2 MoreTrees"
    ):

        final_model = XGBClassifier(

            n_estimators=700,

            max_depth=6,

            learning_rate=0.04,

            min_child_weight=5,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="binary:logistic",

            eval_metric="aucpr",

            tree_method="hist",

            n_jobs=-1,

            random_state=RANDOM_STATE,

            scale_pos_weight=scale_pos_weight,
        )

    elif best_name == (
        "XGBoost V2 Regularized"
    ):

        final_model = XGBClassifier(

            n_estimators=600,

            max_depth=6,

            learning_rate=0.04,

            min_child_weight=10,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="binary:logistic",

            eval_metric="aucpr",

            tree_method="hist",

            n_jobs=-1,

            random_state=RANDOM_STATE,

            scale_pos_weight=scale_pos_weight,
        )

    else:

        final_model = XGBClassifier(

            n_estimators=500,

            max_depth=6,

            learning_rate=0.05,

            min_child_weight=5,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="binary:logistic",

            eval_metric="aucpr",

            tree_method="hist",

            n_jobs=-1,

            random_state=RANDOM_STATE,

            scale_pos_weight=scale_pos_weight,
        )

    final_model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    final_probabilities = (
        final_model.predict_proba(
            X_final
        )[:, 1]
    )

    # ========================================================
    # 12. Final holdout evaluation
    # ========================================================

    section(
        "12. FINAL UNTOUCHED HOLDOUT RESULTS"
    )

    actual_final = (
        y_final.to_numpy()
    )

    final_metrics = calculate_metrics(
        actual_final,
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
    # 13. Final probability distribution
    # ========================================================

    section(
        "13. Prediction probability distribution"
    )

    print(
        pd.Series(
            final_probabilities,
            name="order_likelihood",
        )
        .describe()
        .round(4)
        .to_string()
    )

    # ========================================================
    # 14. Feature importance
    # ========================================================

    section(
        "14. Final feature importance"
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
    # 15. Save
    # ========================================================

    section(
        "15. Saving results"
    )

    result_row = {

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
            len(selection_holdout),

        "final_holdout_rows":
            len(final_holdout),

        "final_holdout_customers":
            final_holdout[
                "user_id"
            ].nunique(),
    }

    pd.DataFrame(
        [result_row]
    ).to_csv(
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
    # 16. Final summary
    # ========================================================

    section(
        "FINAL SUMMARY"
    )

    print(
        f"Selected model:"
        f" {best_name}"
    )

    print(
        f"Final PR-AUC:"
        f" {final_metrics['PR_AUC']:.6f}"
    )

    print(
        f"Final ROC-AUC:"
        f" {final_metrics['ROC_AUC']:.6f}"
    )

    print(
        f"Final Precision:"
        f" {final_metrics['Precision']:.6f}"
    )

    print(
        f"Final Recall:"
        f" {final_metrics['Recall']:.6f}"
    )

    print(
        f"Final F1:"
        f" {final_metrics['F1']:.6f}"
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