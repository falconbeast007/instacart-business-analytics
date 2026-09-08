"""
Step 13: Train XGBoost model for next-product recommendation.

Business Question:
Can we predict which products a customer is likely to purchase
in their next order based on previous shopping behavior?

Model:
XGBoost

Validation:
Deterministic customer-level split.

Metrics:
- PR-AUC
- Precision
- Recall
- F1
- Precision@5, @10, @20
- Recall@5, @10, @20

Outputs:
- xgboost_feature_importance.csv
- xgboost_results.csv
- xgboost_ranking_results.csv
- logistic_vs_xgboost_comparison.csv
- logistic_vs_xgboost_ranking_comparison.csv
"""

import os
import time
import duckdb
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "instacart.duckdb"
)

ML_DATASET = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_final_dataset.parquet"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RANDOM_STATE = 42

FEATURES = [
    "previous_purchases",
    "previous_reorders",
    "customer_product_reorder_rate",
    "first_purchase_order",
    "last_purchase_order",
    "purchase_recency",
    "avg_cart_position",

    "previous_orders",
    "avg_basket_size",
    "avg_days_between_orders",
    "customer_reorder_rate",

    "product_purchase_count",
    "product_unique_customers",
    "product_reorder_rate",

    "customer_department_purchases",
    "customer_department_share",

    "product_department_popularity",
]


# ============================================================
# TIMER
# ============================================================

START_TIME = time.time()


def log(message):
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:8.1f}s] {message}")


# ============================================================
# HELPER: TOP-K METRICS
# ============================================================

def calculate_ranking_metrics(
    df,
    score_column="prediction",
    target_column="target",
    k_values=(5, 10, 20)
):
    """
    Calculate Precision@K and Recall@K at customer level.

    For every customer:
        1. Sort candidate products by model score.
        2. Take top K.
        3. Compare against actual purchased products.
    """

    results = []

    # Only customers with at least one positive target
    customers_with_targets = (
        df.groupby("user_id")[target_column]
        .sum()
    )

    customers_with_targets = customers_with_targets[
        customers_with_targets > 0
    ].index

    df = df[
        df["user_id"].isin(customers_with_targets)
    ].copy()

    # Sort once
    df = df.sort_values(
        ["user_id", score_column],
        ascending=[True, False]
    )

    for k in k_values:

        top_k = (
            df.groupby("user_id", sort=False)
            .head(k)
        )

        precision_values = (
            top_k.groupby("user_id")[target_column]
            .sum()
            / k
        )

        actual_counts = (
            df.groupby("user_id")[target_column]
            .sum()
        )

        hits = (
            top_k.groupby("user_id")[target_column]
            .sum()
        )

        recall_values = (
            hits / actual_counts
        )

        results.append({
            "k": k,
            "precision_at_k": precision_values.mean(),
            "recall_at_k": recall_values.mean()
        })

    return pd.DataFrame(results)


# ============================================================
# CONNECT TO DUCKDB
# ============================================================

log("Connecting to DuckDB...")

con = duckdb.connect(DB_PATH)


# ============================================================
# CHECK ML DATASET
# ============================================================

log("Checking ML dataset...")

dataset_check = con.execute(
    f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT user_id) AS customers,
        SUM(target) AS positive,
        SUM(CASE WHEN target = 0 THEN 1 ELSE 0 END) AS negative
    FROM read_parquet('{ML_DATASET}')
    """
).fetchone()

rows, customers, positive, negative = dataset_check

print()
print("=" * 70)
print("ML DATASET")
print("=" * 70)
print(f"Rows:                 {rows:,}")
print(f"Customers:            {customers:,}")
print(f"Positive:             {positive:,}")
print(f"Negative:             {negative:,}")
print("=" * 70)


# ============================================================
# CREATE DETERMINISTIC CUSTOMER SPLIT
# ============================================================

log("Creating deterministic customer-level split...")

split_query = f"""
SELECT
    user_id,
    CASE
        WHEN MOD(ABS(HASH(user_id)), 100) < 20
        THEN 'validation'
        ELSE 'train'
    END AS split
FROM read_parquet('{ML_DATASET}')
GROUP BY user_id
"""

split_df = con.execute(split_query).fetchdf()

train_customers = (
    split_df.loc[
        split_df["split"] == "train",
        "user_id"
    ]
    .tolist()
)

validation_customers = (
    split_df.loc[
        split_df["split"] == "validation",
        "user_id"
    ]
    .tolist()
)

print()
print("=" * 70)
print("CUSTOMER SPLIT")
print("=" * 70)
print(f"Train               : {len(train_customers):,}")
print(f"Validation          : {len(validation_customers):,}")
print("=" * 70)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

log("Loading training data...")

training_query = f"""
SELECT
    user_id,
    product_id,
    {", ".join(FEATURES)},
    target
FROM read_parquet('{ML_DATASET}')
WHERE MOD(ABS(HASH(user_id)), 100) >= 20
"""

train_df = con.execute(training_query).fetchdf()

log(f"Training rows loaded: {len(train_df):,}")


# ============================================================
# TRAINING DATA
# ============================================================

X_train_full = train_df[FEATURES]
y_train_full = train_df["target"].astype(int)

positive_train = int(y_train_full.sum())

log(f"All positive rows: {positive_train:,}")

negative_train_df = train_df[
    train_df["target"] == 0
]

negative_count_available = len(negative_train_df)

# 5 negative examples per positive
negative_sample_size = min(
    negative_count_available,
    positive_train * 5
)

log(
    f"Available negative rows: "
    f"{negative_count_available:,}"
)

log(
    f"Sampling {negative_sample_size:,} "
    f"negative rows..."
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

negative_sample = negative_train_df.sample(
    n=negative_sample_size,
    random_state=RANDOM_STATE
)

positive_sample = train_df[
    train_df["target"] == 1
]

training_sample = pd.concat(
    [
        positive_sample,
        negative_sample
    ],
    ignore_index=True
)

training_sample = training_sample.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


X_train = training_sample[FEATURES]
y_train = training_sample["target"].astype(int)


print()
print("=" * 70)
print("XGBOOST TRAINING SAMPLE")
print("=" * 70)
print(f"Rows:                 {len(training_sample):,}")
print(f"Positive:             {int(y_train.sum()):,}")
print(
    f"Negative:             "
    f"{int((y_train == 0).sum()):,}"
)
print(
    f"Positive rate:        "
    f"{y_train.mean():.2%}"
)
print("=" * 70)


# ============================================================
# INITIALIZE XGBOOST
# ============================================================

log("Initializing XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=5,

    objective="binary:logistic",
    eval_metric="aucpr",

    tree_method="hist",

    n_jobs=-1,
    random_state=RANDOM_STATE
)


# ============================================================
# TRAIN MODEL
# ============================================================

log("Training XGBoost...")

model.fit(
    X_train,
    y_train
)

log("XGBoost training complete.")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


print()
print("=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)
print(
    feature_importance.to_string(index=False)
)
print("=" * 70)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_path,
    index=False
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

log("Loading validation data...")


# IMPORTANT:
# Every column that can be ambiguous is explicitly prefixed
# with "d." below.
#
# This fixes the DuckDB BinderException:
# "Ambiguous reference to column name user_id"
#
# We don't actually need another table for the validation set,
# so this query simply reads the parquet dataset and applies
# the exact same deterministic split.

validation_query = f"""
SELECT
    d.user_id,
    d.product_id,
    {", ".join([f"d.{feature}" for feature in FEATURES])},
    d.target
FROM read_parquet('{ML_DATASET}') AS d
WHERE MOD(ABS(HASH(d.user_id)), 100) < 20
"""

validation_df = con.execute(
    validation_query
).fetchdf()

log(
    f"Validation rows loaded: "
    f"{len(validation_df):,}"
)


# ============================================================
# VALIDATION DATA
# ============================================================

X_validation = validation_df[FEATURES]
y_validation = validation_df["target"].astype(int)


print()
print("=" * 70)
print("VALIDATION DATA")
print("=" * 70)
print(f"Rows:                 {len(validation_df):,}")
print(f"Positive:             {int(y_validation.sum()):,}")
print(
    f"Negative:             "
    f"{int((y_validation == 0).sum()):,}"
)
print(
    f"Positive rate:        "
    f"{y_validation.mean():.2%}"
)
print("=" * 70)


# ============================================================
# PREDICT
# ============================================================

log("Generating validation predictions...")

validation_predictions = model.predict_proba(
    X_validation
)[:, 1]

validation_df["prediction"] = validation_predictions


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

log("Calculating classification metrics...")

PR_AUC = average_precision_score(
    y_validation,
    validation_predictions
)


# Use 0.5 threshold for standard classification metrics
prediction_class = (
    validation_predictions >= 0.5
).astype(int)


precision = precision_score(
    y_validation,
    prediction_class,
    zero_division=0
)

recall = recall_score(
    y_validation,
    prediction_class,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    prediction_class,
    zero_division=0
)


print()
print("=" * 70)
print("XGBOOST CLASSIFICATION RESULTS")
print("=" * 70)
print(f"PR-AUC               : {PR_AUC:.4f}")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1                   : {f1:.4f}")
print("=" * 70)


# ============================================================
# RANKING METRICS
# ============================================================

log("Calculating ranking metrics...")

ranking_results = calculate_ranking_metrics(
    validation_df,
    score_column="prediction",
    target_column="target",
    k_values=(5, 10, 20)
)


print()
print("=" * 70)
print("XGBOOST RANKING RESULTS")
print("=" * 70)

for _, row in ranking_results.iterrows():

    k = int(row["k"])

    print(
        f"Precision@{k:<2}         : "
        f"{row['precision_at_k']:.4f}"
    )

    print(
        f"Recall@{k:<2}            : "
        f"{row['recall_at_k']:.4f}"
    )

print("=" * 70)


# ============================================================
# SAVE XGBOOST RESULTS
# ============================================================

results_df = pd.DataFrame([
    {
        "model": "XGBoost",
        "pr_auc": PR_AUC,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
])

xgb_results_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_results.csv"
)

results_df.to_csv(
    xgb_results_path,
    index=False
)


# ============================================================
# SAVE RANKING RESULTS
# ============================================================

ranking_save = ranking_results.copy()

ranking_save.insert(
    0,
    "model",
    "XGBoost"
)

xgb_ranking_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_ranking_results.csv"
)

ranking_save.to_csv(
    xgb_ranking_path,
    index=False
)


# ============================================================
# LOAD LOGISTIC REGRESSION RESULTS
# ============================================================

log("Loading Logistic Regression results...")

logistic_results_path = os.path.join(
    OUTPUT_DIR,
    "logistic_regression_results.csv"
)

logistic_ranking_path = os.path.join(
    OUTPUT_DIR,
    "logistic_regression_ranking_results.csv"
)


# ============================================================
# COMPARE CLASSIFICATION MODELS
# ============================================================

if os.path.exists(logistic_results_path):

    logistic_results = pd.read_csv(
        logistic_results_path
    )

    comparison = pd.concat(
        [
            logistic_results,
            results_df
        ],
        ignore_index=True
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "logistic_vs_xgboost_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION VS XGBOOST")
    print("=" * 70)
    print(
        comparison.to_string(index=False)
    )
    print("=" * 70)

else:

    print(
        "\nLogistic Regression results file "
        "not found. Skipping comparison."
    )


# ============================================================
# COMPARE RANKING MODELS
# ============================================================

if os.path.exists(logistic_ranking_path):

    logistic_ranking = pd.read_csv(
        logistic_ranking_path
    )

    # Normalize column naming if needed
    if "model" not in logistic_ranking.columns:
        logistic_ranking.insert(
            0,
            "model",
            "Logistic Regression"
        )

    xgb_ranking_for_compare = ranking_results.copy()

    xgb_ranking_for_compare.insert(
        0,
        "model",
        "XGBoost"
    )

    ranking_comparison = pd.concat(
        [
            logistic_ranking,
            xgb_ranking_for_compare
        ],
        ignore_index=True
    )

    ranking_comparison_path = os.path.join(
        OUTPUT_DIR,
        "logistic_vs_xgboost_ranking_comparison.csv"
    )

    ranking_comparison.to_csv(
        ranking_comparison_path,
        index=False
    )

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION VS XGBOOST — RANKING")
    print("=" * 70)
    print(
        ranking_comparison.to_string(index=False)
    )
    print("=" * 70)

else:

    print(
        "\nLogistic Regression ranking results "
        "not found. Skipping ranking comparison."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL XGBOOST SUMMARY")
print("=" * 70)

print(
    f"Training sample       : "
    f"{len(X_train):,}"
)

print(
    f"Validation rows       : "
    f"{len(X_validation):,}"
)

print(
    f"PR-AUC                : "
    f"{PR_AUC:.4f}"
)

print(
    f"Precision             : "
    f"{precision:.4f}"
)

print(
    f"Recall                : "
    f"{recall:.4f}"
)

print(
    f"F1                    : "
    f"{f1:.4f}"
)

for _, row in ranking_results.iterrows():

    k = int(row["k"])

    print(
        f"Precision@{k:<2}          : "
        f"{row['precision_at_k']:.4f}"
    )

    print(
        f"Recall@{k:<2}             : "
        f"{row['recall_at_k']:.4f}"
    )

print()
print("Saved files:")
print(
    f"  {feature_importance_path}"
)
print(
    f"  {xgb_results_path}"
)
print(
    f"  {xgb_ranking_path}"
)

if os.path.exists(
    os.path.join(
        OUTPUT_DIR,
        "logistic_vs_xgboost_comparison.csv"
    )
):
    print(
        f"  {comparison_path}"
    )

if os.path.exists(
    os.path.join(
        OUTPUT_DIR,
        "logistic_vs_xgboost_ranking_comparison.csv"
    )
):
    print(
        f"  {ranking_comparison_path}"
    )

print("=" * 70)


# ============================================================
# CLOSE CONNECTION
# ============================================================

con.close()

log("Done.")