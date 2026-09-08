"""
Step 14: Tune XGBoost for next-product recommendation.

Goal:
Find a stronger XGBoost configuration than the baseline while
keeping training computationally reasonable.

Primary metrics:
- PR-AUC
- Precision@10
- Recall@10
- Precision@20
- Recall@20

Validation:
Same deterministic customer-level split used in Step 13.

Important:
Negative sampling is kept identical across experiments so that
model comparisons are fair.
"""

import os
import time
import duckdb
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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
# TOP-K METRICS
# ============================================================

def ranking_metrics(
    df,
    prediction_column="prediction",
    target_column="target"
):

    df = df.sort_values(
        ["user_id", prediction_column],
        ascending=[True, False]
    )

    results = {}

    # Only customers with actual positive products
    actual = (
        df.groupby("user_id")[target_column]
        .sum()
    )

    actual = actual[actual > 0]

    df = df[
        df["user_id"].isin(actual.index)
    ]

    for k in [5, 10, 20]:

        top_k = (
            df.groupby("user_id", sort=False)
            .head(k)
        )

        hits = (
            top_k.groupby("user_id")[target_column]
            .sum()
        )

        hits = hits.reindex(
            actual.index,
            fill_value=0
        )

        precision = (
            hits / k
        ).mean()

        recall = (
            hits / actual
        ).mean()

        results[f"precision_at_{k}"] = precision
        results[f"recall_at_{k}"] = recall

    return results


# ============================================================
# CONNECT
# ============================================================

log("Connecting to DuckDB...")

con = duckdb.connect(DB_PATH)


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

train_df = con.execute(
    training_query
).fetchdf()

log(
    f"Training rows loaded: "
    f"{len(train_df):,}"
)


# ============================================================
# SPLIT POSITIVE / NEGATIVE
# ============================================================

positive_train = train_df[
    train_df["target"] == 1
].copy()

negative_train = train_df[
    train_df["target"] == 0
].copy()

log(
    f"Positive rows: "
    f"{len(positive_train):,}"
)

log(
    f"Negative rows available: "
    f"{len(negative_train):,}"
)


# ============================================================
# FIXED NEGATIVE SAMPLE
# ============================================================

negative_sample_size = min(
    len(negative_train),
    len(positive_train) * 5
)

log(
    f"Sampling {negative_sample_size:,} "
    f"negative rows..."
)

# IMPORTANT:
# Same negative sample is used for every experiment.

negative_sample = negative_train.sample(
    n=negative_sample_size,
    random_state=RANDOM_STATE
)

training_sample = pd.concat(
    [
        positive_train,
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


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

log("Loading validation data...")

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

X_validation = validation_df[FEATURES]

y_validation = (
    validation_df["target"]
    .astype(int)
)


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

experiments = [

    {
        "name": "Baseline",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.08,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "name": "Deeper",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.08,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "name": "Shallower",
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.06,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "name": "Regularized",
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.06,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "name": "MoreTrees",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
]


# ============================================================
# RUN EXPERIMENTS
# ============================================================

all_results = []

print()
print("=" * 80)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 80)

for i, params in enumerate(experiments, start=1):

    name = params["name"]

    print()
    print("-" * 80)
    print(
        f"EXPERIMENT {i}/{len(experiments)}: "
        f"{name}"
    )
    print("-" * 80)

    print(
        f"n_estimators       : "
        f"{params['n_estimators']}"
    )

    print(
        f"max_depth          : "
        f"{params['max_depth']}"
    )

    print(
        f"learning_rate      : "
        f"{params['learning_rate']}"
    )

    print(
        f"min_child_weight   : "
        f"{params['min_child_weight']}"
    )

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],

        min_child_weight=params[
            "min_child_weight"
        ],

        subsample=params["subsample"],
        colsample_bytree=params[
            "colsample_bytree"
        ],

        objective="binary:logistic",
        eval_metric="aucpr",

        tree_method="hist",

        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    start = time.time()

    log("Training model...")

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start

    log(
        f"Training complete in "
        f"{training_time:.1f}s"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    log("Generating predictions...")

    predictions = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        y_validation,
        predictions
    )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    evaluation_df = validation_df[
        [
            "user_id",
            "product_id",
            "target"
        ]
    ].copy()

    evaluation_df["prediction"] = predictions

    metrics = ranking_metrics(
        evaluation_df
    )

    result = {
        "experiment": name,

        "n_estimators":
            params["n_estimators"],

        "max_depth":
            params["max_depth"],

        "learning_rate":
            params["learning_rate"],

        "min_child_weight":
            params["min_child_weight"],

        "subsample":
            params["subsample"],

        "colsample_bytree":
            params["colsample_bytree"],

        "training_time_seconds":
            training_time,

        "pr_auc":
            pr_auc,

        "precision_at_5":
            metrics["precision_at_5"],

        "recall_at_5":
            metrics["recall_at_5"],

        "precision_at_10":
            metrics["precision_at_10"],

        "recall_at_10":
            metrics["recall_at_10"],

        "precision_at_20":
            metrics["precision_at_20"],

        "recall_at_20":
            metrics["recall_at_20"],
    }

    all_results.append(result)

    print()
    print(
        f"PR-AUC       : "
        f"{pr_auc:.4f}"
    )

    print(
        f"Precision@10 : "
        f"{metrics['precision_at_10']:.4f}"
    )

    print(
        f"Recall@10    : "
        f"{metrics['recall_at_10']:.4f}"
    )

    print(
        f"Precision@20 : "
        f"{metrics['precision_at_20']:.4f}"
    )

    print(
        f"Recall@20    : "
        f"{metrics['recall_at_20']:.4f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# RANKING
# ============================================================

# We primarily care about recommendation ranking.
#
# PR-AUC is used as the first criterion,
# Recall@10 as second,
# Recall@20 as third.

results_df = results_df.sort_values(
    [
        "pr_auc",
        "recall_at_10",
        "recall_at_20"
    ],
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_tuning_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print()
print("=" * 100)
print("XGBOOST TUNING RESULTS")
print("=" * 100)

display_columns = [
    "experiment",
    "pr_auc",
    "precision_at_5",
    "recall_at_5",
    "precision_at_10",
    "recall_at_10",
    "precision_at_20",
    "recall_at_20",
    "training_time_seconds"
]

print(
    results_df[
        display_columns
    ].to_string(index=False)
)

print("=" * 100)


# ============================================================
# BEST MODEL
# ============================================================

best = results_df.iloc[0]

print()
print("=" * 80)
print("BEST XGBOOST CONFIGURATION")
print("=" * 80)

print(
    f"Experiment          : "
    f"{best['experiment']}"
)

print(
    f"PR-AUC              : "
    f"{best['pr_auc']:.4f}"
)

print(
    f"Precision@5         : "
    f"{best['precision_at_5']:.4f}"
)

print(
    f"Recall@5            : "
    f"{best['recall_at_5']:.4f}"
)

print(
    f"Precision@10        : "
    f"{best['precision_at_10']:.4f}"
)

print(
    f"Recall@10           : "
    f"{best['recall_at_10']:.4f}"
)

print(
    f"Precision@20        : "
    f"{best['precision_at_20']:.4f}"
)

print(
    f"Recall@20           : "
    f"{best['recall_at_20']:.4f}"
)

print()
print(
    f"n_estimators        : "
    f"{int(best['n_estimators'])}"
)

print(
    f"max_depth           : "
    f"{int(best['max_depth'])}"
)

print(
    f"learning_rate       : "
    f"{best['learning_rate']}"
)

print(
    f"min_child_weight    : "
    f"{int(best['min_child_weight'])}"
)

print("=" * 80)


# ============================================================
# CLOSE
# ============================================================

con.close()

log("Done.")