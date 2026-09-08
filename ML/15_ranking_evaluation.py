"""
Step 15: Proper Ranking Evaluation

Business Question:
How well does the recommendation model rank products
that the customer actually purchases?

Models:
1. Global Popularity
2. Logistic Regression
3. Tuned XGBoost

Metrics:
- MAP@5
- MAP@10
- MAP@20
- NDCG@5
- NDCG@10
- NDCG@20
- Precision@K
- Recall@K

Validation:
Same deterministic customer-level split used throughout
the ML experiments.

Important:
This evaluates ranking quality, not merely binary classification.
"""

import os
import time

import duckdb
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
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

K_VALUES = [5, 10, 20]

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

    print(
        f"[{elapsed:8.1f}s] {message}"
    )


# ============================================================
# RANKING METRICS
# ============================================================

def calculate_ranking_metrics(
    df,
    score_column,
    model_name
):

    """
    Calculate:

        Precision@K
        Recall@K
        AP@K
        MAP@K
        NDCG@K

    at customer level.

    Each customer has a set of candidate products.
    Products are ranked by the supplied score.
    """

    print()
    print(
        "=" * 80
    )

    print(
        f"RANKING EVALUATION: "
        f"{model_name}"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Sort candidates by score
    # --------------------------------------------------------

    ranked = df.sort_values(
        [
            "user_id",
            score_column
        ],
        ascending=[
            True,
            False
        ]
    ).copy()

    # --------------------------------------------------------
    # Only customers with actual positives
    # --------------------------------------------------------

    actual_counts = (
        ranked
        .groupby("user_id")["target"]
        .sum()
    )

    actual_counts = actual_counts[
        actual_counts > 0
    ]

    ranked = ranked[
        ranked["user_id"].isin(
            actual_counts.index
        )
    ]

    total_customers = len(
        actual_counts
    )

    print(
        f"Customers evaluated: "
        f"{total_customers:,}"
    )

    results = []

    # --------------------------------------------------------
    # Evaluate each K
    # --------------------------------------------------------

    for k in K_VALUES:

        top_k = (
            ranked
            .groupby(
                "user_id",
                sort=False
            )
            .head(k)
            .copy()
        )

        # ----------------------------------------------------
        # Precision@K
        # ----------------------------------------------------

        hits = (
            top_k
            .groupby("user_id")["target"]
            .sum()
        )

        hits = hits.reindex(
            actual_counts.index,
            fill_value=0
        )

        precision_at_k = (
            hits / k
        ).mean()

        # ----------------------------------------------------
        # Recall@K
        # ----------------------------------------------------

        recall_at_k = (
            hits / actual_counts
        ).mean()

        # ----------------------------------------------------
        # Average Precision@K
        # ----------------------------------------------------

        ap_values = []

        for user_id, group in top_k.groupby(
            "user_id",
            sort=False
        ):

            relevant_count = int(
                actual_counts.loc[user_id]
            )

            if relevant_count == 0:
                continue

            hits_so_far = 0
            precision_sum = 0.0

            for rank, target in enumerate(
                group["target"].values,
                start=1
            ):

                if target == 1:

                    hits_so_far += 1

                    precision_at_rank = (
                        hits_so_far / rank
                    )

                    precision_sum += (
                        precision_at_rank
                    )

            # AP@K is normalized by
            # min(number of actual relevant items, K)

            denominator = min(
                relevant_count,
                k
            )

            if denominator > 0:

                ap = (
                    precision_sum
                    / denominator
                )

                ap_values.append(ap)

        map_at_k = (
            np.mean(ap_values)
            if ap_values
            else 0.0
        )

        # ----------------------------------------------------
        # NDCG@K
        # ----------------------------------------------------

        ndcg_values = []

        for user_id, group in top_k.groupby(
            "user_id",
            sort=False
        ):

            relevance = (
                group["target"]
                .values
                .astype(float)
            )

            # DCG
            discounts = np.log2(
                np.arange(
                    2,
                    len(relevance) + 2
                )
            )

            dcg = np.sum(
                relevance / discounts
            )

            # Ideal DCG
            relevant_count = int(
                actual_counts.loc[user_id]
            )

            ideal_length = min(
                relevant_count,
                k
            )

            if ideal_length == 0:
                continue

            ideal_relevance = np.ones(
                ideal_length
            )

            ideal_discounts = np.log2(
                np.arange(
                    2,
                    ideal_length + 2
                )
            )

            idcg = np.sum(
                ideal_relevance
                / ideal_discounts
            )

            if idcg > 0:

                ndcg = dcg / idcg

                ndcg_values.append(
                    ndcg
                )

        ndcg_at_k = (
            np.mean(ndcg_values)
            if ndcg_values
            else 0.0
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "model": model_name,

            "k": k,

            "precision_at_k":
                precision_at_k,

            "recall_at_k":
                recall_at_k,

            "map_at_k":
                map_at_k,

            "ndcg_at_k":
                ndcg_at_k,
        })

        print()
        print(
            f"K = {k}"
        )

        print(
            f"Precision@{k:<2} : "
            f"{precision_at_k:.4f}"
        )

        print(
            f"Recall@{k:<2}    : "
            f"{recall_at_k:.4f}"
        )

        print(
            f"MAP@{k:<2}        : "
            f"{map_at_k:.4f}"
        )

        print(
            f"NDCG@{k:<2}       : "
            f"{ndcg_at_k:.4f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# CONNECT
# ============================================================

log(
    "Connecting to DuckDB..."
)

con = duckdb.connect(
    DB_PATH
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

log(
    "Loading training data..."
)

training_query = f"""
SELECT
    d.user_id,
    d.product_id,
    {", ".join(
        [f"d.{feature}" for feature in FEATURES]
    )},
    d.target
FROM read_parquet(
    '{ML_DATASET}'
) AS d
WHERE MOD(
    ABS(HASH(d.user_id)),
    100
) >= 20
"""

train_df = con.execute(
    training_query
).fetchdf()

log(
    f"Training rows loaded: "
    f"{len(train_df):,}"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

log(
    "Loading validation data..."
)

validation_query = f"""
SELECT
    d.user_id,
    d.product_id,
    {", ".join(
        [f"d.{feature}" for feature in FEATURES]
    )},
    d.target
FROM read_parquet(
    '{ML_DATASET}'
) AS d
WHERE MOD(
    ABS(HASH(d.user_id)),
    100
) < 20
"""

validation_df = con.execute(
    validation_query
).fetchdf()

log(
    f"Validation rows loaded: "
    f"{len(validation_df):,}"
)


# ============================================================
# TRAINING SAMPLE
# ============================================================

positive_train = train_df[
    train_df["target"] == 1
].copy()

negative_train = train_df[
    train_df["target"] == 0
].copy()

negative_sample_size = min(
    len(negative_train),
    len(positive_train) * 5
)

log(
    f"Sampling "
    f"{negative_sample_size:,} "
    f"negative rows..."
)

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
).reset_index(
    drop=True
)


X_train = training_sample[
    FEATURES
]

y_train = training_sample[
    "target"
].astype(int)


X_validation = validation_df[
    FEATURES
]

y_validation = validation_df[
    "target"
].astype(int)


# ============================================================
# BASE VALIDATION DATA
# ============================================================

evaluation_df = validation_df[
    [
        "user_id",
        "product_id",
        "target"
    ]
].copy()


# ============================================================
# MODEL 1 — GLOBAL POPULARITY
# ============================================================

log(
    "Creating global popularity baseline..."
)

popularity_query = f"""
SELECT
    product_id,
    SUM(target) AS popularity
FROM read_parquet(
    '{ML_DATASET}'
)
WHERE MOD(
    ABS(HASH(user_id)),
    100
) >= 20
GROUP BY product_id
"""

# We use training-set target frequency as the
# popularity signal.

popularity_df = con.execute(
    popularity_query
).fetchdf()

evaluation_popularity = (
    evaluation_df
    .merge(
        popularity_df,
        on="product_id",
        how="left"
    )
)

evaluation_popularity[
    "popularity"
] = evaluation_popularity[
    "popularity"
].fillna(0)


popularity_results = calculate_ranking_metrics(
    evaluation_popularity,
    score_column="popularity",
    model_name="Global Popularity"
)


# ============================================================
# MODEL 2 — LOGISTIC REGRESSION
# ============================================================

log(
    "Training Logistic Regression..."
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
)

logistic_model = LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

log(
    "Generating Logistic Regression predictions..."
)

logistic_predictions = (
    logistic_model
    .predict_proba(
        X_validation_scaled
    )[:, 1]
)

evaluation_logistic = (
    evaluation_df.copy()
)

evaluation_logistic[
    "prediction"
] = logistic_predictions


logistic_results = calculate_ranking_metrics(
    evaluation_logistic,
    score_column="prediction",
    model_name="Logistic Regression"
)


# ============================================================
# MODEL 3 — TUNED XGBOOST
# ============================================================

log(
    "Training tuned XGBoost..."
)

xgb_model = XGBClassifier(

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

    random_state=RANDOM_STATE
)


xgb_model.fit(
    X_train,
    y_train
)

log(
    "Generating XGBoost predictions..."
)

xgb_predictions = (
    xgb_model
    .predict_proba(
        X_validation
    )[:, 1]
)

evaluation_xgb = (
    evaluation_df.copy()
)

evaluation_xgb[
    "prediction"
] = xgb_predictions


xgb_results = calculate_ranking_metrics(
    evaluation_xgb,
    score_column="prediction",
    model_name="XGBoost"
)


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = pd.concat(
    [
        popularity_results,
        logistic_results,
        xgb_results
    ],
    ignore_index=True
)


# ============================================================
# SAVE FULL RESULTS
# ============================================================

full_results_path = os.path.join(
    OUTPUT_DIR,
    "ranking_evaluation_results.csv"
)

all_results.to_csv(
    full_results_path,
    index=False
)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison = all_results.pivot(
    index="k",
    columns="model",
    values=[
        "precision_at_k",
        "recall_at_k",
        "map_at_k",
        "ndcg_at_k"
    ]
)


comparison_path = os.path.join(
    OUTPUT_DIR,
    "ranking_model_comparison.csv"
)

comparison.to_csv(
    comparison_path
)


# ============================================================
# FINAL DISPLAY
# ============================================================

print()
print("=" * 110)
print(
    "FINAL RANKING MODEL COMPARISON"
)
print("=" * 110)

print(
    all_results.to_string(
        index=False
    )
)

print("=" * 110)


# ============================================================
# BEST MODEL SUMMARY
# ============================================================

print()
print("=" * 80)
print(
    "MODEL PERFORMANCE SUMMARY"
)
print("=" * 80)

for model_name in [
    "Global Popularity",
    "Logistic Regression",
    "XGBoost"
]:

    model_results = all_results[
        all_results["model"] == model_name
    ]

    print()
    print(
        f"{model_name}"
    )

    for _, row in model_results.iterrows():

        k = int(row["k"])

        print(
            f"  K={k:<2} | "
            f"P={row['precision_at_k']:.4f} | "
            f"R={row['recall_at_k']:.4f} | "
            f"MAP={row['map_at_k']:.4f} | "
            f"NDCG={row['ndcg_at_k']:.4f}"
        )

print("=" * 80)


# ============================================================
# FILES
# ============================================================

print()
print(
    "Saved:"
)

print(
    f"  {full_results_path}"
)

print(
    f"  {comparison_path}"
)

print("=" * 80)


# ============================================================
# CLOSE
# ============================================================

con.close()

log(
    "Done."
)