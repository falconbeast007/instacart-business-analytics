import duckdb
import pandas as pd
import numpy as np

from pathlib import Path
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "processed" / "instacart.duckdb"

ML_DATASET = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_final_dataset.parquet"
)

ORDERS_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "orders.csv"
)

RANDOM_SEED = 42

VALIDATION_FRACTION = 0.20

MAX_ITER = 200


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    # Customer × Product
    "previous_purchases",
    "previous_reorders",
    "customer_product_reorder_rate",
    "first_purchase_order",
    "last_purchase_order",
    "purchase_recency",
    "avg_cart_position",

    # Customer
    "previous_orders",
    "avg_basket_size",
    "avg_days_between_orders",
    "customer_reorder_rate",

    # Product
    "product_purchase_count",
    "product_unique_customers",
    "product_reorder_rate",

    # Customer × Department
    "customer_department_purchases",
    "customer_department_share",

    # Product × Department
    "product_department_popularity",
]


# ============================================================
# LOGGING
# ============================================================

START_TIME = time()


def log(message):
    elapsed = time() - START_TIME
    print(f"[{elapsed:8.1f}s] {message}")


# ============================================================
# CONNECT
# ============================================================

log("Connecting to DuckDB...")

con = duckdb.connect(str(DB_PATH))


# ============================================================
# DATASET CHECK
# ============================================================

log("Checking ML dataset...")

dataset_stats = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT user_id) AS customers,
        SUM(target) AS positives,
        SUM(
            CASE
                WHEN target = 0 THEN 1
                ELSE 0
            END
        ) AS negatives
    FROM read_parquet(
        '{ML_DATASET.as_posix()}'
    )
""").fetchone()


print()
print("=" * 70)
print("ML DATASET")
print("=" * 70)

print(f"Rows:                 {dataset_stats[0]:,}")
print(f"Customers:            {dataset_stats[1]:,}")
print(f"Positive:             {dataset_stats[2]:,}")
print(f"Negative:             {dataset_stats[3]:,}")

print("=" * 70)


# ============================================================
# DETERMINISTIC CUSTOMER-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

log("Creating deterministic customer-level split...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE customer_split AS

    SELECT
        user_id,

        CASE
            WHEN
                MOD(
                    ABS(
                        HASH(user_id)
                    ),
                    100
                ) < 20
            THEN 'validation'

            ELSE 'train'
        END AS split

    FROM (
        SELECT DISTINCT
            user_id

        FROM read_parquet(
            ?
        )
    )
""", [str(ML_DATASET)])


split_stats = con.execute("""
    SELECT
        split,
        COUNT(*) AS customers

    FROM customer_split

    GROUP BY split

    ORDER BY split
""").fetchall()


print()
print("=" * 70)
print("CUSTOMER SPLIT")
print("=" * 70)

for split, count in split_stats:

    print(
        f"{split.capitalize():20s}: "
        f"{count:,}"
    )

print("=" * 70)


# ============================================================
# CREATE TRAIN VIEW
# ============================================================

log("Creating training view...")

feature_sql = ", ".join(
    f"d.{feature}"
    for feature in FEATURES
)

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW ml_train AS

    SELECT
        d.user_id,
        d.product_id,

        {feature_sql},

        d.target

    FROM read_parquet(
        '{ML_DATASET.as_posix()}'
    ) d

    INNER JOIN customer_split s
        ON d.user_id = s.user_id

    WHERE s.split = 'train'
""")


# ============================================================
# CREATE VALIDATION VIEW
# ============================================================

log("Creating validation view...")

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW ml_validation AS

    SELECT
        d.user_id,
        d.product_id,

        {feature_sql},

        d.target

    FROM read_parquet(
        '{ML_DATASET.as_posix()}'
    ) d

    INNER JOIN customer_split s
        ON d.user_id = s.user_id

    WHERE s.split = 'validation'
""")


# ============================================================
# LOAD TRAINING DATA
# ============================================================

log("Loading training data...")

train_query = f"""
    SELECT
        {", ".join(FEATURES)},
        target

    FROM ml_train
"""

train_df = con.execute(
    train_query
).fetchdf()


log(
    f"Training rows loaded: "
    f"{len(train_df):,}"
)


X_train = train_df[
    FEATURES
].astype("float32")

y_train = train_df[
    "target"
].astype("int8")


del train_df


# ============================================================
# TRAINING STATISTICS
# ============================================================

print()
print("=" * 70)
print("TRAINING SET")
print("=" * 70)

print(
    f"Rows:                 "
    f"{len(X_train):,}"
)

print(
    f"Positive:             "
    f"{int(y_train.sum()):,}"
)

print(
    f"Negative:             "
    f"{int((y_train == 0).sum()):,}"
)

print(
    f"Positive rate:        "
    f"{y_train.mean() * 100:.2f}%"
)

print("=" * 70)


# ============================================================
# SCALE FEATURES
# ============================================================

log("Scaling numerical features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

del X_train


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

log("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=MAX_ITER,

    # Important because only ~4% of candidates
    # are positive.
    class_weight="balanced",

    solver="lbfgs",

    random_state=RANDOM_SEED
)

model.fit(
    X_train_scaled,
    y_train
)

log(
    "Logistic Regression training complete."
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({

    "feature": FEATURES,

    "coefficient": model.coef_[0]

})

coefficients[
    "abs_coefficient"
] = coefficients[
    "coefficient"
].abs()


coefficients = coefficients.sort_values(
    "abs_coefficient",
    ascending=False
)


print()
print("=" * 70)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 70)

print(
    coefficients[
        [
            "feature",
            "coefficient"
        ]
    ].to_string(
        index=False
    )
)

print("=" * 70)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

log("Loading validation data...")

validation_query = f"""
    SELECT

        user_id,
        product_id,

        {", ".join(FEATURES)},

        target

    FROM ml_validation
"""

validation_df = con.execute(
    validation_query
).fetchdf()


log(
    f"Validation rows loaded: "
    f"{len(validation_df):,}"
)


X_valid = validation_df[
    FEATURES
].astype("float32")

y_valid = validation_df[
    "target"
].astype("int8")


# ============================================================
# SCALE VALIDATION DATA
# ============================================================

log(
    "Applying training scaler "
    "to validation data..."
)

X_valid_scaled = scaler.transform(
    X_valid
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

log("Generating validation predictions...")

y_probability = model.predict_proba(
    X_valid_scaled
)[:, 1]


# Classification threshold
THRESHOLD = 0.50

y_prediction = (
    y_probability >= THRESHOLD
).astype(int)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

precision = precision_score(
    y_valid,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_valid,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_valid,
    y_prediction,
    zero_division=0
)

pr_auc = average_precision_score(
    y_valid,
    y_probability
)


print()
print("=" * 70)
print(
    "LOGISTIC REGRESSION — "
    "CLASSIFICATION RESULTS"
)
print("=" * 70)

print(
    f"PR-AUC:               "
    f"{pr_auc:.4f}"
)

print(
    f"Precision:             "
    f"{precision:.4f}"
)

print(
    f"Recall:                "
    f"{recall:.4f}"
)

print(
    f"F1 Score:              "
    f"{f1:.4f}"
)

print("=" * 70)


# ============================================================
# RANKING EVALUATION FUNCTION
# ============================================================

def evaluate_at_k(
    df,
    k
):

    # Rank candidate products for each customer
    ranked = (
        df
        .sort_values(
            [
                "user_id",
                "probability"
            ],
            ascending=[
                True,
                False
            ]
        )
        .groupby(
            "user_id",
            sort=False
        )
        .head(k)
    )


    # Number of actual purchased products
    actual_counts = (
        df
        .groupby(
            "user_id"
        )["target"]
        .sum()
    )


    # Number of correct recommendations
    hits = (
        ranked
        .groupby(
            "user_id"
        )["target"]
        .sum()
    )


    hits = hits.reindex(
        actual_counts.index,
        fill_value=0
    )


    # Precision@K
    precision_at_k = (
        hits / k
    ).mean()


    # Recall@K
    recall_at_k = (
        hits
        /
        actual_counts.clip(
            lower=1
        )
    ).mean()


    return (
        precision_at_k,
        recall_at_k
    )


# ============================================================
# PREPARE RANKING DATA
# ============================================================

log(
    "Preparing ranking evaluation..."
)

validation_for_ranking = validation_df[
    [
        "user_id",
        "product_id",
        "target"
    ]
].copy()


validation_for_ranking[
    "probability"
] = y_probability


# ============================================================
# LOGISTIC REGRESSION RANKING
# ============================================================

log(
    "Evaluating recommendation ranking..."
)


ranking_results = []


for k in [5, 10, 20]:

    precision_k, recall_k = (
        evaluate_at_k(
            validation_for_ranking,
            k
        )
    )

    ranking_results.append({

        "K": k,

        "Precision@K":
            precision_k,

        "Recall@K":
            recall_k
    })


ranking_df = pd.DataFrame(
    ranking_results
)


print()
print("=" * 70)
print(
    "LOGISTIC REGRESSION — "
    "RANKING RESULTS"
)
print("=" * 70)

print(
    ranking_df.to_string(
        index=False,
        formatters={
            "Precision@K":
                "{:.4f}".format,

            "Recall@K":
                "{:.4f}".format
        }
    )
)

print("=" * 70)


# ============================================================
# GLOBAL POPULARITY BASELINE
# ============================================================

log(
    "Evaluating global popularity baseline..."
)


# ------------------------------------------------------------
# Load orders because this is a separate DuckDB connection
# context from script 11.
# ------------------------------------------------------------

log(
    "Loading orders.csv for popularity baseline..."
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ml_orders AS

    SELECT *

    FROM read_csv_auto(
        '{ORDERS_PATH.as_posix()}'
    )
""")


# ------------------------------------------------------------
# Calculate global popularity from PRIOR orders only
# ------------------------------------------------------------

con.execute("""
    CREATE OR REPLACE TEMP TABLE global_product_scores AS

    SELECT

        op.product_id,

        COUNT(*) AS popularity

    FROM order_products op

    INNER JOIN ml_orders o

        ON op.order_id = o.order_id

    WHERE o.eval_set = 'prior'

    GROUP BY op.product_id
""")


# ------------------------------------------------------------
# Get popularity scores
# ------------------------------------------------------------

popularity_df = con.execute("""
    SELECT
        product_id,
        popularity

    FROM global_product_scores
""").fetchdf()


# ------------------------------------------------------------
# Attach popularity to validation candidates
# ------------------------------------------------------------

baseline_df = validation_df[
    [
        "user_id",
        "product_id",
        "target"
    ]
].copy()


baseline_df = baseline_df.merge(
    popularity_df,
    on="product_id",
    how="left"
)


baseline_df[
    "popularity"
] = baseline_df[
    "popularity"
].fillna(0)


# ============================================================
# GLOBAL POPULARITY RANKING
# ============================================================

baseline_results = []


for k in [5, 10, 20]:

    ranked = (

        baseline_df

        .sort_values(
            [
                "user_id",
                "popularity"
            ],

            ascending=[
                True,
                False
            ]
        )

        .groupby(
            "user_id",
            sort=False
        )

        .head(k)
    )


    actual_counts = (
        baseline_df
        .groupby(
            "user_id"
        )["target"]
        .sum()
    )


    hits = (
        ranked
        .groupby(
            "user_id"
        )["target"]
        .sum()
        .reindex(
            actual_counts.index,
            fill_value=0
        )
    )


    precision_k = (
        hits / k
    ).mean()


    recall_k = (
        hits
        /
        actual_counts.clip(
            lower=1
        )
    ).mean()


    baseline_results.append({

        "K": k,

        "Precision@K":
            precision_k,

        "Recall@K":
            recall_k
    })


baseline_results_df = pd.DataFrame(
    baseline_results
)


print()
print("=" * 70)
print(
    "GLOBAL POPULARITY BASELINE"
)
print("=" * 70)

print(
    baseline_results_df.to_string(
        index=False,
        formatters={
            "Precision@K":
                "{:.4f}".format,

            "Recall@K":
                "{:.4f}".format
        }
    )
)

print("=" * 70)


# ============================================================
# MODEL VS BASELINE
# ============================================================

comparison_df = ranking_df.merge(
    baseline_results_df,
    on="K",
    suffixes=(
        "_LogisticRegression",
        "_Popularity"
    )
)


comparison_df[
    "Precision_Improvement"
] = (
    comparison_df[
        "Precision@K_LogisticRegression"
    ]
    -
    comparison_df[
        "Precision@K_Popularity"
    ]
)


comparison_df[
    "Recall_Improvement"
] = (
    comparison_df[
        "Recall@K_LogisticRegression"
    ]
    -
    comparison_df[
        "Recall@K_Popularity"
    ]
)


print()
print("=" * 70)
print(
    "MODEL VS GLOBAL POPULARITY"
)
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "Precision@K_LogisticRegression":
                "{:.4f}".format,

            "Recall@K_LogisticRegression":
                "{:.4f}".format,

            "Precision@K_Popularity":
                "{:.4f}".format,

            "Recall@K_Popularity":
                "{:.4f}".format,

            "Precision_Improvement":
                "{:.4f}".format,

            "Recall_Improvement":
                "{:.4f}".format
        }
    )
)

print("=" * 70)


# ============================================================
# SAVE COEFFICIENTS
# ============================================================

coefficients_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "logistic_regression_coefficients.csv"
)


coefficients[
    [
        "feature",
        "coefficient"
    ]
].to_csv(
    coefficients_path,
    index=False
)


# ============================================================
# SAVE CLASSIFICATION RESULTS
# ============================================================

classification_results_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "logistic_regression_results.csv"
)


classification_results = pd.DataFrame([{

    "model":
        "Logistic Regression",

    "PR_AUC":
        pr_auc,

    "Precision":
        precision,

    "Recall":
        recall,

    "F1":
        f1

}])


classification_results.to_csv(
    classification_results_path,
    index=False
)


# ============================================================
# SAVE RANKING RESULTS
# ============================================================

ranking_results_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "logistic_regression_ranking_results.csv"
)


ranking_df.to_csv(
    ranking_results_path,
    index=False
)


# ============================================================
# SAVE BASELINE RESULTS
# ============================================================

baseline_results_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "global_popularity_baseline_results.csv"
)


baseline_results_df.to_csv(
    baseline_results_path,
    index=False
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_vs_popularity_comparison.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

validation_customer_count = (
    validation_df[
        "user_id"
    ].nunique()
)


print()
print("=" * 70)
print("MODEL 1 COMPLETE")
print("=" * 70)

print(
    f"Model:                "
    f"Logistic Regression"
)

print(
    f"Validation customers: "
    f"{validation_customer_count:,}"
)

print(
    f"Validation rows:      "
    f"{len(validation_df):,}"
)

print(
    f"PR-AUC:               "
    f"{pr_auc:.4f}"
)

print(
    f"Precision:             "
    f"{precision:.4f}"
)

print(
    f"Recall:                "
    f"{recall:.4f}"
)

print(
    f"F1:                    "
    f"{f1:.4f}"
)

print()
print("Ranking Results:")

print(
    ranking_df.to_string(
        index=False,
        formatters={
            "Precision@K":
                "{:.4f}".format,

            "Recall@K":
                "{:.4f}".format
        }
    )
)

print()
print(
    f"Coefficients saved:    "
    f"{coefficients_path}"
)

print(
    f"Classification saved:  "
    f"{classification_results_path}"
)

print(
    f"Ranking saved:         "
    f"{ranking_results_path}"
)

print(
    f"Baseline saved:        "
    f"{baseline_results_path}"
)

print(
    f"Comparison saved:      "
    f"{comparison_path}"
)

print("=" * 70)


# ============================================================
# CLOSE
# ============================================================

con.close()

log("Done.")