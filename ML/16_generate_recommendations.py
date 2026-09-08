"""
Step 16: Generate production-style next-product recommendations.

Purpose:
Train the final tuned XGBoost recommendation model on the
historical train customers and generate personalized Top-20
recommendations for the test customers.

The test customers have historical "prior" orders but no known
target order, making them the appropriate inference population.

Candidate Strategy B:
1. Historical customer products
2. Global top 100 products
3. Top 15 products from each customer's top 3 departments

Output:
- recommendation_results.parquet
- recommendation_results.csv
"""

import os
import time

import duckdb
import pandas as pd
import numpy as np

from xgboost import XGBClassifier


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

ORDERS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "orders.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

PARQUET_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "recommendation_results.parquet"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "recommendation_results.csv"
)

RANDOM_STATE = 42

GLOBAL_TOP_N = 100
TOP_DEPARTMENTS = 3
TOP_PRODUCTS_PER_DEPARTMENT = 15
TOP_K = 20

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
# CONNECT
# ============================================================

log("Connecting to DuckDB...")

con = duckdb.connect(DB_PATH)


# ============================================================
# LOAD ORDERS
# ============================================================

log("Loading orders.csv...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ml_orders AS

    SELECT *

    FROM read_csv_auto(
        '{ORDERS_PATH}'
    )
""")


# ============================================================
# TEST CUSTOMERS
# ============================================================

log("Identifying test customers...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE test_customers AS

    SELECT
        user_id,
        order_id AS unknown_target_order_id

    FROM ml_orders

    WHERE eval_set = 'test'
""")


test_customer_count = con.execute("""
    SELECT COUNT(*)
    FROM test_customers
""").fetchone()[0]

log(
    f"Test customers: "
    f"{test_customer_count:,}"
)


# ============================================================
# TRAINING DATA
# ============================================================

log("Loading training data from final ML dataset...")

training_query = f"""
SELECT
    user_id,
    product_id,
    {", ".join(FEATURES)},
    target

FROM read_parquet(
    '{ML_DATASET}'
)

WHERE MOD(
    ABS(HASH(user_id)),
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
# NEGATIVE SAMPLING
# ============================================================

positive_train = train_df[
    train_df["target"] == 1
].copy()

negative_train = train_df[
    train_df["target"] == 0
].copy()

log(
    f"Positive training rows: "
    f"{len(positive_train):,}"
)

log(
    f"Available negative rows: "
    f"{len(negative_train):,}"
)

negative_sample_size = min(
    len(negative_train),
    len(positive_train) * 5
)

log(
    f"Sampling {negative_sample_size:,} "
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
).reset_index(drop=True)

X_train = training_sample[FEATURES]

y_train = (
    training_sample["target"]
    .astype(int)
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

log("Training final tuned XGBoost model...")

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

    n_jobs=-1,
    random_state=RANDOM_STATE
)

train_start = time.time()

model.fit(
    X_train,
    y_train
)

log(
    f"Final model trained in "
    f"{time.time() - train_start:.1f}s"
)


# ============================================================
# TEST PRIOR HISTORY
# ============================================================

log("Building test-customer prior history...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE test_prior_orders AS

    SELECT
        o.*

    FROM ml_orders o

    INNER JOIN test_customers tc
        ON o.user_id = tc.user_id

    WHERE o.eval_set = 'prior'
""")


# ============================================================
# CUSTOMER FEATURES
# ============================================================

log("Building customer features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE customer_features_test AS

    SELECT
        o.user_id,

        COUNT(DISTINCT o.order_id)
            AS previous_orders,

        AVG(order_item_counts.item_count)
            AS avg_basket_size,

        AVG(o.days_since_prior_order)
            AS avg_days_between_orders,

        AVG(
            CASE
                WHEN op.reordered = 1
                THEN 1.0
                ELSE 0.0
            END
        ) AS customer_reorder_rate

    FROM test_prior_orders o

    INNER JOIN (
        SELECT
            order_id,
            COUNT(*) AS item_count

        FROM order_products

        GROUP BY order_id

    ) order_item_counts

        ON o.order_id =
           order_item_counts.order_id

    INNER JOIN order_products op

        ON o.order_id =
           op.order_id

    GROUP BY o.user_id
""")


# ============================================================
# CUSTOMER × PRODUCT FEATURES
# ============================================================

log("Building customer-product features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE
    customer_product_features_test AS

    SELECT
        o.user_id,
        op.product_id,

        COUNT(*) AS previous_purchases,

        SUM(
            CASE
                WHEN op.reordered = 1
                THEN 1
                ELSE 0
            END
        ) AS previous_reorders,

        AVG(
            CASE
                WHEN op.reordered = 1
                THEN 1.0
                ELSE 0.0
            END
        ) AS customer_product_reorder_rate,

        MIN(o.order_number)
            AS first_purchase_order,

        MAX(o.order_number)
            AS last_purchase_order,

        AVG(op.add_to_cart_order)
            AS avg_cart_position

    FROM test_prior_orders o

    INNER JOIN order_products op

        ON o.order_id =
           op.order_id

    GROUP BY
        o.user_id,
        op.product_id
""")


# ============================================================
# PRODUCT FEATURES
# ============================================================

log("Building product features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE
    product_features_test AS

    SELECT
        op.product_id,

        COUNT(*)
            AS product_purchase_count,

        COUNT(DISTINCT o.user_id)
            AS product_unique_customers,

        AVG(
            CASE
                WHEN op.reordered = 1
                THEN 1.0
                ELSE 0.0
            END
        ) AS product_reorder_rate

    FROM ml_orders o

    INNER JOIN order_products op
        ON o.order_id =
           op.order_id

    WHERE o.eval_set = 'prior'

    GROUP BY op.product_id
""")


# ============================================================
# CUSTOMER × DEPARTMENT
# ============================================================

log("Building customer-department features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE
    customer_department_features_test AS

    SELECT
        o.user_id,
        p.department_id,

        COUNT(*)
            AS customer_department_purchases,

        COUNT(*) * 1.0
        /
        SUM(
            COUNT(*)
        ) OVER (
            PARTITION BY o.user_id
        )
        AS customer_department_share

    FROM test_prior_orders o

    INNER JOIN order_products op
        ON o.order_id =
           op.order_id

    INNER JOIN products p
        ON op.product_id =
           p.product_id

    GROUP BY
        o.user_id,
        p.department_id
""")


# ============================================================
# PRODUCT × DEPARTMENT
# ============================================================

log("Building product-department popularity...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE
    product_department_popularity_test AS

    SELECT
        p.product_id,
        p.department_id,

        COUNT(*) * 1.0
        /
        SUM(
            COUNT(*)
        ) OVER (
            PARTITION BY p.department_id
        )
        AS product_department_popularity

    FROM ml_orders o

    INNER JOIN order_products op
        ON o.order_id =
           op.order_id

    INNER JOIN products p
        ON op.product_id =
           p.product_id

    WHERE o.eval_set = 'prior'

    GROUP BY
        p.product_id,
        p.department_id
""")


# ============================================================
# CUSTOMER TOP DEPARTMENTS
# ============================================================

log(
    f"Finding top {TOP_DEPARTMENTS} "
    f"departments for each test customer..."
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE
    customer_top_departments_test AS

    SELECT
        user_id,
        department_id

    FROM (

        SELECT
            user_id,
            department_id,
            customer_department_purchases,

            ROW_NUMBER() OVER (
                PARTITION BY user_id

                ORDER BY
                    customer_department_purchases DESC,
                    department_id
            ) AS rn

        FROM customer_department_features_test

    )

    WHERE rn <= {TOP_DEPARTMENTS}
""")


# ============================================================
# GLOBAL TOP PRODUCTS
# ============================================================

log(
    f"Building global top "
    f"{GLOBAL_TOP_N} products..."
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE
    global_top_products_test AS

    SELECT
        product_id

    FROM (

        SELECT
            product_id,

            ROW_NUMBER() OVER (
                ORDER BY
                    product_purchase_count DESC,
                    product_id
            ) AS rn

        FROM product_features_test

    )

    WHERE rn <= {GLOBAL_TOP_N}
""")


# ============================================================
# DEPARTMENT TOP PRODUCTS
# ============================================================

log(
    f"Building top "
    f"{TOP_PRODUCTS_PER_DEPARTMENT} products "
    f"per department..."
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE
    department_top_products_test AS

    SELECT
        ctd.user_id,
        ranked_products.product_id

    FROM customer_top_departments_test ctd

    INNER JOIN (

        SELECT
            pf.product_id,
            p.department_id,
            pf.product_purchase_count,

            ROW_NUMBER() OVER (
                PARTITION BY p.department_id

                ORDER BY
                    pf.product_purchase_count DESC,
                    pf.product_id
            ) AS rn

        FROM product_features_test pf

        INNER JOIN products p
            ON pf.product_id =
               p.product_id

    ) ranked_products

        ON ctd.department_id =
           ranked_products.department_id

    WHERE ranked_products.rn
          <= {TOP_PRODUCTS_PER_DEPARTMENT}
""")


# ============================================================
# HISTORICAL CANDIDATES
# ============================================================

log("Building historical candidates...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE
    historical_candidates_test AS

    SELECT DISTINCT
        user_id,
        product_id

    FROM customer_product_features_test
""")


# ============================================================
# GLOBAL CANDIDATES
# ============================================================

log("Building global candidates...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE
    global_candidates_test AS

    SELECT
        tc.user_id,
        gtp.product_id

    FROM test_customers tc

    CROSS JOIN global_top_products_test gtp
""")


# ============================================================
# FINAL CANDIDATES
# ============================================================

log("Combining candidate sources...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE
    final_candidates_test AS

    SELECT
        user_id,
        product_id

    FROM historical_candidates_test

    UNION

    SELECT
        user_id,
        product_id

    FROM global_candidates_test

    UNION

    SELECT
        user_id,
        product_id

    FROM department_top_products_test
""")


candidate_count = con.execute("""
    SELECT COUNT(*)
    FROM final_candidates_test
""").fetchone()[0]

log(
    f"Test candidate rows: "
    f"{candidate_count:,}"
)


# ============================================================
# BUILD INFERENCE FEATURES
# ============================================================

log("Building inference feature dataset...")

inference_query = f"""

SELECT

    fc.user_id,
    fc.product_id,

    COALESCE(
        cpf.previous_purchases,
        0
    ) AS previous_purchases,

    COALESCE(
        cpf.previous_reorders,
        0
    ) AS previous_reorders,

    COALESCE(
        cpf.customer_product_reorder_rate,
        0.0
    ) AS customer_product_reorder_rate,

    COALESCE(
        cpf.first_purchase_order,
        0
    ) AS first_purchase_order,

    COALESCE(
        cpf.last_purchase_order,
        0
    ) AS last_purchase_order,

    COALESCE(
        cf.previous_orders -
        cpf.last_purchase_order,
        0
    ) AS purchase_recency,

    COALESCE(
        cpf.avg_cart_position,
        0.0
    ) AS avg_cart_position,

    COALESCE(
        cf.previous_orders,
        0
    ) AS previous_orders,

    COALESCE(
        cf.avg_basket_size,
        0.0
    ) AS avg_basket_size,

    COALESCE(
        cf.avg_days_between_orders,
        0.0
    ) AS avg_days_between_orders,

    COALESCE(
        cf.customer_reorder_rate,
        0.0
    ) AS customer_reorder_rate,

    COALESCE(
        pf.product_purchase_count,
        0
    ) AS product_purchase_count,

    COALESCE(
        pf.product_unique_customers,
        0
    ) AS product_unique_customers,

    COALESCE(
        pf.product_reorder_rate,
        0.0
    ) AS product_reorder_rate,

    COALESCE(
        cdf.customer_department_purchases,
        0
    ) AS customer_department_purchases,

    COALESCE(
        cdf.customer_department_share,
        0.0
    ) AS customer_department_share,

    COALESCE(
        pdp.product_department_popularity,
        0.0
    ) AS product_department_popularity

FROM final_candidates_test fc

INNER JOIN products p
    ON fc.product_id =
       p.product_id

LEFT JOIN customer_product_features_test cpf
    ON fc.user_id =
       cpf.user_id
    AND fc.product_id =
        cpf.product_id

LEFT JOIN customer_features_test cf
    ON fc.user_id =
       cf.user_id

LEFT JOIN product_features_test pf
    ON fc.product_id =
       pf.product_id

LEFT JOIN customer_department_features_test cdf
    ON fc.user_id =
       cdf.user_id
    AND p.department_id =
        cdf.department_id

LEFT JOIN product_department_popularity_test pdp
    ON fc.product_id =
       pdp.product_id
"""


# ============================================================
# LOAD INFERENCE DATA
# ============================================================

log("Loading inference data into pandas...")

inference_df = con.execute(
    inference_query
).fetchdf()

log(
    f"Inference rows loaded: "
    f"{len(inference_df):,}"
)


# ============================================================
# SANITY CHECK
# ============================================================

null_count = (
    inference_df[FEATURES]
    .isnull()
    .any(axis=1)
    .sum()
)

log(
    f"Rows with null features: "
    f"{null_count:,}"
)

if null_count > 0:
    raise ValueError(
        "Inference dataset contains NULL features."
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

log("Generating recommendation probabilities...")

prediction_start = time.time()

inference_df["predicted_probability"] = (
    model.predict_proba(
        inference_df[FEATURES]
    )[:, 1]
)

log(
    f"Predictions generated in "
    f"{time.time() - prediction_start:.1f}s"
)


# ============================================================
# RANK PRODUCTS PER CUSTOMER
# ============================================================

log(
    f"Ranking products and keeping Top {TOP_K}..."
)

inference_df = inference_df.sort_values(
    [
        "user_id",
        "predicted_probability",
        "product_id"
    ],
    ascending=[
        True,
        False,
        True
    ]
)

inference_df["recommendation_rank"] = (
    inference_df
    .groupby("user_id")
    .cumcount()
    + 1
)

recommendations = inference_df[
    inference_df["recommendation_rank"] <= TOP_K
].copy()


# ============================================================
# PRODUCT INFORMATION
# ============================================================

log("Adding product names and departments...")

product_info = con.execute("""
    SELECT
        p.product_id,
        p.product_name,
        p.department_id,
        d.department

    FROM products p

    LEFT JOIN (
        SELECT *
        FROM read_csv_auto(
            '""" + os.path.join(
                BASE_DIR,
                "data",
                "raw",
                "departments.csv"
            ) + """'
        )
    ) d

        ON p.department_id =
           d.department_id
""").fetchdf()

recommendations = recommendations.merge(
    product_info,
    on="product_id",
    how="left"
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

final_columns = [
    "user_id",
    "recommendation_rank",
    "product_id",
    "product_name",
    "department",
    "predicted_probability",

    "previous_purchases",
    "previous_reorders",
    "customer_product_reorder_rate",
    "purchase_recency",

    "previous_orders",
    "avg_basket_size",
    "customer_reorder_rate",

    "product_purchase_count",
    "product_unique_customers",
    "product_reorder_rate",

    "customer_department_purchases",
    "customer_department_share",
    "product_department_popularity",
]

recommendations = recommendations[
    final_columns
]


# ============================================================
# SAVE PARQUET
# ============================================================

log("Saving recommendation results to Parquet...")

recommendations.to_parquet(
    PARQUET_OUTPUT,
    index=False
)


# ============================================================
# SAVE CSV
# ============================================================

log("Saving recommendation results to CSV...")

recommendations.to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# SANITY CHECKS
# ============================================================

customer_count = (
    recommendations["user_id"]
    .nunique()
)

row_count = len(
    recommendations
)

average_probability = (
    recommendations[
        "predicted_probability"
    ].mean()
)

max_rank = (
    recommendations[
        "recommendation_rank"
    ].max()
)

missing_names = (
    recommendations[
        "product_name"
    ].isnull()
    .sum()
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("RECOMMENDATION GENERATION COMPLETE")
print("=" * 80)

print(
    f"Test customers:        "
    f"{test_customer_count:,}"
)

print(
    f"Candidate rows:        "
    f"{candidate_count:,}"
)

print(
    f"Recommendation rows:   "
    f"{row_count:,}"
)

print(
    f"Customers recommended: "
    f"{customer_count:,}"
)

print(
    f"Max recommendation rank: "
    f"{max_rank}"
)

print(
    f"Average probability:    "
    f"{average_probability:.4f}"
)

print(
    f"Missing product names:  "
    f"{missing_names:,}"
)

print()
print(
    f"Parquet output: "
    f"{PARQUET_OUTPUT}"
)

print(
    f"CSV output:     "
    f"{CSV_OUTPUT}"
)

print("=" * 80)


# ============================================================
# SAMPLE RECOMMENDATIONS
# ============================================================

print()
print("=" * 80)
print("SAMPLE RECOMMENDATIONS")
print("=" * 80)

sample_customer = (
    recommendations["user_id"]
    .iloc[0]
)

print(
    recommendations[
        recommendations["user_id"]
        == sample_customer
    ][
        [
            "user_id",
            "recommendation_rank",
            "product_name",
            "department",
            "predicted_probability",
            "previous_purchases",
            "customer_product_reorder_rate"
        ]
    ].to_string(index=False)
)

print("=" * 80)


# ============================================================
# CLOSE
# ============================================================

con.close()

log("Done.")