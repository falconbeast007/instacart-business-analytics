import duckdb
from pathlib import Path
import time


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "processed" / "instacart.duckdb"
ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv"

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_final_dataset.parquet"
)

# Final Candidate Strategy B
GLOBAL_TOP_N = 100
TOP_DEPARTMENTS = 3
TOP_PRODUCTS_PER_DEPARTMENT = 15


# ============================================================
# LOGGING
# ============================================================

START_TIME = time.time()


def log(message):
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:7.1f}s] {message}")


# ============================================================
# CONNECT TO DUCKDB
# ============================================================

log("Connecting to DuckDB...")

con = duckdb.connect(str(DB_PATH))


# ============================================================
# LOAD ORDERS WITH EVAL_SET
# ============================================================

log("Loading orders.csv with eval_set...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ml_orders AS
    SELECT *
    FROM read_csv_auto('{ORDERS_PATH.as_posix()}')
""")


# ============================================================
# IDENTIFY TRAIN CUSTOMERS
# ============================================================

log("Identifying train customers...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE train_customers AS
    SELECT
        user_id,
        order_id AS target_order_id
    FROM ml_orders
    WHERE eval_set = 'train'
""")

train_customer_count = con.execute("""
    SELECT COUNT(*)
    FROM train_customers
""").fetchone()[0]

log(f"Train customers: {train_customer_count:,}")


# ============================================================
# PRIOR ORDERS
# ============================================================

log("Building prior-order history...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE train_prior_orders AS
    SELECT
        o.*
    FROM ml_orders o

    INNER JOIN train_customers tc
        ON o.user_id = tc.user_id

    WHERE o.eval_set = 'prior'
""")


# ============================================================
# CUSTOMER FEATURES
# ============================================================

log("Building customer features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE customer_features AS

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
                WHEN op.reordered = 1 THEN 1.0
                ELSE 0.0
            END
        ) AS customer_reorder_rate

    FROM train_prior_orders o

    INNER JOIN (
        SELECT
            order_id,
            COUNT(*) AS item_count

        FROM order_products

        GROUP BY order_id
    ) order_item_counts

        ON o.order_id = order_item_counts.order_id

    INNER JOIN order_products op

        ON o.order_id = op.order_id

    GROUP BY o.user_id
""")


# ============================================================
# CUSTOMER × PRODUCT FEATURES
# ============================================================

log("Building customer-product features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE customer_product_features AS

    SELECT
        o.user_id,
        op.product_id,

        COUNT(*) AS previous_purchases,

        SUM(
            CASE
                WHEN op.reordered = 1 THEN 1
                ELSE 0
            END
        ) AS previous_reorders,

        AVG(
            CASE
                WHEN op.reordered = 1 THEN 1.0
                ELSE 0.0
            END
        ) AS customer_product_reorder_rate,

        MIN(o.order_number)
            AS first_purchase_order,

        MAX(o.order_number)
            AS last_purchase_order,

        AVG(op.add_to_cart_order)
            AS avg_cart_position

    FROM train_prior_orders o

    INNER JOIN order_products op
        ON o.order_id = op.order_id

    GROUP BY
        o.user_id,
        op.product_id
""")


# ============================================================
# PRODUCT FEATURES
# ============================================================

log("Building product features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE product_features_ml AS

    SELECT
        op.product_id,

        COUNT(*)
            AS product_purchase_count,

        COUNT(DISTINCT o.user_id)
            AS product_unique_customers,

        AVG(
            CASE
                WHEN op.reordered = 1 THEN 1.0
                ELSE 0.0
            END
        ) AS product_reorder_rate

    FROM train_prior_orders o

    INNER JOIN order_products op
        ON o.order_id = op.order_id

    GROUP BY op.product_id
""")


# ============================================================
# CUSTOMER × DEPARTMENT FEATURES
# ============================================================

log("Building customer-department features...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE customer_department_features AS

    SELECT
        o.user_id,
        p.department_id,

        COUNT(*) AS customer_department_purchases,

        COUNT(*) * 1.0
        /
        SUM(COUNT(*)) OVER (
            PARTITION BY o.user_id
        )
        AS customer_department_share

    FROM train_prior_orders o

    INNER JOIN order_products op
        ON o.order_id = op.order_id

    INNER JOIN products p
        ON op.product_id = p.product_id

    GROUP BY
        o.user_id,
        p.department_id
""")


# ============================================================
# PRODUCT × DEPARTMENT POPULARITY
# ============================================================

log("Building product-department popularity...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE product_department_popularity AS

    SELECT
        p.product_id,
        p.department_id,

        COUNT(*) * 1.0
        /
        SUM(COUNT(*)) OVER (
            PARTITION BY p.department_id
        )
        AS product_department_popularity

    FROM train_prior_orders o

    INNER JOIN order_products op
        ON o.order_id = op.order_id

    INNER JOIN products p
        ON op.product_id = p.product_id

    GROUP BY
        p.product_id,
        p.department_id
""")


# ============================================================
# CUSTOMER TOP 3 DEPARTMENTS
# ============================================================

log(
    f"Identifying each customer's top "
    f"{TOP_DEPARTMENTS} departments..."
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE customer_top_departments AS

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

        FROM customer_department_features
    )

    WHERE rn <= {TOP_DEPARTMENTS}
""")


# ============================================================
# GLOBAL TOP 100 PRODUCTS
# ============================================================

log(
    f"Building global top "
    f"{GLOBAL_TOP_N} products..."
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE global_top_products AS

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

        FROM product_features_ml
    )

    WHERE rn <= {GLOBAL_TOP_N}
""")


# ============================================================
# TOP PRODUCTS PER CUSTOMER'S TOP DEPARTMENTS
# ============================================================

log(
    f"Building top {TOP_PRODUCTS_PER_DEPARTMENT} products "
    f"for each customer's top {TOP_DEPARTMENTS} departments..."
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE department_top_products AS

    SELECT
        ctd.user_id,
        ranked_products.product_id

    FROM customer_top_departments ctd

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

        FROM product_features_ml pf

        INNER JOIN products p
            ON pf.product_id = p.product_id

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
    CREATE OR REPLACE TEMP TABLE historical_candidates AS

    SELECT DISTINCT
        user_id,
        product_id

    FROM customer_product_features
""")


# ============================================================
# GLOBAL CANDIDATES
# ============================================================

log("Building global candidates...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE global_candidates AS

    SELECT
        tc.user_id,
        gtp.product_id

    FROM train_customers tc

    CROSS JOIN global_top_products gtp
""")


# ============================================================
# FINAL STRATEGY B CANDIDATES
# ============================================================

log("Combining candidate sources...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE final_candidates AS

    SELECT
        user_id,
        product_id

    FROM historical_candidates

    UNION

    SELECT
        user_id,
        product_id

    FROM global_candidates

    UNION

    SELECT
        user_id,
        product_id

    FROM department_top_products
""")


candidate_count = con.execute("""
    SELECT COUNT(*)
    FROM final_candidates
""").fetchone()[0]

log(
    f"Final candidate rows: "
    f"{candidate_count:,}"
)


# ============================================================
# TARGET PRODUCTS
# ============================================================

log("Building target labels...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE target_products AS

    SELECT DISTINCT
        tc.user_id,
        op.product_id

    FROM train_customers tc

    INNER JOIN order_products op
        ON tc.target_order_id = op.order_id
""")


# ============================================================
# FINAL ML DATASET
# ============================================================

log("Joining features and target...")

query = """
    SELECT

        fc.user_id,
        fc.product_id,


        -- ====================================================
        -- CUSTOMER × PRODUCT
        -- ====================================================

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
            cf.previous_orders
            - cpf.last_purchase_order,
            0
        ) AS purchase_recency,

        COALESCE(
            cpf.avg_cart_position,
            0.0
        ) AS avg_cart_position,


        -- ====================================================
        -- CUSTOMER
        -- ====================================================

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


        -- ====================================================
        -- PRODUCT
        -- ====================================================

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


        -- ====================================================
        -- CUSTOMER × DEPARTMENT
        -- ====================================================

        COALESCE(
            cdf.customer_department_purchases,
            0
        ) AS customer_department_purchases,

        COALESCE(
            cdf.customer_department_share,
            0.0
        ) AS customer_department_share,


        -- ====================================================
        -- PRODUCT × DEPARTMENT
        -- ====================================================

        COALESCE(
            pdp.product_department_popularity,
            0.0
        ) AS product_department_popularity,


        -- ====================================================
        -- TARGET
        -- ====================================================

        CASE
            WHEN tp.product_id IS NOT NULL
            THEN 1
            ELSE 0
        END AS target


    FROM final_candidates fc


    INNER JOIN products p
        ON fc.product_id = p.product_id


    LEFT JOIN customer_product_features cpf
        ON fc.user_id = cpf.user_id
        AND fc.product_id = cpf.product_id


    LEFT JOIN customer_features cf
        ON fc.user_id = cf.user_id


    LEFT JOIN product_features_ml pf
        ON fc.product_id = pf.product_id


    LEFT JOIN customer_department_features cdf
        ON fc.user_id = cdf.user_id
        AND p.department_id = cdf.department_id


    LEFT JOIN product_department_popularity pdp
        ON fc.product_id = pdp.product_id


    LEFT JOIN target_products tp
        ON fc.user_id = tp.user_id
        AND fc.product_id = tp.product_id
"""


# ============================================================
# WRITE PARQUET
# ============================================================

log("Writing final ML dataset to Parquet...")

con.execute(f"""
    COPY (
        {query}
    )
    TO '{OUTPUT_PATH.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")


# ============================================================
# SANITY CHECKS
# ============================================================

log("Running sanity checks...")


stats = con.execute(f"""
    SELECT

        COUNT(*) AS rows,

        SUM(target) AS positives,

        SUM(
            CASE
                WHEN target = 0
                THEN 1
                ELSE 0
            END
        ) AS negatives,

        AVG(target) AS positive_rate,

        COUNT(DISTINCT user_id) AS customers,

        COUNT(DISTINCT product_id) AS products

    FROM read_parquet(
        '{OUTPUT_PATH.as_posix()}'
    )
""").fetchone()


null_check = con.execute(f"""
    SELECT COUNT(*)

    FROM read_parquet(
        '{OUTPUT_PATH.as_posix()}'
    )

    WHERE

        previous_purchases IS NULL
        OR previous_reorders IS NULL
        OR customer_product_reorder_rate IS NULL
        OR first_purchase_order IS NULL
        OR last_purchase_order IS NULL
        OR purchase_recency IS NULL
        OR avg_cart_position IS NULL

        OR previous_orders IS NULL
        OR avg_basket_size IS NULL
        OR avg_days_between_orders IS NULL
        OR customer_reorder_rate IS NULL

        OR product_purchase_count IS NULL
        OR product_unique_customers IS NULL
        OR product_reorder_rate IS NULL

        OR customer_department_purchases IS NULL
        OR customer_department_share IS NULL

        OR product_department_popularity IS NULL

""").fetchone()[0]


duplicate_check = con.execute(f"""
    SELECT
        COUNT(*) -
        COUNT(
            DISTINCT
            CAST(user_id AS VARCHAR)
            || '-'
            || CAST(product_id AS VARCHAR)
        )

    FROM read_parquet(
        '{OUTPUT_PATH.as_posix()}'
    )
""").fetchone()[0]


target_values = con.execute(f"""
    SELECT

        MIN(target),

        MAX(target),

        COUNT(DISTINCT target)

    FROM read_parquet(
        '{OUTPUT_PATH.as_posix()}'
    )
""").fetchone()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FINAL ML DATASET COMPLETE")
print("=" * 70)

print(f"Output:              {OUTPUT_PATH}")
print(f"Rows:                {stats[0]:,}")
print(f"Positive:            {stats[1]:,}")
print(f"Negative:            {stats[2]:,}")
print(f"Positive rate:       {stats[3] * 100:.2f}%")
print(f"Customers:           {stats[4]:,}")
print(f"Products:            {stats[5]:,}")
print(f"Null feature rows:   {null_check:,}")
print(f"Duplicate pairs:     {duplicate_check:,}")
print(f"Target min:          {target_values[0]}")
print(f"Target max:          {target_values[1]}")
print(f"Distinct targets:    {target_values[2]}")

print("=" * 70)


# ============================================================
# CLOSE
# ============================================================

con.close()

log("Done.")