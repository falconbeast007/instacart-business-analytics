import duckdb
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "instacart.duckdb"
)

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
)

TABLEAU_DIR = (
    BASE_DIR
    / "tableau"
    / "data"
)

TABLEAU_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("PREPARING TABLEAU DATA")
print("=" * 80)

print(f"\nDatabase:")
print(DB_PATH)

print(f"\nTableau output directory:")
print(TABLEAU_DIR)


# ============================================================
# CONNECT TO DUCKDB
# ============================================================

con = duckdb.connect(
    str(DB_PATH)
)


# ============================================================
# VALIDATE CORE TABLES
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING DUCKDB TABLES")
print("=" * 80)

required_tables = [
    "orders",
    "products",
    "order_products",
    "customers",
    "product_features"
]

existing_tables = con.execute(
    """
    SHOW TABLES
    """
).fetchdf()

existing_table_names = set(
    existing_tables["name"].tolist()
)

for table in required_tables:

    if table not in existing_table_names:

        raise RuntimeError(
            f"Required DuckDB table '{table}' does not exist."
        )

    print(f"✓ {table}")


# ============================================================
# LOAD REFERENCE CSVs AS TEMPORARY VIEWS
# ============================================================

print("\n" + "=" * 80)
print("LOADING REFERENCE DATA")
print("=" * 80)

departments_path = (
    RAW_DIR
    / "departments.csv"
)

aisles_path = (
    RAW_DIR
    / "aisles.csv"
)


if not departments_path.exists():

    raise FileNotFoundError(
        f"Missing file: {departments_path}"
    )


if not aisles_path.exists():

    raise FileNotFoundError(
        f"Missing file: {aisles_path}"
    )


# Convert paths to strings that DuckDB can read.
departments_path_sql = str(
    departments_path
).replace("'", "''")

aisles_path_sql = str(
    aisles_path
).replace("'", "''")


con.execute(
    f"""
    CREATE OR REPLACE TEMP VIEW departments AS

    SELECT *

    FROM read_csv_auto(
        '{departments_path_sql}'
    )
    """
)


con.execute(
    f"""
    CREATE OR REPLACE TEMP VIEW aisles AS

    SELECT *

    FROM read_csv_auto(
        '{aisles_path_sql}'
    )
    """
)


print("✓ departments")
print("✓ aisles")


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_query(
    query,
    filename
):

    print(
        f"\nCreating {filename}..."
    )

    df = con.execute(
        query
    ).fetchdf()

    output_path = (
        TABLEAU_DIR
        / filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Saved: {output_path}"
    )

    return df


# ============================================================
# 1. EXECUTIVE KPIs
# ============================================================

print("\n" + "=" * 80)
print("1. EXECUTIVE KPIs")
print("=" * 80)


executive_kpis = """

SELECT

    COUNT(
        DISTINCT o.user_id
    ) AS total_customers,

    COUNT(
        DISTINCT o.order_id
    ) AS total_orders,

    COUNT(*) AS total_items,

    ROUND(
        COUNT(*) * 1.0
        /
        COUNT(
            DISTINCT o.order_id
        ),
        2
    ) AS avg_basket_size,

    ROUND(
        AVG(op.reordered) * 100,
        2
    ) AS overall_reorder_rate

FROM orders o

JOIN order_products op

    ON o.order_id = op.order_id

"""


save_query(
    executive_kpis,
    "tableau_executive_kpis.csv"
)


# ============================================================
# 2. CUSTOMER ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("2. CUSTOMER ANALYSIS")
print("=" * 80)


customer_analysis = """

SELECT

    c.user_id,

    c.total_orders,

    c.total_items,

    ROUND(
        c.avg_items_per_order,
        2
    ) AS avg_basket_size,

    ROUND(
        c.reorder_rate * 100,
        2
    ) AS reorder_rate_percent,

    ROUND(
        c.avg_days_between_orders,
        2
    ) AS avg_days_between_orders,

    c.max_order_number,

    CASE

        WHEN c.total_orders <= 5
            THEN '1-5 Orders'

        WHEN c.total_orders <= 10
            THEN '6-10 Orders'

        WHEN c.total_orders <= 20
            THEN '11-20 Orders'

        WHEN c.total_orders <= 30
            THEN '21-30 Orders'

        ELSE '31+ Orders'

    END AS order_frequency_band

FROM customers c

"""


save_query(
    customer_analysis,
    "tableau_customer_analysis.csv"
)


# ============================================================
# 3. CUSTOMER SEGMENTS
# ============================================================

print("\n" + "=" * 80)
print("3. CUSTOMER SEGMENTS")
print("=" * 80)


customer_segments = """

SELECT

    CASE

        WHEN total_orders <= 5
             AND reorder_rate < 0.30

            THEN 'Occasional'

        WHEN total_orders <= 15
             AND reorder_rate >= 0.50

            THEN 'Repeat-Oriented'

        WHEN total_orders >= 30
             AND reorder_rate >= 0.65

            THEN 'Highly Loyal'

        WHEN total_orders >= 20

            THEN 'Frequent'

        ELSE 'Regular'

    END AS customer_segment,

    COUNT(*) AS customers,

    ROUND(
        AVG(total_orders),
        2
    ) AS avg_orders,

    ROUND(
        AVG(avg_items_per_order),
        2
    ) AS avg_basket_size,

    ROUND(
        AVG(reorder_rate) * 100,
        2
    ) AS avg_reorder_rate_percent

FROM customers

GROUP BY 1

ORDER BY customers DESC

"""


save_query(
    customer_segments,
    "tableau_customer_segments.csv"
)


# ============================================================
# 4. DEPARTMENT ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("4. DEPARTMENT ANALYSIS")
print("=" * 80)


department_analysis = """

SELECT

    d.department,

    COUNT(*) AS total_purchases,

    COUNT(
        DISTINCT o.order_id
    ) AS unique_orders,

    COUNT(
        DISTINCT o.user_id
    ) AS unique_customers,

    ROUND(
        COUNT(*) * 100.0
        /
        SUM(
            COUNT(*)
        ) OVER (),
        2
    ) AS purchase_share_percent,

    ROUND(
        COUNT(
            DISTINCT o.user_id
        ) * 100.0
        /
        (
            SELECT COUNT(
                DISTINCT user_id
            )
            FROM orders
        ),
        2
    ) AS customer_penetration_percent,

    ROUND(
        AVG(op.reordered) * 100,
        2
    ) AS reorder_rate_percent,

    ROUND(
        COUNT(*) * 1.0
        /
        COUNT(
            DISTINCT o.user_id
        ),
        2
    ) AS purchases_per_customer

FROM orders o

JOIN order_products op

    ON o.order_id = op.order_id

JOIN products p

    ON op.product_id = p.product_id

JOIN departments d

    ON p.department_id = d.department_id

GROUP BY
    d.department

ORDER BY
    total_purchases DESC

"""


save_query(
    department_analysis,
    "tableau_department_analysis.csv"
)


# ============================================================
# 5. PRODUCT ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("5. PRODUCT ANALYSIS")
print("=" * 80)


# IMPORTANT:
#
# We first calculate product-level statistics in product_stats.
#
# Then we calculate the median purchase volume separately.
#
# Finally we assign product segments.
#
# This avoids referencing an alias such as
# "product_purchase_count" before it exists.


product_analysis = """

WITH product_stats AS (

    SELECT

        p.product_id,

        p.product_name,

        a.aisle,

        d.department,

        COUNT(*) AS total_purchases,

        COUNT(
            DISTINCT o.order_id
        ) AS unique_orders,

        COUNT(
            DISTINCT o.user_id
        ) AS unique_customers,

        AVG(
            op.reordered
        ) AS reorder_rate,

        AVG(
            op.add_to_cart_order
        ) AS avg_cart_position

    FROM orders o

    JOIN order_products op

        ON o.order_id = op.order_id

    JOIN products p

        ON op.product_id = p.product_id

    JOIN aisles a

        ON p.aisle_id = a.aisle_id

    JOIN departments d

        ON p.department_id = d.department_id

    GROUP BY

        p.product_id,

        p.product_name,

        a.aisle,

        d.department
),

product_threshold AS (

    SELECT

        MEDIAN(
            total_purchases
        ) AS median_purchases

    FROM product_stats
)

SELECT

    ps.product_id,

    ps.product_name,

    ps.aisle,

    ps.department,

    ps.total_purchases,

    ps.unique_orders,

    ps.unique_customers,

    ROUND(
        ps.reorder_rate * 100,
        2
    ) AS reorder_rate_percent,

    ROUND(
        ps.avg_cart_position,
        2
    ) AS avg_cart_position,

    CASE

        WHEN
            ps.total_purchases
            >= pt.median_purchases

            AND
            ps.reorder_rate
            >= 0.50

        THEN
            'High Volume / High Reorder'


        WHEN
            ps.total_purchases
            >= pt.median_purchases

            AND
            ps.reorder_rate
            < 0.50

        THEN
            'High Volume / Lower Reorder'


        WHEN
            ps.total_purchases
            < pt.median_purchases

            AND
            ps.reorder_rate
            >= 0.50

        THEN
            'Lower Volume / High Reorder'


        ELSE
            'Lower Volume / Lower Reorder'

    END AS product_segment

FROM product_stats ps

CROSS JOIN product_threshold pt

ORDER BY
    ps.total_purchases DESC

"""


save_query(
    product_analysis,
    "tableau_product_analysis.csv"
)


# ============================================================
# 6. REORDER ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("6. REORDER ANALYSIS")
print("=" * 80)


reorder_analysis = """

SELECT

    CASE

        WHEN op.add_to_cart_order <= 3
            THEN '1-3'

        WHEN op.add_to_cart_order <= 5
            THEN '4-5'

        WHEN op.add_to_cart_order <= 10
            THEN '6-10'

        WHEN op.add_to_cart_order <= 20
            THEN '11-20'

        ELSE '21+'

    END AS cart_position_band,

    COUNT(*) AS purchases,

    ROUND(
        AVG(op.reordered) * 100,
        2
    ) AS reorder_rate_percent

FROM order_products op

GROUP BY 1

ORDER BY

    CASE cart_position_band

        WHEN '1-3'
            THEN 1

        WHEN '4-5'
            THEN 2

        WHEN '6-10'
            THEN 3

        WHEN '11-20'
            THEN 4

        ELSE 5

    END

"""


save_query(
    reorder_analysis,
    "tableau_reorder_analysis.csv"
)


# ============================================================
# 7. ML RECOMMENDATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. ML RECOMMENDATION ANALYSIS")
print("=" * 80)


ml_ranking = pd.DataFrame({

    "model": [

        "Global Popularity",

        "Logistic Regression",

        "XGBoost"

    ],

    "P@5": [

        0.099311,

        0.368296,

        0.399339

    ],

    "R@5": [

        0.072788,

        0.333142,

        0.359738

    ],

    "MAP@5": [

        0.069566,

        0.351626,

        0.398222

    ],

    "NDCG@5": [

        0.119577,

        0.454613,

        0.498822

    ],

    "P@10": [

        0.075431,

        0.294128,

        0.312617

    ],

    "R@10": [

        0.107021,

        0.489498,

        0.514675

    ],

    "MAP@10": [

        0.057902,

        0.340521,

        0.381079

    ],

    "NDCG@10": [

        0.116757,

        0.476705,

        0.514670

    ],

    "P@20": [

        0.054996,

        0.215689,

        0.223571

    ],

    "R@20": [

        0.151195,

        0.661444,

        0.679406

    ],

    "MAP@20": [

        0.058239,

        0.364455,

        0.401551

    ],

    "NDCG@20": [

        0.128420,

        0.531480,

        0.564286

    ]

})


ml_output_path = (
    TABLEAU_DIR
    / "tableau_ml_recommendation_analysis.csv"
)


ml_ranking.to_csv(
    ml_output_path,
    index=False
)


print(
    f"Rows: {len(ml_ranking)}"
)

print(
    f"Columns: {len(ml_ranking.columns)}"
)

print(
    f"Saved: {ml_output_path}"
)


# ============================================================
# 8. ML FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("8. ML FEATURE IMPORTANCE")
print("=" * 80)


feature_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "xgboost_feature_importance.csv"
)


if feature_file.exists():

    features = pd.read_csv(
        feature_file
    )

    required_feature_columns = {
        "feature",
        "importance"
    }

    missing_columns = (
        required_feature_columns
        -
        set(features.columns)
    )

    if missing_columns:

        raise ValueError(
            "xgboost_feature_importance.csv "
            f"is missing columns: {missing_columns}"
        )


    features = features.sort_values(
        "importance",
        ascending=False
    )


    feature_output_path = (
        TABLEAU_DIR
        / "tableau_ml_feature_importance.csv"
    )


    features.to_csv(
        feature_output_path,
        index=False
    )


    print(
        f"Rows: {len(features):,}"
    )

    print(
        f"Columns: {len(features.columns)}"
    )

    print(
        f"Saved: {feature_output_path}"
    )


else:

    print(
        "WARNING:"
    )

    print(
        f"Could not find {feature_file}"
    )

    print(
        "Feature importance file was skipped."
    )


# ============================================================
# 9. VALIDATE GENERATED FILES
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING TABLEAU OUTPUTS")
print("=" * 80)


expected_files = [

    "tableau_executive_kpis.csv",

    "tableau_customer_analysis.csv",

    "tableau_customer_segments.csv",

    "tableau_department_analysis.csv",

    "tableau_product_analysis.csv",

    "tableau_reorder_analysis.csv",

    "tableau_ml_recommendation_analysis.csv",

    "tableau_ml_feature_importance.csv"

]


all_files_ok = True


for filename in expected_files:

    path = (
        TABLEAU_DIR
        / filename
    )

    if path.exists():

        file_size_mb = (
            path.stat().st_size
            /
            (1024 * 1024)
        )

        print(
            f"✓ {filename}"
            f" ({file_size_mb:.2f} MB)"
        )

    else:

        print(
            f"✗ {filename}"
        )

        all_files_ok = False


# ============================================================
# CLOSE DATABASE
# ============================================================

con.close()


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)

if all_files_ok:

    print(
        "TABLEAU DATA PREPARATION COMPLETE"
    )

else:

    print(
        "TABLEAU DATA PREPARATION FINISHED WITH WARNINGS"
    )

print("=" * 80)


print(
    f"\nOutput directory:\n{TABLEAU_DIR}"
)

print("\nDone.")