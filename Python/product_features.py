import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

con = duckdb.connect(str(DB_FILE))

print("=" * 70)
print("PRODUCT FEATURE ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# Create product analytics table
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE product_features AS

SELECT
    p.product_id,
    p.product_name,
    p.aisle,
    p.department,

    COUNT(*) AS total_purchases,

    COUNT(DISTINCT op.order_id)
        AS unique_orders,

    COUNT(DISTINCT o.user_id)
        AS unique_customers,

    ROUND(AVG(op.reordered) * 100, 2)
        AS reorder_rate,

    ROUND(AVG(op.add_to_cart_order), 2)
        AS avg_cart_position

FROM order_products op

JOIN products p
    USING (product_id)

JOIN orders o
    USING (order_id)

GROUP BY
    p.product_id,
    p.product_name,
    p.aisle,
    p.department;
""")


# --------------------------------------------------
# Overall summary
# --------------------------------------------------

print("\n1. PRODUCT FEATURE SUMMARY")
print("-" * 70)

result = con.execute("""
SELECT
    COUNT(*) AS products,
    ROUND(AVG(total_purchases), 2) AS avg_purchases,
    ROUND(AVG(unique_customers), 2) AS avg_customers,
    ROUND(AVG(reorder_rate), 2) AS avg_reorder_rate,
    ROUND(AVG(avg_cart_position), 2) AS avg_cart_position
FROM product_features;
""").fetchone()

print(f"Products            : {result[0]:,}")
print(f"Avg purchases       : {result[1]}")
print(f"Avg customers       : {result[2]}")
print(f"Avg reorder rate    : {result[3]}%")
print(f"Avg cart position   : {result[4]}")


# --------------------------------------------------
# Popularity vs reorder
# --------------------------------------------------

print("\n2. POPULAR PRODUCTS WITH REORDER RATE")
print("-" * 70)

result = con.execute("""
SELECT
    product_name,
    department,
    total_purchases,
    unique_customers,
    reorder_rate
FROM product_features
ORDER BY total_purchases DESC
LIMIT 20;
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# Products with strong repeat behavior
# --------------------------------------------------

print("\n3. HIGH-REORDER PRODUCTS")
print("-" * 70)

result = con.execute("""
SELECT
    product_name,
    department,
    total_purchases,
    unique_customers,
    reorder_rate
FROM product_features
WHERE total_purchases >= 1000
ORDER BY reorder_rate DESC
LIMIT 20;
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# Cart position vs reorder
# --------------------------------------------------

print("\n4. CART POSITION VS REORDER RATE")
print("-" * 70)

result = con.execute("""
SELECT
    CASE
        WHEN avg_cart_position <= 3 THEN '1-3'
        WHEN avg_cart_position <= 5 THEN '4-5'
        WHEN avg_cart_position <= 10 THEN '6-10'
        WHEN avg_cart_position <= 20 THEN '11-20'
        ELSE '21+'
    END AS cart_position_group,

    COUNT(*) AS products,

    ROUND(AVG(total_purchases), 0)
        AS avg_purchases,

    ROUND(AVG(reorder_rate), 2)
        AS avg_reorder_rate

FROM product_features

GROUP BY cart_position_group

ORDER BY
    CASE cart_position_group
        WHEN '1-3' THEN 1
        WHEN '4-5' THEN 2
        WHEN '6-10' THEN 3
        WHEN '11-20' THEN 4
        ELSE 5
    END;
""").fetchdf()

print(result.to_string(index=False))


con.close()

print("\n" + "=" * 70)
print("PRODUCT FEATURE ANALYSIS COMPLETE")
print("=" * 70)