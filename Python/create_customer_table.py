import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

con = duckdb.connect(str(DB_FILE))

print("Creating customer analytics table...")


# --------------------------------------------------
# Customer-level features
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE customers AS

WITH order_metrics AS (

    SELECT
        user_id,

        COUNT(DISTINCT order_id) AS total_orders,

        COUNT(*) AS total_items,

        ROUND(
            COUNT(*) * 1.0 /
            COUNT(DISTINCT order_id),
            2
        ) AS avg_items_per_order,

        AVG(days_since_prior_order)
            AS avg_days_between_orders,

        MAX(order_number)
            AS max_order_number

    FROM orders
    JOIN order_products
        USING (order_id)

    GROUP BY user_id
),

reorder_metrics AS (

    SELECT
        o.user_id,

        ROUND(
            AVG(op.reordered) * 100,
            2
        ) AS reorder_rate

    FROM orders o

    JOIN order_products op
        USING (order_id)

    GROUP BY o.user_id
)

SELECT
    o.user_id,
    o.total_orders,
    o.total_items,
    o.avg_items_per_order,
    r.reorder_rate,
    ROUND(o.avg_days_between_orders, 2)
        AS avg_days_between_orders,
    o.max_order_number

FROM order_metrics o

JOIN reorder_metrics r
    USING (user_id);
""")


# --------------------------------------------------
# Verify
# --------------------------------------------------

result = con.execute("""
SELECT
    COUNT(*) AS customers,
    ROUND(AVG(total_orders), 2) AS avg_orders,
    ROUND(AVG(total_items), 2) AS avg_items,
    ROUND(AVG(avg_items_per_order), 2) AS avg_basket,
    ROUND(AVG(reorder_rate), 2) AS avg_reorder_rate,
    ROUND(AVG(avg_days_between_orders), 2) AS avg_days_between_orders
FROM customers;
""").fetchone()


print("\nCUSTOMER TABLE SUMMARY")
print("=" * 60)

print(f"Customers             : {result[0]:,}")
print(f"Average orders        : {result[1]}")
print(f"Average items         : {result[2]}")
print(f"Average basket        : {result[3]}")
print(f"Average reorder rate  : {result[4]}%")
print(f"Avg days between orders: {result[5]}")


con.close()

print("\nCustomer table created successfully.")