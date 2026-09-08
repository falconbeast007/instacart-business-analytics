import duckdb
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "instacart_master.parquet"

con = duckdb.connect()

print("=" * 70)
print("INSTACART BUSINESS ANALYTICS - DATA EXPLORATION")
print("=" * 70)


# --------------------------------------------------
# 1. Overall dataset
# --------------------------------------------------

print("\n1. OVERALL DATASET")

result = con.execute(f"""
    SELECT
        COUNT(*) AS records,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT user_id) AS customers,
        COUNT(DISTINCT product_id) AS products
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Records   : {result[0]:,}")
print(f"Orders    : {result[1]:,}")
print(f"Customers : {result[2]:,}")
print(f"Products  : {result[3]:,}")


# --------------------------------------------------
# 2. Department performance
# --------------------------------------------------

print("\n2. TOP DEPARTMENTS BY PURCHASES")

result = con.execute(f"""
    SELECT
        department,
        COUNT(*) AS items_purchased,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT user_id) AS customers
    FROM read_parquet('{DATA_FILE}')
    GROUP BY department
    ORDER BY items_purchased DESC
    LIMIT 15
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 3. Top products
# --------------------------------------------------

print("\n3. TOP 20 PRODUCTS BY PURCHASE FREQUENCY")

result = con.execute(f"""
    SELECT
        product_name,
        department,
        COUNT(*) AS times_purchased,
        COUNT(DISTINCT user_id) AS customers,
        ROUND(AVG(reordered) * 100, 2) AS reorder_rate
    FROM read_parquet('{DATA_FILE}')
    GROUP BY product_name, department
    ORDER BY times_purchased DESC
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 4. Reorder behavior
# --------------------------------------------------

print("\n4. REORDER ANALYSIS")

result = con.execute(f"""
    SELECT
        reordered,
        COUNT(*) AS records,
        ROUND(
            COUNT(*) * 100.0 /
            SUM(COUNT(*)) OVER (),
            2
        ) AS percentage
    FROM read_parquet('{DATA_FILE}')
    GROUP BY reordered
    ORDER BY reordered
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 5. Orders by hour
# --------------------------------------------------

print("\n5. ORDERS BY HOUR")

result = con.execute(f"""
    SELECT
        order_hour_of_day AS hour,
        COUNT(DISTINCT order_id) AS orders
    FROM read_parquet('{DATA_FILE}')
    GROUP BY order_hour_of_day
    ORDER BY hour
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 6. Orders by day of week
# --------------------------------------------------

print("\n6. ORDERS BY DAY OF WEEK")

result = con.execute(f"""
    SELECT
        order_dow AS day_of_week,
        COUNT(DISTINCT order_id) AS orders
    FROM read_parquet('{DATA_FILE}')
    GROUP BY order_dow
    ORDER BY day_of_week
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 7. Customer order frequency
# --------------------------------------------------

print("\n7. CUSTOMER ORDER FREQUENCY")

result = con.execute(f"""
    SELECT
        order_count,
        COUNT(*) AS customers
    FROM (
        SELECT
            user_id,
            COUNT(DISTINCT order_id) AS order_count
        FROM read_parquet('{DATA_FILE}')
        GROUP BY user_id
    )
    GROUP BY order_count
    ORDER BY order_count
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 8. Average basket size
# --------------------------------------------------

print("\n8. BASKET SIZE")

result = con.execute(f"""
    SELECT
        ROUND(AVG(items), 2) AS avg_items_per_order,
        MIN(items) AS min_items,
        MAX(items) AS max_items
    FROM (
        SELECT
            order_id,
            COUNT(*) AS items
        FROM read_parquet('{DATA_FILE}')
        GROUP BY order_id
    )
""").fetchone()

print(f"Average items/order : {result[0]}")
print(f"Minimum             : {result[1]}")
print(f"Maximum             : {result[2]}")


con.close()

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)