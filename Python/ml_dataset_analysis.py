import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

con = duckdb.connect(str(DB_FILE))

print("=" * 70)
print("ML DATASET PREPARATION ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 1. Customers by number of orders
# --------------------------------------------------

print("\n1. CUSTOMERS BY ORDER COUNT")
print("-" * 70)

result = con.execute("""
    SELECT
        total_orders,
        COUNT(*) AS customers
    FROM customers
    GROUP BY total_orders
    ORDER BY total_orders
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 2. Maximum order number
# --------------------------------------------------

print("\n2. ORDER NUMBER RANGE")
print("-" * 70)

result = con.execute("""
    SELECT
        MIN(order_number) AS min_order,
        MAX(order_number) AS max_order,
        COUNT(DISTINCT order_number) AS unique_order_numbers
    FROM orders
""").fetchone()

print(f"Minimum order number : {result[0]}")
print(f"Maximum order number : {result[1]}")
print(f"Unique order numbers : {result[2]}")


# --------------------------------------------------
# 3. Orders by eval set
# --------------------------------------------------

print("\n3. EVALUATION SET")
print("-" * 70)

result = con.execute("""
    SELECT
        eval_set,
        COUNT(*) AS orders,
        COUNT(DISTINCT user_id) AS customers
    FROM read_csv_auto(
        'data/raw/orders.csv'
    )
    GROUP BY eval_set
    ORDER BY eval_set
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 4. Customers eligible for temporal prediction
# --------------------------------------------------

print("\n4. CUSTOMERS WITH ENOUGH HISTORY")
print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS customers,
        ROUND(AVG(total_orders), 2) AS avg_orders
    FROM customers
    WHERE total_orders >= 3
""").fetchone()

print(f"Customers with 3+ orders : {result[0]:,}")
print(f"Average orders           : {result[1]}")


# --------------------------------------------------
# 5. Orders that could be prediction targets
# --------------------------------------------------

print("\n5. POTENTIAL TARGET ORDERS")
print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS orders,
        COUNT(DISTINCT user_id) AS customers
    FROM orders
    WHERE order_number >= 2
""").fetchone()

print(f"Orders : {result[0]:,}")
print(f"Customers : {result[1]:,}")


con.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)