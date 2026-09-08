import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

con = duckdb.connect(str(DB_FILE))

print("=" * 70)
print("CUSTOMER BEHAVIOR ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 1. Customer order distribution
# --------------------------------------------------

print("\n1. CUSTOMER ORDER DISTRIBUTION")
print("-" * 70)

result = con.execute("""
    SELECT
        CASE
            WHEN total_orders <= 5 THEN '1-5'
            WHEN total_orders <= 10 THEN '6-10'
            WHEN total_orders <= 20 THEN '11-20'
            WHEN total_orders <= 30 THEN '21-30'
            ELSE '31+'
        END AS order_group,
        COUNT(*) AS customers
    FROM customers
    GROUP BY order_group
    ORDER BY
        CASE order_group
            WHEN '1-5' THEN 1
            WHEN '6-10' THEN 2
            WHEN '11-20' THEN 3
            WHEN '21-30' THEN 4
            ELSE 5
        END
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 2. Top customers by number of orders
# --------------------------------------------------

print("\n2. MOST FREQUENT CUSTOMERS")
print("-" * 70)

result = con.execute("""
    SELECT
        user_id,
        total_orders,
        total_items,
        avg_items_per_order,
        reorder_rate
    FROM customers
    ORDER BY total_orders DESC
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 3. Highest reorder customers
# --------------------------------------------------

print("\n3. CUSTOMERS WITH HIGHEST REORDER RATE")
print("-" * 70)

result = con.execute("""
    SELECT
        user_id,
        total_orders,
        total_items,
        avg_items_per_order,
        reorder_rate
    FROM customers
    WHERE total_orders >= 5
    ORDER BY reorder_rate DESC
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 4. Average basket by customer order frequency
# --------------------------------------------------

print("\n4. BASKET SIZE BY CUSTOMER ORDER FREQUENCY")
print("-" * 70)

result = con.execute("""
    SELECT
        CASE
            WHEN total_orders <= 5 THEN '1-5'
            WHEN total_orders <= 10 THEN '6-10'
            WHEN total_orders <= 20 THEN '11-20'
            WHEN total_orders <= 30 THEN '21-30'
            ELSE '31+'
        END AS order_group,

        COUNT(*) AS customers,

        ROUND(AVG(avg_items_per_order), 2)
            AS avg_basket,

        ROUND(AVG(reorder_rate), 2)
            AS avg_reorder_rate

    FROM customers

    GROUP BY order_group

    ORDER BY
        CASE order_group
            WHEN '1-5' THEN 1
            WHEN '6-10' THEN 2
            WHEN '11-20' THEN 3
            WHEN '21-30' THEN 4
            ELSE 5
        END
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 5. Customer segments
# --------------------------------------------------

print("\n5. CUSTOMER SEGMENTS")
print("-" * 70)

result = con.execute("""
    SELECT
        CASE
            WHEN total_orders >= 20
                 AND reorder_rate >= 60
                THEN 'Highly Loyal'

            WHEN total_orders >= 20
                THEN 'Frequent'

            WHEN total_orders < 10
                 AND reorder_rate >= 50
                THEN 'Repeat-Oriented'

            WHEN total_orders < 10
                THEN 'Occasional'

            ELSE 'Regular'
        END AS segment,

        COUNT(*) AS customers,

        ROUND(AVG(total_orders), 2)
            AS avg_orders,

        ROUND(AVG(avg_items_per_order), 2)
            AS avg_basket,

        ROUND(AVG(reorder_rate), 2)
            AS avg_reorder_rate

    FROM customers

    GROUP BY segment

    ORDER BY customers DESC
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 6. Days between orders
# --------------------------------------------------

print("\n6. CUSTOMER ORDER INTERVALS")
print("-" * 70)

result = con.execute("""
    SELECT
        CASE
            WHEN avg_days_between_orders <= 7 THEN '0-7 days'
            WHEN avg_days_between_orders <= 14 THEN '8-14 days'
            WHEN avg_days_between_orders <= 21 THEN '15-21 days'
            WHEN avg_days_between_orders <= 30 THEN '22-30 days'
            ELSE '30+ days'
        END AS interval_group,

        COUNT(*) AS customers

    FROM customers

    GROUP BY interval_group

    ORDER BY
        CASE interval_group
            WHEN '0-7 days' THEN 1
            WHEN '8-14 days' THEN 2
            WHEN '15-21 days' THEN 3
            WHEN '22-30 days' THEN 4
            ELSE 5
        END
""").fetchdf()

print(result.to_string(index=False))


con.close()

print("\n" + "=" * 70)
print("CUSTOMER ANALYSIS COMPLETE")
print("=" * 70)