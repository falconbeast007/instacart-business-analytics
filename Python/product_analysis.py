import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "data" / "processed" / "instacart.duckdb"

con = duckdb.connect(str(DB_FILE))

print("=" * 70)
print("PRODUCT & DEPARTMENT ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 1. Department reorder rates
# --------------------------------------------------

print("\n1. DEPARTMENT PERFORMANCE")
print("-" * 70)

result = con.execute("""
    SELECT
        department,

        COUNT(*) AS items_purchased,

        COUNT(DISTINCT order_id) AS orders,

        COUNT(DISTINCT product_id) AS products,

        ROUND(AVG(reordered) * 100, 2)
            AS reorder_rate

    FROM order_products op

    JOIN products p
        USING (product_id)

    GROUP BY department

    ORDER BY items_purchased DESC
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 2. Highest reorder-rate departments
# --------------------------------------------------

print("\n2. DEPARTMENTS WITH HIGHEST REORDER RATE")
print("-" * 70)

result = con.execute("""
    SELECT
        department,
        COUNT(*) AS purchases,
        ROUND(AVG(reordered) * 100, 2)
            AS reorder_rate

    FROM order_products op

    JOIN products p
        USING (product_id)

    GROUP BY department

    HAVING COUNT(*) >= 10000

    ORDER BY reorder_rate DESC
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 3. Top products by purchase volume
# --------------------------------------------------

print("\n3. TOP 20 PRODUCTS BY PURCHASE VOLUME")
print("-" * 70)

result = con.execute("""
    SELECT
        p.product_id,
        p.product_name,
        p.department,

        COUNT(*) AS purchases,

        COUNT(DISTINCT op.order_id) AS orders,

        COUNT(DISTINCT o.user_id) AS customers,

        ROUND(AVG(op.reordered) * 100, 2)
            AS reorder_rate

    FROM order_products op

    JOIN products p
        USING (product_id)

    JOIN orders o
        USING (order_id)

    GROUP BY
        p.product_id,
        p.product_name,
        p.department

    ORDER BY purchases DESC

    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 4. Products with highest reorder rate
# --------------------------------------------------

print("\n4. PRODUCTS WITH HIGHEST REORDER RATE")
print("-" * 70)

result = con.execute("""
    SELECT
        p.product_id,
        p.product_name,
        p.department,

        COUNT(*) AS purchases,

        ROUND(AVG(op.reordered) * 100, 2)
            AS reorder_rate

    FROM order_products op

    JOIN products p
        USING (product_id)

    GROUP BY
        p.product_id,
        p.product_name,
        p.department

    HAVING COUNT(*) >= 500

    ORDER BY reorder_rate DESC

    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 5. Products with low reorder rate
# --------------------------------------------------

print("\n5. PRODUCTS WITH LOWEST REORDER RATE")
print("-" * 70)

result = con.execute("""
    SELECT
        p.product_id,
        p.product_name,
        p.department,

        COUNT(*) AS purchases,

        ROUND(AVG(op.reordered) * 100, 2)
            AS reorder_rate

    FROM order_products op

    JOIN products p
        USING (product_id)

    GROUP BY
        p.product_id,
        p.product_name,
        p.department

    HAVING COUNT(*) >= 500

    ORDER BY reorder_rate ASC

    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# --------------------------------------------------
# 6. Aisle performance
# --------------------------------------------------

print("\n6. TOP AISLES BY PURCHASE VOLUME")
print("-" * 70)

result = con.execute("""
    SELECT
        p.aisle,

        COUNT(*) AS purchases,

        COUNT(DISTINCT op.product_id)
            AS products,

        ROUND(AVG(op.reordered) * 100, 2)
            AS reorder_rate

    FROM order_products op

    JOIN products p
        USING (product_id)

    GROUP BY p.aisle

    ORDER BY purchases DESC

    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


con.close()

print("\n" + "=" * 70)
print("PRODUCT ANALYSIS COMPLETE")
print("=" * 70)