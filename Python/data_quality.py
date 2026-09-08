import duckdb
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_FILE = BASE_DIR / "data" / "processed" / "instacart_master.parquet"

con = duckdb.connect()

print("=" * 70)
print("INSTACART DATA QUALITY AUDIT")
print("=" * 70)


# ==================================================
# 1. NULL / MISSING VALUES
# ==================================================

print("\n1. NULL VALUE CHECK")
print("-" * 70)

columns = con.execute(f"""
    DESCRIBE SELECT *
    FROM read_parquet('{DATA_FILE}')
""").fetchall()

for column in columns:
    column_name = column[0]

    result = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT("{column_name}") AS non_null,
            COUNT(*) - COUNT("{column_name}") AS null_count
        FROM read_parquet('{DATA_FILE}')
    """).fetchone()

    null_count = result[2]
    null_percentage = (null_count / result[0]) * 100

    print(
        f"{column_name:25} "
        f"NULLs: {null_count:>10,} "
        f"({null_percentage:.2f}%)"
    )


# ==================================================
# 2. DUPLICATE ROWS
# ==================================================

print("\n2. DUPLICATE ROW CHECK")
print("-" * 70)

result = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(*) - COUNT(DISTINCT
            CONCAT_WS('|',
                CAST(order_id AS VARCHAR),
                CAST(product_id AS VARCHAR),
                CAST(add_to_cart_order AS VARCHAR),
                CAST(reordered AS VARCHAR)
            )
        ) AS duplicate_rows
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Total rows     : {result[0]:,}")
print(f"Duplicate rows : {result[1]:,}")


# ==================================================
# 3. DUPLICATE ORDER IDs
# ==================================================

print("\n3. ORDER ID CHECK")
print("-" * 70)

result = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT order_id) AS unique_orders
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Rows          : {result[0]:,}")
print(f"Unique orders : {result[1]:,}")


# ==================================================
# 4. DUPLICATE PRODUCT IDs
# ==================================================

print("\n4. PRODUCT ID CHECK")
print("-" * 70)

result = con.execute(f"""
    SELECT
        COUNT(DISTINCT product_id) AS unique_products,
        COUNT(DISTINCT product_name) AS unique_product_names
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Unique product IDs   : {result[0]:,}")
print(f"Unique product names : {result[1]:,}")


# ==================================================
# 5. REORDER VALUES
# ==================================================

print("\n5. REORDER VALUE CHECK")
print("-" * 70)

result = con.execute(f"""
    SELECT
        reordered,
        COUNT(*) AS records
    FROM read_parquet('{DATA_FILE}')
    GROUP BY reordered
    ORDER BY reordered
""").fetchall()

for row in result:
    print(f"reordered = {row[0]} : {row[1]:,}")


# ==================================================
# 6. ORDER DAY RANGE
# ==================================================

print("\n6. ORDER DAY CHECK")
print("-" * 70)

result = con.execute(f"""
    SELECT
        MIN(order_dow),
        MAX(order_dow),
        COUNT(DISTINCT order_dow)
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Minimum day : {result[0]}")
print(f"Maximum day : {result[1]}")
print(f"Unique days : {result[2]}")


# ==================================================
# 7. ORDER HOUR RANGE
# ==================================================

print("\n7. ORDER HOUR CHECK")
print("-" * 70)

result = con.execute(f"""
    SELECT
        MIN(order_hour_of_day),
        MAX(order_hour_of_day),
        COUNT(DISTINCT order_hour_of_day)
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Minimum hour : {result[0]}")
print(f"Maximum hour : {result[1]}")
print(f"Unique hours : {result[2]}")


# ==================================================
# 8. REORDER VALIDATION
# ==================================================

print("\n8. REORDER VALIDATION")
print("-" * 70)

result = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{DATA_FILE}')
    WHERE reordered NOT IN (0, 1)
""").fetchone()

print(f"Invalid reorder values: {result[0]:,}")


# ==================================================
# 9. CART POSITION CHECK
# ==================================================

print("\n9. ADD-TO-CART POSITION CHECK")
print("-" * 70)

result = con.execute(f"""
    SELECT
        MIN(add_to_cart_order),
        MAX(add_to_cart_order),
        COUNT(*)
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Minimum position : {result[0]}")
print(f"Maximum position : {result[1]}")
print(f"Records          : {result[2]:,}")


# ==================================================
# 10. REFERENTIAL INTEGRITY
# ==================================================

print("\n10. REFERENTIAL INTEGRITY")
print("-" * 70)

# Products missing from product table
result = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{DATA_FILE}')
    WHERE product_name IS NULL
""").fetchone()

print(f"Products without product information: {result[0]:,}")


# Aisle missing
result = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{DATA_FILE}')
    WHERE aisle IS NULL
""").fetchone()

print(f"Products without aisle information  : {result[0]:,}")


# Department missing
result = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{DATA_FILE}')
    WHERE department IS NULL
""").fetchone()

print(f"Products without department info     : {result[0]:,}")


# ==================================================
# 11. DAYS SINCE PRIOR ORDER
# ==================================================

print("\n11. DAYS SINCE PRIOR ORDER")
print("-" * 70)

result = con.execute(f"""
    SELECT
        MIN(days_since_prior_order),
        MAX(days_since_prior_order),
        COUNT(*) AS total,
        COUNT(days_since_prior_order) AS non_null
    FROM read_parquet('{DATA_FILE}')
""").fetchone()

print(f"Minimum : {result[0]}")
print(f"Maximum : {result[1]}")
print(f"Total   : {result[2]:,}")
print(f"Non-null: {result[3]:,}")


# ==================================================
# FINISH
# ==================================================

con.close()

print("\n" + "=" * 70)
print("DATA QUALITY AUDIT COMPLETE")
print("=" * 70)