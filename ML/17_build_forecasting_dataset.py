from pathlib import Path
import duckdb
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "processed"

DB_PATH = PROCESSED_DIR / "instacart.duckdb"

OUTPUT_PATH = PROCESSED_DIR / "forecasting_dataset.parquet"

# Business horizon:
# Predict whether the customer's next order happens within 14 days.
ORDER_LIKELIHOOD_DAYS = 14


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_columns(con, table_name, required_columns):
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE {table_name}"
        ).fetchall()
    }

    missing = set(required_columns) - columns

    if missing:
        raise RuntimeError(
            f"{table_name} is missing columns: {sorted(missing)}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print_section("FORECASTING DATASET BUILD")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"DuckDB       : {DB_PATH}")
    print(f"Output       : {OUTPUT_PATH}")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {DB_PATH}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    # --------------------------------------------------------
    # 1. Validate existing tables
    # --------------------------------------------------------

    print_section("1. Validating source tables")

    tables = {
        row[0]
        for row in con.execute("SHOW TABLES").fetchall()
    }

    print("Available tables:")
    for table in sorted(tables):
        print(f"  - {table}")

    required_tables = {
        "orders",
        "order_products",
        "customers",
    }

    missing_tables = required_tables - tables

    if missing_tables:
        raise RuntimeError(
            f"Missing required tables: {sorted(missing_tables)}"
        )

    check_columns(
        con,
        "orders",
        [
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ],
    )

    check_columns(
        con,
        "order_products",
        [
            "order_id",
            "product_id",
            "reordered",
        ],
    )

    print("Source validation passed.")

    # --------------------------------------------------------
    # 2. Build order-level history
    # --------------------------------------------------------

    print_section("2. Building order-level history")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE order_history AS

        SELECT
            o.order_id,
            o.user_id,
            o.order_number,
            o.order_dow,
            o.order_hour_of_day,
            o.days_since_prior_order,

            COUNT(op.product_id) AS basket_size,

            SUM(
                CASE
                    WHEN op.reordered = 1 THEN 1
                    ELSE 0
                END
            ) AS reorder_items,

            CASE
                WHEN COUNT(op.product_id) > 0
                THEN
                    SUM(
                        CASE
                            WHEN op.reordered = 1 THEN 1
                            ELSE 0
                        END
                    )::DOUBLE
                    / COUNT(op.product_id)
                ELSE 0
            END AS order_reorder_rate

        FROM orders o

        INNER JOIN order_products op
            ON o.order_id = op.order_id

        GROUP BY
            o.order_id,
            o.user_id,
            o.order_number,
            o.order_dow,
            o.order_hour_of_day,
            o.days_since_prior_order
    """)

    order_count = con.execute(
        "SELECT COUNT(*) FROM order_history"
    ).fetchone()[0]

    print(f"Order-level rows: {order_count:,}")

    # --------------------------------------------------------
    # 3. Construct historical prediction points
    # --------------------------------------------------------
    #
    # Each current order becomes a prediction point.
    #
    # Features:
    #   information available UP TO current order
    #
    # Targets:
    #   information from the NEXT order
    #
    # The final observed order for a customer is excluded
    # because there is no observed next order.
    # --------------------------------------------------------

    print_section("3. Constructing prediction points")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE forecasting_base AS

        WITH ordered AS (

            SELECT
                *,

                LEAD(order_id) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                ) AS next_order_id,

                LEAD(order_number) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                ) AS next_order_number,

                LEAD(days_since_prior_order) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                ) AS next_days_since_prior_order,

                LEAD(basket_size) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                ) AS next_basket_size,

                LEAD(order_reorder_rate) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                ) AS next_reorder_rate,

                LAG(basket_size) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                ) AS previous_basket_size,

                LAG(order_reorder_rate) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                ) AS previous_order_reorder_rate

            FROM order_history
        ),

        history_features AS (

            SELECT
                *,

                order_number - 1
                    AS previous_orders,

                AVG(basket_size) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND 1 PRECEDING
                ) AS historical_avg_basket_size,

                AVG(order_reorder_rate) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND 1 PRECEDING
                ) AS historical_reorder_rate,

                AVG(days_since_prior_order) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND 1 PRECEDING
                ) AS historical_avg_days_between_orders,

                SUM(basket_size) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND 1 PRECEDING
                ) AS previous_total_items,

                SUM(reorder_items) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND 1 PRECEDING
                ) AS previous_total_reorder_items,

                COUNT(*) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND 1 PRECEDING
                ) AS observed_previous_orders

            FROM ordered
        )

        SELECT

            user_id,

            order_id AS prediction_order_id,

            order_number AS prediction_order_number,

            previous_orders,

            basket_size AS current_basket_size,

            COALESCE(
                historical_avg_basket_size,
                0
            ) AS historical_avg_basket_size,

            COALESCE(
                previous_basket_size,
                0
            ) AS previous_basket_size,

            COALESCE(
                historical_reorder_rate,
                0
            ) AS historical_reorder_rate,

            COALESCE(
                previous_order_reorder_rate,
                0
            ) AS previous_order_reorder_rate,

            COALESCE(
                historical_avg_days_between_orders,
                0
            ) AS historical_avg_days_between_orders,

            COALESCE(
                days_since_prior_order,
                0
            ) AS current_order_days_since_prior,

            previous_total_items,

            previous_total_reorder_items,

            COALESCE(
                previous_total_reorder_items::DOUBLE
                / NULLIF(previous_total_items, 0),
                0
            ) AS cumulative_reorder_rate,

            order_dow,

            order_hour_of_day,

            next_order_id,

            next_order_number,

            next_days_since_prior_order,

            next_basket_size,

            next_reorder_rate,

            CASE
                WHEN next_order_id IS NOT NULL
                     AND next_days_since_prior_order <= {horizon}
                THEN 1
                ELSE 0
            END AS next_order_within_horizon,

            CASE
                WHEN next_order_id IS NOT NULL
                THEN 1
                ELSE 0
            END AS next_order_observed

        FROM history_features

        WHERE next_order_id IS NOT NULL
    """.replace(
        "{horizon}",
        str(ORDER_LIKELIHOOD_DAYS)
    ))

    # --------------------------------------------------------
    # 4. Add recent behavior features
    # --------------------------------------------------------

    print_section("4. Adding recent behavior features")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE forecasting_dataset AS

        WITH recent AS (

            SELECT
                *,

                AVG(current_basket_size) OVER (
                    PARTITION BY user_id
                    ORDER BY prediction_order_number
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS recent_3_order_basket,

                AVG(
                    CASE
                        WHEN current_basket_size > 0
                        THEN
                            previous_order_reorder_rate
                        ELSE 0
                    END
                ) OVER (
                    PARTITION BY user_id
                    ORDER BY prediction_order_number
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS recent_3_order_reorder_rate,

                AVG(current_order_days_since_prior) OVER (
                    PARTITION BY user_id
                    ORDER BY prediction_order_number
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS recent_3_order_interval

            FROM forecasting_base
        )

        SELECT

            user_id,

            prediction_order_id,
            prediction_order_number,

            previous_orders,

            current_basket_size,
            historical_avg_basket_size,
            previous_basket_size,
            recent_3_order_basket,

            historical_reorder_rate,
            previous_order_reorder_rate,
            recent_3_order_reorder_rate,
            cumulative_reorder_rate,

            historical_avg_days_between_orders,
            current_order_days_since_prior,
            recent_3_order_interval,

            previous_total_items,
            previous_total_reorder_items,

            order_dow,
            order_hour_of_day,

            -- Targets
            next_order_within_horizon,
            next_order_observed,
            next_days_since_prior_order,
            next_basket_size,
            next_reorder_rate

        FROM recent
    """)

    # --------------------------------------------------------
    # 5. Materialize dataset
    # --------------------------------------------------------

    print_section("5. Materializing forecasting dataset")

    df = con.execute("""
        SELECT *
        FROM forecasting_dataset
        ORDER BY user_id, prediction_order_number
    """).fetchdf()

    print(f"Rows:              {len(df):,}")
    print(f"Customers:         {df['user_id'].nunique():,}")
    print(f"Prediction points: {df['prediction_order_id'].nunique():,}")

    # --------------------------------------------------------
    # 6. Basic validation
    # --------------------------------------------------------

    print_section("6. Dataset validation")

    print("\nNULL counts:")

    null_counts = df.isna().sum()

    for column, count in null_counts.items():
        if count > 0:
            print(f"  {column}: {count:,}")

    print("\nDuplicate prediction points:")

    duplicate_count = df.duplicated(
        subset=["user_id", "prediction_order_id"]
    ).sum()

    print(f"  {duplicate_count:,}")

    print("\nPrediction-order sequence check:")

    sequence_errors = (
        df["next_order_number"]
        <= df["prediction_order_number"]
    ).sum() if "next_order_number" in df.columns else 0

    print(f"  Invalid next-order sequences: {sequence_errors:,}")

    # --------------------------------------------------------
    # 7. Target distributions
    # --------------------------------------------------------

    print_section("7. Target distributions")

    likelihood_counts = (
        df["next_order_within_horizon"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nNext order within {ORDER_LIKELIHOOD_DAYS} days:"
    )

    for value, count in likelihood_counts.items():
        pct = count / len(df) * 100

        label = (
            "No"
            if value == 0
            else "Yes"
        )

        print(
            f"  {label:<5}: "
            f"{count:>12,} "
            f"({pct:6.2f}%)"
        )

    print("\nNext basket size:")

    print(
        df["next_basket_size"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nNext reorder rate:")

    print(
        df["next_reorder_rate"]
        .describe()
        .round(4)
        .to_string()
    )

    print(
        "\nNext-order interval (days):"
    )

    print(
        df["next_days_since_prior_order"]
        .describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # 8. Feature sanity checks
    # --------------------------------------------------------

    print_section("8. Leakage / feature sanity checks")

    forbidden_columns = {
        "next_order_id",
        "next_order_number",
        "next_days_since_prior_order",
        "next_basket_size",
        "next_reorder_rate",
        "next_order_within_horizon",
        "next_order_observed",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in forbidden_columns
    ]

    print("Model feature columns:")

    for column in feature_columns:
        print(f"  - {column}")

    print(
        "\nFuture target columns excluded from features:"
    )

    for column in sorted(forbidden_columns):
        print(f"  - {column}")

    # --------------------------------------------------------
    # 9. Save
    # --------------------------------------------------------

    print_section("9. Saving dataset")

    df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved successfully:\n"
        f"{OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print_section("FINAL SUMMARY")

    print(f"Forecasting horizon       : {ORDER_LIKELIHOOD_DAYS} days")
    print(f"Rows                      : {len(df):,}")
    print(f"Customers                 : {df['user_id'].nunique():,}")
    print(
        f"Positive order likelihood : "
        f"{df['next_order_within_horizon'].mean() * 100:.2f}%"
    )
    print(
        f"Average next basket       : "
        f"{df['next_basket_size'].mean():.2f}"
    )
    print(
        f"Average next reorder rate : "
        f"{df['next_reorder_rate'].mean() * 100:.2f}%"
    )
    print(f"NULL values               : {df.isna().sum().sum():,}")
    print(f"Duplicate prediction pts  : {duplicate_count:,}")

    con.close()

    print("\nForecasting dataset build completed.")


if __name__ == "__main__":
    main()