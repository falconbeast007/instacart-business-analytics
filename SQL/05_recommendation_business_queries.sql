-- ============================================================
-- RECOMMENDATION BUSINESS ANALYSIS
-- ============================================================


-- ============================================================
-- Q1. What products does a customer repeatedly purchase?
-- ============================================================

SELECT
    o.user_id,

    p.product_id,
    p.product_name,
    p.department,

    COUNT(*) AS previous_purchases,

    SUM(op.reordered) AS previous_reorders,

    ROUND(
        AVG(op.reordered) * 100,
        2
    ) AS customer_product_reorder_rate

FROM orders o

JOIN order_products op
    ON o.order_id = op.order_id

JOIN products p
    ON op.product_id = p.product_id

GROUP BY
    o.user_id,
    p.product_id,
    p.product_name,
    p.department

HAVING COUNT(*) >= 2

ORDER BY
    o.user_id,
    previous_purchases DESC;


-- ============================================================
-- Q2. Which products are strong candidates for
--     personalized replenishment?
--
-- High customer-specific repeat behavior +
-- meaningful product demand.
-- ============================================================

WITH customer_product AS (

    SELECT
        o.user_id,
        p.product_id,
        p.product_name,
        p.department,

        COUNT(*) AS previous_purchases,

        AVG(op.reordered) AS customer_product_reorder_rate

    FROM orders o

    JOIN order_products op
        ON o.order_id = op.order_id

    JOIN products p
        ON op.product_id = p.product_id

    GROUP BY
        o.user_id,
        p.product_id,
        p.product_name,
        p.department
),

product_demand AS (

    SELECT
        product_id,

        COUNT(*) AS product_purchase_count,

        COUNT(DISTINCT user_id) AS product_unique_customers

    FROM orders o

    JOIN order_products op
        ON o.order_id = op.order_id

    GROUP BY product_id
)

SELECT
    cp.user_id,

    cp.product_id,
    cp.product_name,
    cp.department,

    cp.previous_purchases,

    ROUND(
        cp.customer_product_reorder_rate * 100,
        2
    ) AS customer_product_reorder_rate,

    pd.product_purchase_count,

    pd.product_unique_customers

FROM customer_product cp

JOIN product_demand pd
    ON cp.product_id = pd.product_id

WHERE
    cp.previous_purchases >= 2

ORDER BY
    cp.user_id,
    cp.previous_purchases DESC;


-- ============================================================
-- Q3. What is the candidate coverage of historical products?
--
-- This is useful for understanding the limitation of a
-- recommendation strategy based only on previously purchased
-- products.
-- ============================================================

WITH customer_products AS (

    SELECT DISTINCT
        o.user_id,
        op.product_id

    FROM orders o

    JOIN order_products op
        ON o.order_id = op.order_id
),

customer_product_counts AS (

    SELECT
        user_id,
        COUNT(DISTINCT product_id) AS distinct_products

    FROM customer_products

    GROUP BY user_id
)

SELECT
    ROUND(
        AVG(distinct_products),
        2
    ) AS avg_distinct_products_per_customer,

    MEDIAN(distinct_products)
        AS median_distinct_products_per_customer,

    MAX(distinct_products)
        AS max_distinct_products_per_customer

FROM customer_product_counts;


-- ============================================================
-- Q4. Most frequently purchased products overall
--     as a global popularity baseline.
-- ============================================================

SELECT
    p.product_id,
    p.product_name,
    p.department,

    COUNT(*) AS total_purchases,

    COUNT(DISTINCT o.user_id) AS unique_customers,

    ROUND(
        AVG(op.reordered) * 100,
        2
    ) AS reorder_rate

FROM order_products op

JOIN orders o
    ON op.order_id = o.order_id

JOIN products p
    ON op.product_id = p.product_id

GROUP BY
    p.product_id,
    p.product_name,
    p.department

ORDER BY
    total_purchases DESC

LIMIT 100;


-- ============================================================
-- Q5. Products frequently bought together
-- ============================================================

SELECT
    p1.product_name AS product_a,
    p2.product_name AS product_b,

    COUNT(DISTINCT op1.order_id) AS co_purchase_orders

FROM order_products op1

JOIN order_products op2
    ON op1.order_id = op2.order_id
    AND op1.product_id < op2.product_id

JOIN products p1
    ON op1.product_id = p1.product_id

JOIN products p2
    ON op2.product_id = p2.product_id

GROUP BY
    p1.product_name,
    p2.product_name

HAVING
    COUNT(DISTINCT op1.order_id) >= 100

ORDER BY
    co_purchase_orders DESC

LIMIT 100;