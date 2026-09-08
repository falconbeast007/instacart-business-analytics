-- ============================================================
-- DEPARTMENT ANALYSIS
-- ============================================================


-- ============================================================
-- Q1. Which departments drive the most purchases?
-- ============================================================

SELECT
    p.department,

    COUNT(*) AS total_purchases,

    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (),
        2
    ) AS purchase_share,

    COUNT(DISTINCT o.user_id) AS unique_customers

FROM order_products op

JOIN orders o
    ON op.order_id = o.order_id

JOIN products p
    ON op.product_id = p.product_id

GROUP BY p.department

ORDER BY total_purchases DESC;


-- ============================================================
-- Q2. Which departments have the highest customer penetration?
-- ============================================================

WITH total_customers AS (

    SELECT
        COUNT(DISTINCT user_id) AS customers

    FROM orders
),

department_metrics AS (

    SELECT
        p.department,

        COUNT(DISTINCT o.user_id) AS unique_customers,

        COUNT(*) AS total_purchases,

        AVG(op.reordered) * 100 AS reorder_rate

    FROM order_products op

    JOIN orders o
        ON op.order_id = o.order_id

    JOIN products p
        ON op.product_id = p.product_id

    GROUP BY p.department
)

SELECT
    dm.department,

    dm.unique_customers,

    ROUND(
        dm.unique_customers * 100.0 /
        tc.customers,
        2
    ) AS customer_penetration,

    dm.total_purchases,

    ROUND(dm.reorder_rate, 2) AS reorder_rate

FROM department_metrics dm

CROSS JOIN total_customers tc

ORDER BY customer_penetration DESC;


-- ============================================================
-- Q3. Find departments with high reach but low reorder.
-- ============================================================

WITH department_metrics AS (

    SELECT
        p.department,

        COUNT(DISTINCT o.user_id) AS unique_customers,

        COUNT(*) AS total_purchases,

        AVG(op.reordered) * 100 AS reorder_rate

    FROM order_products op

    JOIN orders o
        ON op.order_id = o.order_id

    JOIN products p
        ON op.product_id = p.product_id

    GROUP BY p.department
),

total_customers AS (

    SELECT COUNT(DISTINCT user_id) AS customers
    FROM orders
)

SELECT
    dm.department,

    ROUND(
        dm.unique_customers * 100.0 /
        tc.customers,
        2
    ) AS customer_penetration,

    ROUND(dm.reorder_rate, 2) AS reorder_rate,

    dm.total_purchases

FROM department_metrics dm

CROSS JOIN total_customers tc

WHERE
    dm.unique_customers * 100.0 / tc.customers >= 70
    AND dm.reorder_rate < 50

ORDER BY
    reorder_rate ASC;