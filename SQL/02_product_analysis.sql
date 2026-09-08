-- ============================================================
-- PRODUCT ANALYSIS
-- ============================================================


-- ============================================================
-- Q1. What are the top products by purchase volume?
-- ============================================================

SELECT
    p.product_id,
    p.product_name,
    p.department,

    COUNT(*) AS total_purchases,

    COUNT(DISTINCT o.user_id) AS unique_customers,

    ROUND(AVG(op.reordered) * 100, 2) AS reorder_rate

FROM order_products op

JOIN orders o
    ON op.order_id = o.order_id

JOIN products p
    ON op.product_id = p.product_id

GROUP BY
    p.product_id,
    p.product_name,
    p.department

ORDER BY total_purchases DESC

LIMIT 50;


-- ============================================================
-- Q2. Which products have the highest reorder rate?
--     Require at least 1,000 purchases to avoid tiny samples.
-- ============================================================

SELECT
    p.product_id,
    p.product_name,
    p.department,

    COUNT(*) AS total_purchases,

    COUNT(DISTINCT o.user_id) AS unique_customers,

    ROUND(AVG(op.reordered) * 100, 2) AS reorder_rate

FROM order_products op

JOIN orders o
    ON op.order_id = o.order_id

JOIN products p
    ON op.product_id = p.product_id

GROUP BY
    p.product_id,
    p.product_name,
    p.department

HAVING COUNT(*) >= 1000

ORDER BY reorder_rate DESC

LIMIT 50;


-- ============================================================
-- Q3. Which products have both high volume and high reorder?
-- ============================================================

WITH product_metrics AS (

    SELECT
        p.product_id,
        p.product_name,
        p.department,

        COUNT(*) AS total_purchases,

        COUNT(DISTINCT o.user_id) AS unique_customers,

        AVG(op.reordered) * 100 AS reorder_rate

    FROM order_products op

    JOIN orders o
        ON op.order_id = o.order_id

    JOIN products p
        ON op.product_id = p.product_id

    GROUP BY
        p.product_id,
        p.product_name,
        p.department
),

thresholds AS (

    SELECT
        MEDIAN(total_purchases) AS volume_median,
        MEDIAN(reorder_rate) AS reorder_median

    FROM product_metrics
)

SELECT
    pm.*

FROM product_metrics pm

CROSS JOIN thresholds t

WHERE
    pm.total_purchases >= t.volume_median
    AND pm.reorder_rate >= t.reorder_median

ORDER BY
    pm.total_purchases DESC

LIMIT 50;


-- ============================================================
-- Q4. Which departments contain the strongest products?
-- ============================================================

SELECT
    p.department,

    COUNT(DISTINCT p.product_id) AS products,

    COUNT(*) AS total_purchases,

    ROUND(AVG(op.reordered) * 100, 2) AS avg_reorder_rate

FROM order_products op

JOIN products p
    ON op.product_id = p.product_id

GROUP BY p.department

ORDER BY total_purchases DESC;