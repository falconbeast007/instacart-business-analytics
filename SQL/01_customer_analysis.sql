-- ============================================================
-- INSTACART BUSINESS ANALYTICS
-- CUSTOMER ANALYSIS
-- ============================================================


-- ============================================================
-- Q1. How many customers do we have?
-- ============================================================

SELECT
    COUNT(DISTINCT user_id) AS total_customers
FROM orders;


-- ============================================================
-- Q2. What is the distribution of customers by order frequency?
-- ============================================================

WITH customer_orders AS (

    SELECT
        user_id,
        MAX(order_number) AS total_orders

    FROM orders

    GROUP BY user_id
)

SELECT
    CASE
        WHEN total_orders BETWEEN 1 AND 5 THEN '1-5'
        WHEN total_orders BETWEEN 6 AND 10 THEN '6-10'
        WHEN total_orders BETWEEN 11 AND 20 THEN '11-20'
        WHEN total_orders BETWEEN 21 AND 30 THEN '21-30'
        ELSE '31+'
    END AS order_group,

    COUNT(*) AS customers

FROM customer_orders

GROUP BY order_group

ORDER BY
    CASE order_group
        WHEN '1-5' THEN 1
        WHEN '6-10' THEN 2
        WHEN '11-20' THEN 3
        WHEN '21-30' THEN 4
        WHEN '31+' THEN 5
    END;


-- ============================================================
-- Q3. What is the average basket size and reorder rate
--     by customer order frequency?
-- ============================================================

WITH customer_metrics AS (

    SELECT
        o.user_id,

        COUNT(DISTINCT o.order_id) AS total_orders,

        COUNT(*) * 1.0 /
            COUNT(DISTINCT o.order_id) AS avg_basket_size,

        AVG(op.reordered) * 100 AS reorder_rate

    FROM orders o

    JOIN order_products op
        ON o.order_id = op.order_id

    GROUP BY o.user_id
)

SELECT
    CASE
        WHEN total_orders BETWEEN 1 AND 5 THEN '1-5'
        WHEN total_orders BETWEEN 6 AND 10 THEN '6-10'
        WHEN total_orders BETWEEN 11 AND 20 THEN '11-20'
        WHEN total_orders BETWEEN 21 AND 30 THEN '21-30'
        ELSE '31+'
    END AS order_group,

    COUNT(*) AS customers,

    ROUND(AVG(avg_basket_size), 2) AS avg_basket_size,

    ROUND(AVG(reorder_rate), 2) AS avg_reorder_rate

FROM customer_metrics

GROUP BY order_group

ORDER BY
    CASE order_group
        WHEN '1-5' THEN 1
        WHEN '6-10' THEN 2
        WHEN '11-20' THEN 3
        WHEN '21-30' THEN 4
        WHEN '31+' THEN 5
    END;


-- ============================================================
-- Q4. What are the most repeat-oriented customers?
-- ============================================================

SELECT
    o.user_id,

    COUNT(*) AS total_items,

    COUNT(DISTINCT o.order_id) AS total_orders,

    ROUND(AVG(op.reordered) * 100, 2) AS reorder_rate

FROM orders o

JOIN order_products op
    ON o.order_id = op.order_id

GROUP BY o.user_id

HAVING COUNT(DISTINCT o.order_id) >= 10

ORDER BY reorder_rate DESC

LIMIT 50;


-- ============================================================
-- Q5. What are the average order intervals?
-- ============================================================

SELECT
    CASE
        WHEN days_since_prior_order BETWEEN 0 AND 7
            THEN '0-7 days'

        WHEN days_since_prior_order BETWEEN 8 AND 14
            THEN '8-14 days'

        WHEN days_since_prior_order BETWEEN 15 AND 21
            THEN '15-21 days'

        ELSE '22-30 days'
    END AS interval_group,

    COUNT(*) AS orders

FROM orders

WHERE days_since_prior_order IS NOT NULL

GROUP BY interval_group

ORDER BY
    CASE interval_group
        WHEN '0-7 days' THEN 1
        WHEN '8-14 days' THEN 2
        WHEN '15-21 days' THEN 3
        WHEN '22-30 days' THEN 4
    END;