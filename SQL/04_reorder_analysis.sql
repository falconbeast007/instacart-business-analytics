-- ============================================================
-- REORDER ANALYSIS
-- ============================================================


-- ============================================================
-- Q1. Overall reorder rate
-- ============================================================

SELECT
    ROUND(
        AVG(reordered) * 100,
        2
    ) AS overall_reorder_rate

FROM order_products;


-- ============================================================
-- Q2. Reorder rate by department
-- ============================================================

SELECT
    p.department,

    COUNT(*) AS purchases,

    SUM(op.reordered) AS reordered_items,

    ROUND(
        AVG(op.reordered) * 100,
        2
    ) AS reorder_rate

FROM order_products op

JOIN products p
    ON op.product_id = p.product_id

GROUP BY p.department

ORDER BY reorder_rate DESC;


-- ============================================================
-- Q3. Reorder rate by cart position
-- ============================================================

SELECT
    CASE
        WHEN add_to_cart_order BETWEEN 1 AND 3
            THEN '1-3'

        WHEN add_to_cart_order BETWEEN 4 AND 5
            THEN '4-5'

        WHEN add_to_cart_order BETWEEN 6 AND 10
            THEN '6-10'

        WHEN add_to_cart_order BETWEEN 11 AND 20
            THEN '11-20'

        ELSE '21+'
    END AS cart_position_group,

    COUNT(*) AS purchases,

    ROUND(
        AVG(reordered) * 100,
        2
    ) AS reorder_rate

FROM order_products

GROUP BY cart_position_group

ORDER BY
    CASE cart_position_group
        WHEN '1-3' THEN 1
        WHEN '4-5' THEN 2
        WHEN '6-10' THEN 3
        WHEN '11-20' THEN 4
        WHEN '21+' THEN 5
    END;


-- ============================================================
-- Q4. Products with strong repeat behavior and meaningful scale
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

HAVING
    COUNT(*) >= 1000
    AND AVG(op.reordered) >= 0.75

ORDER BY
    total_purchases DESC;


-- ============================================================
-- Q5. Department × customer maturity
-- ============================================================

WITH customer_orders AS (

    SELECT
        user_id,
        MAX(order_number) AS total_orders

    FROM orders

    GROUP BY user_id
),

customer_segments AS (

    SELECT
        user_id,

        CASE
            WHEN total_orders <= 5
                THEN 'Occasional'

            WHEN total_orders <= 10
                THEN 'Regular'

            WHEN total_orders <= 20
                THEN 'Established'

            WHEN total_orders <= 30
                THEN 'Frequent'

            ELSE 'Highly Loyal'

        END AS customer_segment

    FROM customer_orders
)

SELECT
    cs.customer_segment,

    p.department,

    COUNT(*) AS purchases,

    COUNT(DISTINCT o.user_id) AS customers,

    ROUND(
        AVG(op.reordered) * 100,
        2
    ) AS reorder_rate

FROM order_products op

JOIN orders o
    ON op.order_id = o.order_id

JOIN products p
    ON op.product_id = p.product_id

JOIN customer_segments cs
    ON o.user_id = cs.user_id

GROUP BY
    cs.customer_segment,
    p.department

ORDER BY
    cs.customer_segment,
    reorder_rate DESC;