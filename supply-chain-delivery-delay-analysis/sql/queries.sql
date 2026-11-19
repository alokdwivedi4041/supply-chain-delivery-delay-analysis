-- SQL analysis on data/processed/supply_chain.db  (table: orders)
-- Run:  python src/run_sql.py   (or open the .db in DB Browser for SQLite)

-- Q1. Overall KPIs
SELECT COUNT(*) AS total_orders,
       ROUND(100.0 * AVG(is_delayed), 1) AS delay_rate_pct,
       ROUND(AVG(CASE WHEN is_delayed = 1 THEN delay_days END), 2) AS avg_delay_days,
       ROUND(SUM(revenue_at_risk), 0) AS revenue_at_risk_inr
FROM orders;

-- Q2. Carrier scorecard with rank (window function)
SELECT carrier,
       COUNT(*) AS orders,
       ROUND(100.0 * AVG(is_delayed), 1) AS delay_rate_pct,
       ROUND(AVG(shipping_cost), 0) AS avg_ship_cost,
       RANK() OVER (ORDER BY AVG(is_delayed) DESC) AS worst_rank
FROM orders
WHERE carrier <> 'Unknown'
GROUP BY carrier;

-- Q3. Month-over-month delay rate with change (LAG)
WITH monthly AS (
  SELECT month, ROUND(100.0 * AVG(is_delayed), 1) AS delay_rate
  FROM orders GROUP BY month
)
SELECT month, delay_rate,
       ROUND(delay_rate - LAG(delay_rate) OVER (ORDER BY month), 1) AS mom_change
FROM monthly;

-- Q4. Worst 10 city + carrier lanes (min 30 orders)
SELECT customer_city, carrier, COUNT(*) AS orders,
       ROUND(100.0 * AVG(is_delayed), 1) AS delay_rate_pct
FROM orders
WHERE carrier <> 'Unknown'
GROUP BY customer_city, carrier
HAVING COUNT(*) >= 30
ORDER BY delay_rate_pct DESC
LIMIT 10;

-- Q5. Root-cause Pareto with cumulative share
WITH r AS (
  SELECT delay_reason, COUNT(*) AS cnt
  FROM orders WHERE is_delayed = 1 GROUP BY delay_reason
)
SELECT delay_reason, cnt,
       ROUND(100.0 * cnt / SUM(cnt) OVER (), 1) AS pct,
       ROUND(100.0 * SUM(cnt) OVER (ORDER BY cnt DESC) / SUM(cnt) OVER (), 1) AS cumulative_pct
FROM r ORDER BY cnt DESC;

-- Q6. Does delay severity hurt ratings?
SELECT delay_severity,
       COUNT(*) AS orders,
       ROUND(AVG(customer_rating), 2) AS avg_rating
FROM orders
GROUP BY delay_severity
ORDER BY avg_rating DESC;

-- Q7. Revenue at risk by product category
SELECT product_category,
       ROUND(SUM(order_value), 0) AS total_revenue,
       ROUND(SUM(revenue_at_risk), 0) AS revenue_at_risk,
       ROUND(100.0 * SUM(revenue_at_risk) / SUM(order_value), 1) AS risk_pct
FROM orders
GROUP BY product_category
ORDER BY revenue_at_risk DESC;

-- Q8. Warehouse x season delay rate
SELECT warehouse, season, ROUND(100.0 * AVG(is_delayed), 1) AS delay_rate_pct
FROM orders
GROUP BY warehouse, season
ORDER BY warehouse, delay_rate_pct DESC;
