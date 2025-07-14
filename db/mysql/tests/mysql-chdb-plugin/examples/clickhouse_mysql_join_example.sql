-- Example: Joining ClickHouse tables with MySQL tables
-- This demonstrates a real-world scenario where ClickHouse has transaction data
-- and MySQL has reference data

USE test;

-- Step 1: Create MySQL reference tables
DROP TABLE IF EXISTS product_categories;
CREATE TABLE product_categories (
    category_id INT PRIMARY KEY,
    category_name VARCHAR(50),
    commission_rate DECIMAL(3,2)
);

INSERT INTO product_categories VALUES 
    (1, 'Electronics', 0.05),
    (2, 'Clothing', 0.08),
    (3, 'Books', 0.10),
    (4, 'Food', 0.03);

DROP TABLE IF EXISTS sales_regions;
CREATE TABLE sales_regions (
    city VARCHAR(100),
    region VARCHAR(50),
    region_manager VARCHAR(100)
);

INSERT INTO sales_regions VALUES 
    ('New York', 'Northeast', 'John Smith'),
    ('Los Angeles', 'West', 'Jane Doe'),
    ('Chicago', 'Midwest', 'Bob Johnson'),
    ('Houston', 'South', 'Alice Brown'),
    ('Phoenix', 'Southwest', 'Charlie Wilson');

-- Step 2: Create a view for ClickHouse customers table
DROP VIEW IF EXISTS v_customers;
CREATE VIEW v_customers AS
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
)
SELECT 
    chdb_customers_get_id(n) AS customer_id,
    chdb_customers_get_name(n) AS customer_name,
    chdb_customers_get_city(n) AS city
FROM row_numbers;

-- Step 3: Join ClickHouse customers with MySQL regions
SELECT 
    '=== Customer Regional Distribution ===' AS report_title;

SELECT 
    sr.region,
    sr.region_manager,
    COUNT(*) AS customer_count
FROM v_customers c
JOIN sales_regions sr ON c.city = sr.city
GROUP BY sr.region, sr.region_manager
ORDER BY customer_count DESC;

-- Step 4: Complex multi-table join example
-- Simulate getting order data from ClickHouse
SELECT 
    '=== Regional Sales Analysis ===' AS report_title;

-- Get aggregated order data from ClickHouse
SET @order_stats = (
    SELECT CAST(chdb_query('
        SELECT 
            customer_id,
            COUNT(*) as order_count,
            SUM(price * quantity) as total_revenue
        FROM mysql_import.orders
        GROUP BY customer_id
        ORDER BY customer_id
    ') AS CHAR)
);

-- For demonstration, let's show customer details with their city and region
SELECT 
    c.customer_id,
    c.customer_name,
    c.city,
    COALESCE(sr.region, 'Unknown') AS region,
    COALESCE(sr.region_manager, 'Unassigned') AS region_manager
FROM v_customers c
LEFT JOIN sales_regions sr ON c.city = sr.city
ORDER BY c.customer_id
LIMIT 10;

-- Step 5: Create a comprehensive report combining both data sources
SELECT 
    '=== Comprehensive Customer Report ===' AS report_title;

-- Calculate metrics by region
WITH customer_regions AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.city,
        COALESCE(sr.region, 'Unknown') AS region
    FROM v_customers c
    LEFT JOIN sales_regions sr ON c.city = sr.city
)
SELECT 
    region,
    COUNT(*) AS total_customers,
    COUNT(DISTINCT city) AS cities_served
FROM customer_regions
GROUP BY region
ORDER BY total_customers DESC;

-- Step 6: Performance comparison
SELECT 
    '=== Performance Metrics ===' AS report_title;

-- Time the view-based join
SET @start = NOW(6);
SELECT COUNT(*) FROM v_customers c JOIN sales_regions sr ON c.city = sr.city;
SET @view_join_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;

-- Time a direct ClickHouse query
SET @start = NOW(6);
SELECT chdb_count('mysql_import.customers');
SET @direct_query_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;

SELECT 
    CONCAT('View-based join: ', @view_join_time, ' ms') AS view_performance,
    CONCAT('Direct query: ', @direct_query_time, ' ms') AS direct_performance,
    CONCAT('Overhead: ', ROUND(@view_join_time - @direct_query_time, 2), ' ms') AS join_overhead;

-- Step 7: Practical use case - Customer segmentation
SELECT 
    '=== Customer Segmentation ===' AS report_title;

-- Create customer segments based on location
WITH customer_segments AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.city,
        CASE 
            WHEN sr.region IN ('Northeast', 'West') THEN 'Premium Markets'
            WHEN sr.region IN ('Midwest', 'South') THEN 'Growth Markets'
            ELSE 'Emerging Markets'
        END AS market_segment
    FROM v_customers c
    LEFT JOIN sales_regions sr ON c.city = sr.city
)
SELECT 
    market_segment,
    COUNT(*) AS customer_count,
    GROUP_CONCAT(DISTINCT city ORDER BY city) AS cities
FROM customer_segments
GROUP BY market_segment
ORDER BY customer_count DESC;