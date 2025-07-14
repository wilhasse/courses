-- Comparison: JSON_TABLE vs Recursive CTE approach
-- Demonstrates the advantages of JSON_TABLE for table-valued functions

USE test;

-- Setup: Create a MySQL table for joining
DROP TABLE IF EXISTS region_info;
CREATE TABLE region_info (
    city VARCHAR(100),
    region VARCHAR(50),
    timezone VARCHAR(50)
);

INSERT INTO region_info VALUES 
    ('New York', 'Northeast', 'EST'),
    ('Los Angeles', 'West Coast', 'PST'),
    ('Chicago', 'Midwest', 'CST'),
    ('Houston', 'South', 'CST'),
    ('Phoenix', 'Southwest', 'MST');

SELECT '=== METHOD 1: JSON_TABLE Approach (MySQL 8.0.19+) ===' AS comparison;

-- Single clean query using JSON_TABLE
SELECT 
    c.id,
    c.name,
    c.city,
    r.region,
    r.timezone
FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        city VARCHAR(100) PATH '$.city'
    )
) AS c
LEFT JOIN region_info r ON c.city = r.city
ORDER BY c.id;

SELECT '=== METHOD 2: Recursive CTE Approach (Older MySQL) ===' AS comparison;

-- Complex recursive CTE approach
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
),
clickhouse_customers AS (
    SELECT 
        chdb_customers_get_id(n) AS id,
        chdb_customers_get_name(n) AS name,
        chdb_customers_get_city(n) AS city
    FROM row_numbers
)
SELECT 
    c.id,
    c.name,
    c.city,
    r.region,
    r.timezone
FROM clickhouse_customers c
LEFT JOIN region_info r ON c.city = r.city
ORDER BY c.id;

SELECT '=== PERFORMANCE COMPARISON ===' AS comparison;

-- Test JSON_TABLE performance
SET @start = NOW(6);
SELECT COUNT(*) FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (id INT PATH '$.id')
) AS c;
SET @json_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;

-- Test CTE performance (if functions are available)
SET @start = NOW(6);
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
)
SELECT COUNT(*) FROM row_numbers;
SET @cte_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;

SELECT 
    CONCAT('JSON_TABLE time: ', @json_time, ' ms') AS json_performance,
    CONCAT('CTE time: ', @cte_time, ' ms') AS cte_performance,
    CONCAT('Performance gain: ', ROUND(@cte_time / @json_time, 1), 'x faster') AS improvement;

SELECT '=== ADVANCED JSON_TABLE EXAMPLES ===' AS comparison;

-- Example 1: Dynamic ClickHouse aggregations as tables
SELECT 
    month,
    order_count,
    total_revenue
FROM JSON_TABLE(
    chdb_query_json('
        SELECT 
            toMonth(order_date) as month,
            COUNT(*) as order_count,
            SUM(price * quantity) as total_revenue
        FROM mysql_import.orders
        GROUP BY month
        ORDER BY month
    '),
    '$[*]' COLUMNS (
        month INT PATH '$.month',
        order_count INT PATH '$.order_count',
        total_revenue DECIMAL(10,2) PATH '$.total_revenue'
    )
) AS monthly_stats;

-- Example 2: Complex JOIN with multiple data sources
SELECT 
    r.region,
    COUNT(DISTINCT c.id) as customer_count,
    AVG(c.age) as avg_age,
    SUM(stats.total_orders) as total_orders
FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        age INT PATH '$.age',
        city VARCHAR(100) PATH '$.city'
    )
) AS c
LEFT JOIN region_info r ON c.city = r.city
LEFT JOIN JSON_TABLE(
    chdb_query_json('
        SELECT 
            customer_id,
            COUNT(*) as total_orders
        FROM mysql_import.orders
        GROUP BY customer_id
    '),
    '$[*]' COLUMNS (
        customer_id INT PATH '$.customer_id',
        total_orders INT PATH '$.total_orders'
    )
) AS stats ON c.id = stats.customer_id
WHERE r.region IS NOT NULL
GROUP BY r.region
ORDER BY customer_count DESC;

-- Example 3: Creating a materialized view for performance
DROP VIEW IF EXISTS v_customer_analysis;
CREATE VIEW v_customer_analysis AS
SELECT 
    c.id,
    c.name,
    c.age,
    c.city,
    r.region,
    r.timezone,
    CASE 
        WHEN c.age < 30 THEN 'Young'
        WHEN c.age < 50 THEN 'Middle'
        ELSE 'Senior'
    END as age_group
FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        age INT PATH '$.age',
        city VARCHAR(100) PATH '$.city'
    )
) AS c
LEFT JOIN region_info r ON c.city = r.city;

-- Use the view like any regular table
SELECT 
    region,
    age_group,
    COUNT(*) as customer_count
FROM v_customer_analysis
WHERE region IS NOT NULL
GROUP BY region, age_group
ORDER BY region, age_group;

SELECT '=== SUMMARY ===' AS comparison;

SELECT 
    'JSON_TABLE Benefits:' as aspect,
    '✓ Single API call, ✓ Clean syntax, ✓ Better performance, ✓ Native MySQL' as advantages
UNION ALL
SELECT 
    'CTE Benefits:',
    '✓ Works on older MySQL, ✓ More granular control'
UNION ALL
SELECT 
    'Recommendation:',
    'Use JSON_TABLE for MySQL 8.0.19+, CTE for older versions';