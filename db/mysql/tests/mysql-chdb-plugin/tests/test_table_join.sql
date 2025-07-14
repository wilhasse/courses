-- Test joining ClickHouse tables with MySQL tables

USE teste;

-- Create a MySQL table with additional customer information
DROP TABLE IF EXISTS mysql_customer_categories;
CREATE TABLE mysql_customer_categories (
    city VARCHAR(100),
    category VARCHAR(50),
    discount_rate DECIMAL(3,2)
);

INSERT INTO mysql_customer_categories VALUES 
    ('New York', 'Premium', 0.15),
    ('Los Angeles', 'Standard', 0.10),
    ('Chicago', 'Premium', 0.15),
    ('Houston', 'Standard', 0.10),
    ('Phoenix', 'Basic', 0.05),
    ('San Antonio', 'Basic', 0.05),
    ('San Diego', 'Standard', 0.10),
    ('Dallas', 'Premium', 0.15),
    ('San Jose', 'Premium', 0.15),
    ('Austin', 'Standard', 0.10);

-- Test 1: Simple table-valued function simulation
SELECT '=== Test 1: Simulating ClickHouse table as rows ===' AS test;

-- Using recursive CTE to generate row numbers
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
SELECT * FROM clickhouse_customers LIMIT 5;

-- Test 2: Join ClickHouse data with MySQL table
SELECT '=== Test 2: Join ClickHouse customers with MySQL categories ===' AS test;

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
    cc.id,
    cc.name,
    cc.city,
    mc.category,
    mc.discount_rate
FROM clickhouse_customers cc
JOIN mysql_customer_categories mc ON cc.city = mc.city
ORDER BY cc.id;

-- Test 3: Aggregate data from both sources
SELECT '=== Test 3: Category statistics ===' AS test;

WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
),
clickhouse_customers AS (
    SELECT 
        chdb_customers_get_city(n) AS city
    FROM row_numbers
)
SELECT 
    mc.category,
    COUNT(*) AS customer_count,
    AVG(mc.discount_rate) AS avg_discount
FROM clickhouse_customers cc
JOIN mysql_customer_categories mc ON cc.city = mc.city
GROUP BY mc.category
ORDER BY customer_count DESC;

-- Test 4: Using generic table functions
SELECT '=== Test 4: Generic table field access ===' AS test;

-- Get specific fields from any table
SELECT 
    chdb_table_get_field('mysql_import.customers', 'name', 1) AS first_customer_name,
    chdb_table_get_field('mysql_import.customers', 'age', 1) AS first_customer_age,
    chdb_table_get_field('mysql_import.customers', 'city', 1) AS first_customer_city;

-- Test 5: Get entire row as TSV
SELECT '=== Test 5: Get full row data ===' AS test;
SELECT chdb_table_get_row('mysql_import.customers', 1) AS first_row_tsv;

-- Test 6: Create a view for easier access
DROP VIEW IF EXISTS v_clickhouse_customers;
CREATE VIEW v_clickhouse_customers AS
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
)
SELECT 
    chdb_customers_get_id(n) AS id,
    chdb_customers_get_name(n) AS name,
    chdb_customers_get_city(n) AS city
FROM row_numbers;

SELECT '=== Test 6: Using view for joins ===' AS test;
SELECT 
    v.name,
    v.city,
    m.category
FROM v_clickhouse_customers v
JOIN mysql_customer_categories m ON v.city = m.city
LIMIT 5;

-- Test 7: Performance test
SELECT '=== Test 7: Performance comparison ===' AS test;

SET @start = NOW(6);
SELECT COUNT(*) FROM v_clickhouse_customers;
SET @view_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000000.0;

SET @start = NOW(6);
SELECT chdb_count('mysql_import.customers');
SET @direct_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000000.0;

SELECT 
    CONCAT('View access time: ', @view_time, ' seconds') AS view_performance,
    CONCAT('Direct count time: ', @direct_time, ' seconds') AS direct_performance;
