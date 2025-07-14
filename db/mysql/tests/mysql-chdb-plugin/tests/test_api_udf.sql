-- Test chDB API UDF functions

-- Make sure the chDB API server is running on port 8125
-- Start it with: cd mysql-to-chdb-example && ./chdb_api_server_simple

USE test;

SELECT '=== Testing chDB API UDF Functions ===' AS test_header;

-- Test 1: Direct query
SELECT '1. Direct query test:' AS test;
SELECT chdb_query('SELECT COUNT(*) FROM mysql_import.customers') AS customer_count;

-- Test 2: Count function
SELECT '2. Count function test:' AS test;
SELECT chdb_count('mysql_import.customers') AS count_result;

-- Test 3: Query with WHERE clause
SELECT '3. Filtered query test:' AS test;
SELECT chdb_query('SELECT city, COUNT(*) as cnt FROM mysql_import.customers GROUP BY city ORDER BY cnt DESC LIMIT 3') AS top_cities;

-- Test 4: Join with MySQL table
SELECT '4. Join ClickHouse data with MySQL:' AS test;
CREATE TEMPORARY TABLE IF NOT EXISTS mysql_cities (city VARCHAR(100), country VARCHAR(100));
INSERT INTO mysql_cities VALUES 
    ('New York', 'USA'),
    ('Los Angeles', 'USA'),
    ('Chicago', 'USA');

-- Get customer count for each city from ClickHouse
SELECT 
    m.city,
    m.country,
    CAST(chdb_query(CONCAT('SELECT COUNT(*) FROM mysql_import.customers WHERE city = ''', m.city, '''')) AS UNSIGNED) AS customer_count
FROM mysql_cities m;

-- Test 5: Complex analytical query
SELECT '5. Analytical query test:' AS test;
SELECT chdb_query('
    SELECT 
        city, 
        COUNT(*) as customers,
        AVG(age) as avg_age,
        MIN(age) as min_age,
        MAX(age) as max_age
    FROM mysql_import.customers 
    GROUP BY city 
    HAVING COUNT(*) > 1
    ORDER BY customers DESC
') AS city_analytics;

-- Test 6: Error handling
SELECT '6. Error handling test:' AS test;
SELECT chdb_query('SELECT * FROM non_existent_table') AS error_test;

-- Test 7: Performance comparison
SELECT '7. Performance test:' AS test;

-- Time a simple count in MySQL (if similar data exists)
-- SET @start = NOW(6);
-- SELECT COUNT(*) FROM some_mysql_table;
-- SET @mysql_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000000.0;

-- Time the same count in ClickHouse
SET @start = NOW(6);
SELECT chdb_count('mysql_import.customers');
SET @chdb_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000000.0;

SELECT CONCAT('ClickHouse query time: ', @chdb_time, ' seconds') AS performance;

SELECT '=== All tests completed ===' AS test_footer;