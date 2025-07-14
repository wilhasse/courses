-- Test JSON table functions in MySQL 8.0.19+
-- These functions use JSON_TABLE to create true table-valued functions

USE teste;

-- Test 1: Basic JSON_TABLE usage with ClickHouse data
SELECT '=== Test 1: Basic JSON_TABLE with ClickHouse customers ===' AS test;

SELECT * FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        email VARCHAR(100) PATH '$.email',
        age INT PATH '$.age',
        city VARCHAR(100) PATH '$.city'
    )
) AS customers
LIMIT 5;

-- Test 2: Join JSON_TABLE result with MySQL table
SELECT '=== Test 2: Join ClickHouse data with MySQL using JSON_TABLE ===' AS test;

-- Create MySQL table for joining
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

-- Now join using JSON_TABLE
SELECT 
    c.id,
    c.name,
    c.city,
    mc.category,
    mc.discount_rate
FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        city VARCHAR(100) PATH '$.city'
    )
) AS c
JOIN mysql_customer_categories mc ON c.city = mc.city
ORDER BY c.id;

-- Test 3: Using generic JSON query function
SELECT '=== Test 3: Generic JSON query with custom columns ===' AS test;

SELECT * FROM JSON_TABLE(
    chdb_query_json('SELECT city, COUNT(*) as customer_count, AVG(age) as avg_age FROM mysql_import.customers GROUP BY city ORDER BY customer_count DESC'),
    '$[*]' COLUMNS (
        city VARCHAR(100) PATH '$.city',
        customer_count INT PATH '$.customer_count',
        avg_age DECIMAL(5,2) PATH '$.avg_age'
    )
) AS city_stats;

-- Test 4: Complex aggregation with JOIN
SELECT '=== Test 4: Complex aggregation across both data sources ===' AS test;

SELECT 
    mc.category,
    COUNT(*) as total_customers,
    AVG(c.age) as avg_age,
    AVG(mc.discount_rate) as avg_discount
FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        age INT PATH '$.age',
        city VARCHAR(100) PATH '$.city'
    )
) AS c
JOIN mysql_customer_categories mc ON c.city = mc.city
GROUP BY mc.category
ORDER BY total_customers DESC;

-- Test 5: Create a view using JSON_TABLE
DROP VIEW IF EXISTS v_clickhouse_customers_json;
CREATE VIEW v_clickhouse_customers_json AS
SELECT * FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        email VARCHAR(100) PATH '$.email',
        age INT PATH '$.age',
        city VARCHAR(100) PATH '$.city'
    )
) AS customers;

SELECT '=== Test 5: Using JSON_TABLE view ===' AS test;
SELECT 
    city,
    COUNT(*) as customer_count,
    AVG(age) as avg_age
FROM v_clickhouse_customers_json
GROUP BY city
ORDER BY customer_count DESC;

-- Test 6: Subquery with JSON_TABLE
SELECT '=== Test 6: Subquery example ===' AS test;

SELECT 
    category,
    customer_count,
    CASE 
        WHEN customer_count >= 2 THEN 'High Volume'
        WHEN customer_count = 1 THEN 'Medium Volume'
        ELSE 'Low Volume'
    END as volume_category
FROM (
    SELECT 
        mc.category,
        COUNT(*) as customer_count
    FROM JSON_TABLE(
        chdb_customers_json(),
        '$[*]' COLUMNS (
            city VARCHAR(100) PATH '$.city'
        )
    ) AS c
    JOIN mysql_customer_categories mc ON c.city = mc.city
    GROUP BY mc.category
) AS category_stats
ORDER BY customer_count DESC;

-- Test 7: Performance comparison
SELECT '=== Test 7: Performance comparison ===' AS test;

-- Time JSON_TABLE approach
SET @start = NOW(6);
SELECT COUNT(*) FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (id INT PATH '$.id')
) AS c;
SET @json_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;

-- Time direct query
SET @start = NOW(6);
SELECT chdb_count('mysql_import.customers');
SET @direct_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;

-- Time CTE approach (if available)
SET @start = NOW(6);
SELECT COUNT(*) FROM v_clickhouse_customers_json;
SET @view_time = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;

SELECT 
    CONCAT('JSON_TABLE time: ', @json_time, ' ms') AS json_performance,
    CONCAT('Direct query time: ', @direct_time, ' ms') AS direct_performance,
    CONCAT('View time: ', @view_time, ' ms') AS view_performance;

-- Test 8: More complex queries
SELECT '=== Test 8: Complex analysis example ===' AS test;

-- Customer segmentation with multiple data sources
WITH customer_analysis AS (
    SELECT 
        c.id,
        c.name,
        c.age,
        c.city,
        mc.category,
        mc.discount_rate,
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
    JOIN mysql_customer_categories mc ON c.city = mc.city
)
SELECT 
    age_group,
    category,
    COUNT(*) as customer_count,
    AVG(age) as avg_age,
    AVG(discount_rate) as avg_discount
FROM customer_analysis
GROUP BY age_group, category
ORDER BY age_group, customer_count DESC;
