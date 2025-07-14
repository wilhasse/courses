-- Test ClickHouse TVF functions

-- Test 1: Check customer count
SELECT ch_customer_count() AS total_customers;

-- Test 2: Get first 5 customers using TVF pattern
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < ch_customer_count()
),
ch_customers AS (
    SELECT 
        ch_get_customer_id(n) AS id,
        ch_get_customer_name(n) AS name,
        ch_get_customer_city(n) AS city,
        ch_get_customer_age(n) AS age
    FROM numbers
    WHERE n <= 5  -- Limit to first 5 rows
)
SELECT * FROM ch_customers;

-- Test 3: Join ClickHouse data with MySQL table
-- First create a test table in MySQL
CREATE TABLE IF NOT EXISTS city_stats (
    city VARCHAR(50) PRIMARY KEY,
    population INT
);

-- Insert some sample data
INSERT IGNORE INTO city_stats VALUES 
    ('New York', 8000000),
    ('Los Angeles', 4000000),
    ('Chicago', 2700000),
    ('Houston', 2300000),
    ('Phoenix', 1600000);

-- Join ClickHouse customers with MySQL city stats
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < ch_customer_count()
),
ch_customers AS (
    SELECT 
        ch_get_customer_id(n) AS id,
        ch_get_customer_name(n) AS name,
        ch_get_customer_city(n) AS city,
        ch_get_customer_age(n) AS age
    FROM numbers
)
SELECT 
    c.id,
    c.name,
    c.city,
    c.age,
    s.population
FROM ch_customers c
LEFT JOIN city_stats s ON c.city = s.city
ORDER BY c.id;

-- Test 4: Use ch_query_scalar for aggregations
SELECT 
    'Total Customers' as metric,
    ch_query_scalar('SELECT COUNT(*) FROM mysql_import.customers') as value
UNION ALL
SELECT 
    'Average Age',
    ch_query_scalar('SELECT AVG(age) FROM mysql_import.customers')
UNION ALL
SELECT 
    'Total Orders',
    ch_query_scalar('SELECT COUNT(*) FROM mysql_import.orders')
UNION ALL
SELECT 
    'Total Revenue',
    ch_query_scalar('SELECT SUM(price * quantity) FROM mysql_import.orders');

-- Test 5: Customer count by city using ch_query_scalar
SELECT 
    ch_query_scalar('SELECT COUNT(*) FROM mysql_import.customers WHERE city = ''New York''') as new_york_customers,
    ch_query_scalar('SELECT COUNT(*) FROM mysql_import.customers WHERE city = ''Los Angeles''') as la_customers,
    ch_query_scalar('SELECT COUNT(*) FROM mysql_import.customers WHERE city = ''Chicago''') as chicago_customers;

-- Test 6: Filter customers by age
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < ch_customer_count()
),
ch_customers AS (
    SELECT 
        ch_get_customer_id(n) AS id,
        ch_get_customer_name(n) AS name,
        ch_get_customer_city(n) AS city,
        ch_get_customer_age(n) AS age
    FROM numbers
)
SELECT * FROM ch_customers
WHERE age > 35
ORDER BY age DESC;

-- Clean up
DROP TABLE IF EXISTS city_stats;