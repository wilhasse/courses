-- teste MySQL chDB API Functions
-- The API server returns binary strings, so we need to CAST them

USE teste;

-- Basic teste
SELECT '=== Basic teste ===' AS teste;
SELECT CAST(chdb_query('SELECT 1') AS CHAR) AS result;

-- Count teste  
SELECT '=== Count teste ===' AS teste;
SELECT chdb_count('mysql_import.customers') AS customer_count;

-- Customer data
SELECT '=== Customer Query ===' AS teste;
SELECT CAST(chdb_query('SELECT COUNT(*) FROM mysql_import.customers') AS CHAR) AS total_customers;

-- Top cities
SELECT '=== Top Cities ===' AS teste;
SELECT CAST(chdb_query('
    SELECT city, COUNT(*) as cnt 
    FROM mysql_import.customers 
    GROUP BY city 
    ORDER BY cnt DESC 
    LIMIT 3
') AS CHAR) AS top_cities;

-- Analytics query
SELECT '=== Analytics ===' AS teste;
SELECT CAST(chdb_query('
    SELECT 
        city,
        COUNT(*) as customers,
        ROUND(AVG(age), 1) as avg_age
    FROM mysql_import.customers
    GROUP BY city
    ORDER BY customers DESC
') AS CHAR) AS city_stats;

-- Version check
SELECT '=== ClickHouse Version ===' AS teste;
SELECT CAST(chdb_query('SELECT version()') AS CHAR) AS clickhouse_version;
