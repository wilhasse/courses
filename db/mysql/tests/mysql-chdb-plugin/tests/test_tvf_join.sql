-- Test UDFs simulating Table-Valued Function with JOIN operations

-- Create test database if it doesn't exist
CREATE DATABASE IF NOT EXISTS test_tvf_db;
USE test_tvf_db;

-- First, ensure TEST1 table is created with data
SOURCE /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/tests/create_test1_table.sql;

-- Test 1: Test individual UDF functions
SELECT '=== Test 1: Test individual UDF functions ===' AS test_description;
SELECT test2_row_count() AS total_rows;
SELECT test2_get_id(1) AS id_1, test2_get_name(1) AS name_1, test2_get_value(1) AS value_1;
SELECT test2_get_id(3) AS id_3, test2_get_name(3) AS name_3, test2_get_value(3) AS value_3;

-- Test 2: Generate all rows using a numbers table
SELECT '=== Test 2: Generate virtual TEST2 table ===' AS test_description;
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < test2_row_count()
)
SELECT 
    test2_get_id(n) AS id,
    test2_get_name(n) AS name,
    test2_get_value(n) AS value
FROM numbers;

-- Test 3: Simple JOIN between TEST1 and virtual TEST2
SELECT '=== Test 3: JOIN TEST1 with virtual TEST2 ===' AS test_description;
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < test2_row_count()
),
test2 AS (
    SELECT 
        test2_get_id(n) AS id,
        test2_get_name(n) AS name,
        test2_get_value(n) AS value
    FROM numbers
)
SELECT 
    t1.id,
    t1.category,
    t1.amount,
    t2.name AS test2_name,
    t2.value AS test2_value
FROM TEST1 t1
JOIN test2 t2 ON t1.id = t2.id;

-- Test 4: LEFT JOIN to show all TEST1 records
SELECT '=== Test 4: LEFT JOIN TEST1 with virtual TEST2 ===' AS test_description;
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < test2_row_count()
),
test2 AS (
    SELECT 
        test2_get_id(n) AS id,
        test2_get_name(n) AS name,
        test2_get_value(n) AS value
    FROM numbers
)
SELECT 
    t1.id,
    t1.category,
    t1.amount,
    COALESCE(t2.name, 'No match') AS test2_name,
    COALESCE(t2.value, 0) AS test2_value
FROM TEST1 t1
LEFT JOIN test2 t2 ON t1.id = t2.id;

-- Test 5: Aggregate query with JOIN
SELECT '=== Test 5: Aggregate with JOIN ===' AS test_description;
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < test2_row_count()
),
test2 AS (
    SELECT 
        test2_get_id(n) AS id,
        test2_get_name(n) AS name,
        test2_get_value(n) AS value
    FROM numbers
)
SELECT 
    t1.category,
    COUNT(*) as count,
    SUM(t1.amount) as total_amount,
    AVG(t2.value) as avg_test2_value
FROM TEST1 t1
JOIN test2 t2 ON t1.id = t2.id
GROUP BY t1.category;

-- Test 6: Complex query with WHERE clause
SELECT '=== Test 6: JOIN with WHERE clause ===' AS test_description;
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < test2_row_count()
),
test2 AS (
    SELECT 
        test2_get_id(n) AS id,
        test2_get_name(n) AS name,
        test2_get_value(n) AS value
    FROM numbers
)
SELECT 
    t1.id,
    t1.category,
    t1.amount,
    t2.name,
    t2.value,
    (t1.amount + t2.value) AS combined_value
FROM TEST1 t1
JOIN test2 t2 ON t1.id = t2.id
WHERE t1.amount > 100 AND t2.value > 20;

-- Test 7: Demonstrate UDF error handling
SELECT '=== Test 7: UDF error handling ===' AS test_description;
SELECT 
    test2_get_id(0) AS id_0,  -- Should return NULL (out of range)
    test2_get_id(6) AS id_6,  -- Should return NULL (out of range)
    test2_get_name(10) AS name_10,  -- Should return NULL
    test2_get_value(-1) AS value_neg1;  -- Should return NULL