-- MySQL chDB UDF Plugin Test Queries
-- Run these queries to test the chDB UDF plugin

-- 1. Basic functionality test
SELECT 'Test 1: Basic functionality' as test_name;
SELECT CAST(chdb_query('SELECT 1 as number') AS CHAR) AS result;

-- 2. Version information
SELECT 'Test 2: chDB version' as test_name;
SELECT CAST(chdb_query('SELECT version()') AS CHAR) AS chdb_version;

-- 3. Math operations (should benefit from AVX optimizations)
SELECT 'Test 3: Math operations (AVX optimized)' as test_name;
SELECT CAST(chdb_query('SELECT sqrt(16) as square_root, power(2, 8) as power_op') AS CHAR) AS math_result;

-- 4. Number generation
SELECT 'Test 4: Number generation' as test_name;
SELECT CAST(chdb_query('SELECT number FROM numbers(5)') AS CHAR) AS numbers;

-- 5. Aggregations
SELECT 'Test 5: Aggregations' as test_name;
SELECT CAST(chdb_query('SELECT count(*), sum(number), avg(number) FROM numbers(100)') AS CHAR) AS agg_result;

-- 6. Array operations
SELECT 'Test 6: Array operations' as test_name;
SELECT CAST(chdb_query('SELECT [1, 2, 3, 4, 5] as arr, arraySum([1, 2, 3, 4, 5]) as sum') AS CHAR) AS array_result;

-- 7. Date functions
SELECT 'Test 7: Date functions' as test_name;
SELECT CAST(chdb_query('SELECT today() as today, now() as current_time') AS CHAR) AS date_result;

-- 8. String functions (without quotes to avoid segfault)
SELECT 'Test 8: String operations' as test_name;
SELECT CAST(chdb_query('SELECT length(toString(123456)) as len, upper(toString(123)) as upper_str') AS CHAR) AS string_result;

-- 9. Error handling test
SELECT 'Test 9: Error handling' as test_name;
SELECT CAST(chdb_query('INVALID SQL QUERY') AS CHAR) AS error_result;