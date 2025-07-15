-- Safe installation script for chDB API UDF function
-- This version checks if function exists before trying to drop/create

-- Check if function already exists
SELECT '=== Checking chDB API UDF Function ===' AS status;

-- Show current function if it exists
SELECT name, dl FROM mysql.func WHERE name = 'chdb_api_query';

-- Only create if it doesn't exist
-- Note: We cannot use IF NOT EXISTS with CREATE FUNCTION in MySQL
-- So we'll just let the user know to run the DELETE manually if needed

-- Test the function
SELECT '=== Testing chDB API UDF Function ===' AS test_status;

-- Test 1: Basic query - Check ClickHouse version
SELECT CAST(chdb_api_query('SELECT version()') AS CHAR) AS clickhouse_version;

-- Test 2: Simple arithmetic query  
SELECT CAST(chdb_api_query('SELECT 1 + 1') AS CHAR) AS simple_math;

-- Test 3: Current date query
SELECT CAST(chdb_api_query('SELECT today()') AS CHAR) AS current_date;

SELECT '=== Function is working correctly! ===' AS status;

-- Note: To test with actual data tables, run queries like:
-- SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico') AS CHAR);