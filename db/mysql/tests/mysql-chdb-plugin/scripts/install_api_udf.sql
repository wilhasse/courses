-- Install chDB API UDF function

-- Drop existing function if it exists
DROP FUNCTION IF EXISTS chdb_api_query;

-- Create the function
CREATE FUNCTION chdb_api_query RETURNS STRING 
    SONAME 'chdb_api_udf.so';

-- Test the functions (basic tests only)
SELECT '=== Testing chDB API UDF Functions ===' AS test_status;

-- Test 1: Basic query - Check ClickHouse version
SELECT chdb_api_query('SELECT version()') AS clickhouse_version;

-- Test 2: Simple arithmetic query
SELECT chdb_api_query('SELECT 1 + 1') AS simple_math;

-- Test 3: Current date query
SELECT chdb_api_query('SELECT today()') AS current_date;

-- Test 4: System information
SELECT chdb_api_query('SELECT hostName()') AS hostname;

SELECT '=== All functions installed successfully! ===' AS status;

-- Note: To test with actual data tables, run queries like:
-- SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico') AS CHAR);