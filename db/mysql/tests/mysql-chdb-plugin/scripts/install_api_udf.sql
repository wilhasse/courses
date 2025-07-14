-- Install chDB API UDF functions

-- Drop existing functions if they exist
DROP FUNCTION IF EXISTS chdb_query;
DROP FUNCTION IF EXISTS chdb_count;
DROP FUNCTION IF EXISTS chdb_sum;

-- Create the functions
CREATE FUNCTION chdb_query RETURNS STRING 
    SONAME 'chdb_api_functions.so';

CREATE FUNCTION chdb_count RETURNS INTEGER 
    SONAME 'chdb_api_functions.so';

CREATE FUNCTION chdb_sum RETURNS REAL 
    SONAME 'chdb_api_functions.so';

-- Test the functions
SELECT '=== Testing chDB API UDF Functions ===' AS test_status;

-- Test 1: Basic query
SELECT chdb_query('SELECT version()') AS clickhouse_version;

-- Test 2: Count function
SELECT chdb_count('mysql_import.customers') AS customer_count;

-- Test 3: Sum function (if orders table exists)
-- SELECT chdb_sum('mysql_import.orders', 'price') AS total_price;

-- Test 4: Custom query
SELECT chdb_query('SELECT COUNT(*) FROM mysql_import.customers WHERE city = ''New York''') AS ny_customers;

SELECT '=== All functions installed successfully! ===' AS status;