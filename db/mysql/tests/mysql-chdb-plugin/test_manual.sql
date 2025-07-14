-- Manual test for ClickHouse TVF functions
-- First, let's check if the ClickHouse data is accessible via command line

-- Build the plugin (run these commands in shell):
-- cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build
-- g++ -shared -fPIC -o mysql_chdb_clickhouse_tvf.so ../src/chdb_clickhouse_tvf.cpp $(mysql_config --cflags) $(mysql_config --libs) -std=c++11
-- sudo cp mysql_chdb_clickhouse_tvf.so /usr/lib/mysql/plugin/

-- Drop existing functions if any
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_get_customer_id;
DROP FUNCTION IF EXISTS ch_get_customer_name;
DROP FUNCTION IF EXISTS ch_get_customer_city;
DROP FUNCTION IF EXISTS ch_get_customer_age;
DROP FUNCTION IF EXISTS ch_query_scalar;

-- Create the functions
CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_id RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_name RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_city RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_age RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';

-- Test 1: Check customer count
SELECT ch_customer_count() AS total_customers;

-- Test 2: Get first customer details
SELECT 
    ch_get_customer_id(1) AS id,
    ch_get_customer_name(1) AS name,
    ch_get_customer_city(1) AS city,
    ch_get_customer_age(1) AS age;

-- Test 3: Use ch_query_scalar for aggregations
SELECT 
    'Total Customers' as metric,
    ch_query_scalar('SELECT COUNT(*) FROM mysql_import.customers') as value
UNION ALL
SELECT 
    'Average Age',
    ch_query_scalar('SELECT AVG(age) FROM mysql_import.customers');
