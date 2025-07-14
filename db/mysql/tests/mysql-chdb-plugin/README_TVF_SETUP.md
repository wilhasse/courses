# ClickHouse TVF Plugin Setup Guide

This guide provides step-by-step instructions to set up and test the MySQL plugin that queries ClickHouse data from the mysql-to-chdb-example project.

## Prerequisites

1. ClickHouse data must exist in `/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data`
2. chDB binary must be available at `/home/cslog/chdb/buildlib/programs/clickhouse-local`
3. MySQL server must be running with user `root` and password `teste`

## Quick Test

To verify the ClickHouse data is accessible:

```bash
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT COUNT(*) FROM mysql_import.customers"
```

## Building the Plugin

1. Navigate to the plugin directory:
```bash
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build
```

2. Compile the plugin:
```bash
g++ -shared -fPIC -o mysql_chdb_clickhouse_tvf.so \
    ../src/chdb_clickhouse_tvf.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -std=c++11
```

3. Copy to MySQL plugin directory:
```bash
sudo cp mysql_chdb_clickhouse_tvf.so /usr/lib/mysql/plugin/
```

## Installing the Functions

Run the following SQL commands in MySQL:

```sql
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
```

## Testing the Functions

### Test 1: Check customer count
```sql
SELECT ch_customer_count() AS total_customers;
```

### Test 2: Get customer details
```sql
SELECT 
    ch_get_customer_id(1) AS id,
    ch_get_customer_name(1) AS name,
    ch_get_customer_city(1) AS city,
    ch_get_customer_age(1) AS age;
```

### Test 3: Table-valued function simulation
```sql
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
```

### Test 4: Use ch_query_scalar for aggregations
```sql
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
```

## Manual Testing

You can run all tests at once:

```bash
mysql -u root -pteste < /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/test_manual.sql
```

## Troubleshooting

1. If functions fail to create, check MySQL error log:
```bash
sudo tail -f /var/log/mysql/error.log
```

2. Verify plugin is loaded:
```sql
SHOW PLUGINS;
```

3. Check function status:
```sql
SHOW FUNCTION STATUS WHERE Name LIKE 'ch_%';
```

4. Test chDB directly:
```bash
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT name, city FROM mysql_import.customers ORDER BY id LIMIT 5"
```
