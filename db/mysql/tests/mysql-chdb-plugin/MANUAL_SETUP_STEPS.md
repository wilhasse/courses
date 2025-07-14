# Manual Setup Steps for MySQL ClickHouse TVF Plugin

Since the automated scripts are having issues, here are the manual steps to set up and test the MySQL plugin that queries ClickHouse data:

## Step 1: Verify ClickHouse Data Access

Test if you can access the ClickHouse data directly:

```bash
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT COUNT(*) FROM mysql_import.customers"
```

Expected output: Should return a count (e.g., 100)

## Step 2: Build the Plugin

Navigate to the build directory and compile:

```bash
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

# Compile the plugin
g++ -shared -fPIC -o mysql_chdb_clickhouse_tvf.so \
    ../src/chdb_clickhouse_tvf.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -std=c++11

# Copy to MySQL plugin directory
sudo cp mysql_chdb_clickhouse_tvf.so /usr/lib/mysql/plugin/
```

## Step 3: Install Functions in MySQL

Connect to MySQL and run:

```bash
mysql -u root -pteste
```

Then execute these SQL commands:

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

## Step 4: Test the Functions

Run these test queries in MySQL:

### Test 1: Check customer count
```sql
SELECT ch_customer_count() AS total_customers;
```

### Test 2: Get first customer details
```sql
SELECT 
    ch_get_customer_id(1) AS id,
    ch_get_customer_name(1) AS name,
    ch_get_customer_city(1) AS city,
    ch_get_customer_age(1) AS age;
```

### Test 3: Simulate table-valued function with first 5 customers
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
    WHERE n <= 5
)
SELECT * FROM ch_customers;
```

### Test 4: Run aggregation queries
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

## Troubleshooting

If you encounter issues:

1. Check MySQL error log:
```bash
sudo tail -f /var/log/mysql/error.log
```

2. Verify the plugin is loaded:
```sql
SHOW PLUGINS;
```

3. Check function status:
```sql
SHOW FUNCTION STATUS WHERE Name LIKE 'ch_%';
```

4. Test the modified C++ code compiles:
```bash
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build
g++ -c ../src/chdb_clickhouse_tvf.cpp $(mysql_config --cflags) -std=c++11
```

## What This Plugin Does

The plugin creates MySQL UDFs that:
1. Execute ClickHouse queries on the data created by mysql-to-chdb-example
2. Return results that can be used in MySQL queries
3. Simulate table-valued functions using recursive CTEs

This allows you to query ClickHouse data directly from MySQL, combining the analytical power of ClickHouse with MySQL's query capabilities.