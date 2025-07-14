# MySQL chDB Plugin Using libchdb.so

This guide explains how to build and use the MySQL plugin that queries ClickHouse data using libchdb.so instead of a standalone ClickHouse binary.

## Understanding the Architecture

chDB is an embedded SQL OLAP engine that runs in-process. It doesn't use a separate ClickHouse binary like `clickhouse-local`. Instead, it provides:

1. **libchdb.so** - A shared library that embeds the ClickHouse engine
2. **Python module** - Python bindings for easy use
3. **C API** - For integration with other languages

## Prerequisites

1. Build libchdb.so:
```bash
cd /home/cslog/chdb
make buildlib
```

2. Verify libchdb.so exists:
```bash
ls -la /home/cslog/chdb/libchdb.so
```

3. Ensure ClickHouse data exists:
```bash
ls -la /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data/
```

## Building the Plugin

1. Make the build script executable:
```bash
chmod +x build_libchdb_tvf.sh
```

2. Run the build script:
```bash
./build_libchdb_tvf.sh
```

This will:
- Compile the plugin using libchdb.so
- Copy it to MySQL's plugin directory
- Show installation instructions

## Installing the Functions

Run the following in MySQL:

```sql
-- Drop existing functions if any
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_get_customer_id;
DROP FUNCTION IF EXISTS ch_get_customer_name;
DROP FUNCTION IF EXISTS ch_get_customer_city;
DROP FUNCTION IF EXISTS ch_get_customer_age;
DROP FUNCTION IF EXISTS ch_query_scalar;

-- Create new functions
CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_tvf_libchdb.so';
CREATE FUNCTION ch_get_customer_id RETURNS INTEGER SONAME 'mysql_chdb_tvf_libchdb.so';
CREATE FUNCTION ch_get_customer_name RETURNS STRING SONAME 'mysql_chdb_tvf_libchdb.so';
CREATE FUNCTION ch_get_customer_city RETURNS STRING SONAME 'mysql_chdb_tvf_libchdb.so';
CREATE FUNCTION ch_get_customer_age RETURNS INTEGER SONAME 'mysql_chdb_tvf_libchdb.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_tvf_libchdb.so';
```

## Testing the Functions

```sql
-- Test 1: Get customer count
SELECT ch_customer_count();

-- Test 2: Get first customer
SELECT 
    ch_get_customer_id(1) AS id,
    ch_get_customer_name(1) AS name,
    ch_get_customer_city(1) AS city,
    ch_get_customer_age(1) AS age;

-- Test 3: Query scalar values
SELECT ch_query_scalar('SELECT COUNT(*) FROM mysql_import.customers');
SELECT ch_query_scalar('SELECT AVG(age) FROM mysql_import.customers');

-- Test 4: Simulate table-valued function
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 5
),
ch_customers AS (
    SELECT 
        ch_get_customer_id(n) AS id,
        ch_get_customer_name(n) AS name,
        ch_get_customer_city(n) AS city,
        ch_get_customer_age(n) AS age
    FROM numbers
)
SELECT * FROM ch_customers;
```

## How It Works

1. The plugin uses `dlopen()` to dynamically load libchdb.so
2. It gets function pointers to `query_stable_v2` and `free_result_v2`
3. Queries are executed by passing arguments similar to command-line usage
4. Results are returned as tab-separated values

## Troubleshooting

### Test libchdb.so directly:
```bash
chmod +x test_libchdb.sh
./test_libchdb.sh
```

### Check MySQL error log:
```bash
sudo tail -f /var/log/mysql/error.log
```

### Verify plugin is loaded:
```sql
SHOW PLUGINS;
```

### Test with Python:
```python
import sys
sys.path.insert(0, '/home/cslog/chdb')
import chdb

# Test query
result = chdb.query(
    "SELECT COUNT(*) FROM mysql_import.customers",
    'CSV',
    path='/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data'
)
print(result)
```

## Key Differences from Previous Approach

1. **No clickhouse binary needed** - Uses libchdb.so instead
2. **In-process execution** - Runs within MySQL process
3. **Direct library calls** - More efficient than subprocess execution
4. **Same query syntax** - ClickHouse SQL remains the same

## API Functions Used

- `query_stable_v2(argc, argv)` - Execute a query with command-line style arguments
- `free_result_v2(result)` - Free the result structure

The arguments passed are:
- `clickhouse` (program name)
- `--multiquery`
- `--output-format=TabSeparated`
- `--path=/path/to/data`
- `--query=SELECT ...`
