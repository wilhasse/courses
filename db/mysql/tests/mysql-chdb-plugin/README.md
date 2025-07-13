# MySQL chDB UDF Plugin

A simple MySQL User Defined Function (UDF) plugin that integrates your AVX-optimized chDB build with MySQL, allowing you to execute ClickHouse queries directly from MySQL.

## Overview

This plugin provides a simple UDF function `chdb_query()` that executes ClickHouse SQL queries using your custom-built chDB binary and returns the results as a string.

```sql
-- Execute chDB queries from MySQL
SELECT chdb_query('SELECT version()');
-- Returns: 25.5.2.1

-- Run ClickHouse queries
SELECT chdb_query('SELECT count(*), sum(number) FROM numbers(1000)');
-- Returns: 1000	499500
```

## Features

- ✅ **AVX-Optimized**: Uses your custom-built chDB with AVX/AVX2 optimizations
- ✅ **Simple Integration**: Single UDF function to execute ClickHouse SQL
- ✅ **Direct Execution**: Subprocess-based execution with efficient data transfer
- ✅ **Error Handling**: Returns error messages for invalid queries
- ✅ **Lightweight**: Single source file, minimal dependencies

## Architecture

```
MySQL Query → chdb_query() UDF → subprocess → chDB binary → Results as string
```

## Prerequisites

- **MySQL 8.0+** with development headers
- **Your AVX-optimized chDB build** at `/home/cslog/chdb/buildlib/programs/clickhouse`
- **C++ compiler** with C++20 support
- **CMake 3.20+**

## Quick Start

### 1. Clone and Build

```bash
# Clone the repository
git clone <repository-url>
cd mysql-chdb-plugin

# Build the plugin
./scripts/build.sh
```

### 2. Install

```bash
# Install the UDF plugin
sudo cp build/mysql_chdb_plugin.so /usr/lib/mysql/plugin/

# Register the function in MySQL
mysql -u root -p -e "CREATE FUNCTION chdb_query RETURNS STRING SONAME 'mysql_chdb_plugin.so';"
```

### 3. Test

```bash
# Test the function
mysql -u root -p -e "SELECT CAST(chdb_query('SELECT 1') AS CHAR) AS result;"
```

## Usage Examples

### Important: Handling Binary Output

MySQL UDFs return binary strings by default, which appear as hex (e.g., `0x310A`). To get readable output, use `CAST` or `CONVERT`:

```sql
-- Use CAST for readable output
SELECT CAST(chdb_query('SELECT 1 as num') AS CHAR) AS result;

-- Or use CONVERT with UTF8
SELECT CONVERT(chdb_query('SELECT version()') USING utf8mb4) AS version;
```

### Basic Queries

```sql
-- Simple query (with CAST for readable output)
SELECT CAST(chdb_query('SELECT 1 as num') AS CHAR);
-- Returns: 1

-- Version check
SELECT CAST(chdb_query('SELECT version()') AS CHAR);
-- Returns: 25.5.2.1

-- Math operations (AVX optimized)
SELECT CAST(chdb_query('SELECT sqrt(16), power(2, 8)') AS CHAR);
-- Returns: 4	256
```

### Aggregations and Analytics

```sql
-- Generate and aggregate numbers
SELECT CAST(chdb_query('SELECT count(*), sum(number), avg(number) FROM numbers(100)') AS CHAR);
-- Returns: 100	4950	49.5

-- Array operations
SELECT CAST(chdb_query('SELECT arraySum([1, 2, 3, 4, 5])') AS CHAR);
-- Returns: 15
```

### Working with Results

Since the UDF returns tab-separated values, you can parse them in your application:

```sql
-- Multiple columns are tab-separated
SELECT CAST(chdb_query('SELECT 1 as id, 2 as value') AS CHAR);
-- Returns: 1	2

-- Multiple rows are newline-separated
SELECT CAST(chdb_query('SELECT number FROM numbers(3)') AS CHAR);
-- Returns: 0
--         1
--         2
```

## Limitations

1. **String Output**: Returns results as a single string (tab-separated for columns, newline-separated for rows)
2. **No Table Output**: This is a scalar UDF, not a table-valued function
3. **Quote Handling**: String literals with quotes may cause issues (chDB binary limitation)
4. **Buffer Size**: Maximum result size is 64KB

## Configuration

The chDB binary path is hardcoded in the source. To change it:

1. Edit `src/simple_chdb_udf.cpp`
2. Update the `CHDB_BINARY_PATH` definition
3. Rebuild the plugin

## Troubleshooting

### Function Not Found

```sql
-- Check if function exists
SHOW FUNCTION STATUS WHERE Name = 'chdb_query';

-- Re-create if needed
DROP FUNCTION IF EXISTS chdb_query;
CREATE FUNCTION chdb_query RETURNS STRING SONAME 'mysql_chdb_plugin.so';
```

### Permission Issues

```bash
# Ensure plugin file has correct permissions
sudo chmod 755 /usr/lib/mysql/plugin/mysql_chdb_plugin.so
```

### chDB Execution Errors

```bash
# Test chDB binary directly
/home/cslog/chdb/buildlib/programs/clickhouse local --query "SELECT 1"
```

## Build from Source

```bash
# Clean build
rm -rf build
mkdir build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make

# Plugin will be at: build/mysql_chdb_plugin.so
```

## Table-Valued Function (TVF) Simulation

This project also includes a demonstration of simulating table-valued functions in MySQL using multiple UDFs. See [TVF_TEST_README.md](TVF_TEST_README.md) for details.

### Quick TVF Test

```bash
# Run the complete TVF test
./scripts/run_tvf_test.sh
```

## Uninstall

```sql
-- Remove chDB function from MySQL
DROP FUNCTION IF EXISTS chdb_query;

-- Remove TVF simulation functions
DROP FUNCTION IF EXISTS test2_row_count;
DROP FUNCTION IF EXISTS test2_get_id;
DROP FUNCTION IF EXISTS test2_get_name;
DROP FUNCTION IF EXISTS test2_get_value;
```

```bash
# Remove plugin files
sudo rm /usr/lib/mysql/plugin/mysql_chdb_plugin.so
sudo rm /usr/lib/mysql/plugin/test_tvf_plugin.so
```

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]