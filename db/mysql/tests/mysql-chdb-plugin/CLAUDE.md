# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the MySQL chDB UDF Plugin project.

## Project Overview

This is a simple MySQL User Defined Function (UDF) plugin that integrates an AVX-optimized chDB build with MySQL. It provides a single function `chdb_query()` that executes ClickHouse SQL queries via subprocess and returns results as a string.

## Project Structure

### Core Component
- **UDF Implementation**: `src/simple_chdb_udf.cpp` - The only source file, implements the MySQL UDF

### Build System
- **CMakeLists.txt**: CMake configuration for building the plugin
- **scripts/build.sh**: Automated build script with dependency checking
- **scripts/install.sh**: UDF installation script
- **scripts/uninstall.sh**: UDF removal script

### Tests
- **tests/test_udf.sql**: Test queries for the UDF function

## How It Works

1. User calls `chdb_query('SELECT ...')` in MySQL
2. UDF executes the chDB binary via subprocess
3. Results are returned as tab-separated values in a binary string
4. Multiple rows are newline-separated
5. **Important**: Must use `CAST(... AS CHAR)` to convert binary output to readable text

## Development Workflow

### Building the Plugin
```bash
# Automated build
./scripts/build.sh

# Manual build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Installation
```bash
# Copy plugin to MySQL
sudo cp build/mysql_chdb_plugin.so /usr/lib/mysql/plugin/

# Register the UDF function
mysql -u root -p -e "CREATE FUNCTION chdb_query RETURNS STRING SONAME 'mysql_chdb_plugin.so';"
```

### Testing
```bash
# Run test queries
mysql -u root -p < tests/test_udf.sql

# Quick test (with CAST for readable output)
mysql -u root -p -e "SELECT CAST(chdb_query('SELECT version()') AS CHAR);"
```

## Code Details

### UDF Implementation (`src/simple_chdb_udf.cpp`)
- **chdb_query()**: Main UDF function that executes queries
- **chdb_query_init()**: Initialization function (sets up buffer)
- **chdb_query_deinit()**: Cleanup function
- Uses popen() to execute chDB binary
- Handles quote escaping for SQL queries
- Returns results in global buffer (64KB max)

### Key Configuration
- **CHDB_BINARY_PATH**: Hardcoded to `/home/cslog/chdb/buildlib/programs/clickhouse`
- **Output Format**: TabSeparated (TSV)
- **Buffer Size**: 65535 bytes

## Common Tasks

### Adding Features
Since this is a simple UDF, features are limited. Possible enhancements:
- Increase buffer size for larger results
- Add result format parameter
- Implement result caching

### Debugging
1. **Check MySQL error log**: `/var/log/mysql/error.log`
2. **Test chDB directly**: `/home/cslog/chdb/buildlib/programs/clickhouse local --query "SELECT 1"`
3. **Enable MySQL query log** to see exact queries

## Limitations

1. **Binary Output**: MySQL UDFs return binary strings, requiring `CAST(... AS CHAR)` for readable output
2. **Single String Output**: Not a table-valued function
3. **Buffer Size**: 64KB maximum result size
4. **Quote Handling**: String literals with quotes may cause segfaults
5. **No Type Information**: All results are strings
6. **No Streaming**: Entire result must fit in memory

## Security Considerations

- Only allows queries (no system commands)
- SQL injection possible if user input not sanitized
- Subprocess execution contained to chDB binary

## Troubleshooting

### Function Creation Fails
```sql
-- Check if plugin loaded
SHOW PLUGINS;

-- Check function status
SHOW FUNCTION STATUS WHERE Name = 'chdb_query';
```

### Segmentation Faults
- Usually caused by string literals with quotes in chDB queries
- Avoid: `SELECT 'text'` 
- Use: `SELECT toString(123)` or numeric values

### No Results or Hex Output
- If seeing hex output (e.g., `0x310A`), use `CAST(chdb_query(...) AS CHAR)`
- Check if chDB binary exists and is executable
- Verify query syntax with chDB directly
- Check MySQL user permissions

## Test Environment

- **MySQL Version**: 8.0+ (tested with Percona Server)
- **Default Credentials**: root / teste
- **Plugin Directory**: `/usr/lib/mysql/plugin/`

## Table-Valued Function (TVF) Simulation

### Overview
The project includes a demonstration of simulating table-valued functions in MySQL using multiple UDFs. Since MySQL doesn't support true TVFs without server plugin headers, we use a creative workaround.

### TVF Components

#### Source Files
- **src/test_tvf_plugin.cpp**: Implements 4 UDFs that simulate a virtual table

#### Scripts
- **scripts/build_tvf.sh**: Builds the TVF plugin
- **scripts/install_tvf.sh**: Installs TVF functions (with DROP IF EXISTS)
- **scripts/uninstall_tvf.sh**: Removes TVF functions
- **scripts/run_tvf_test.sh**: Complete test automation

#### Tests
- **tests/create_test1_table.sql**: Creates TEST1 table with sample data
- **tests/test_tvf_join.sql**: Demonstrates JOIN operations with virtual table

### TVF Functions
```sql
-- Returns row count (5)
test2_row_count()

-- Returns data for given row number (1-5)
test2_get_id(row_num)    -- Returns: 1-5
test2_get_name(row_num)  -- Returns: "Row 1", "Row 2", etc.
test2_get_value(row_num) -- Returns: row_num * 10.5
```

### How It Works
Uses recursive CTEs to generate row numbers, then calls UDFs for each column:
```sql
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
SELECT * FROM test2;
```

### Key Fixes Applied
1. **my_bool → bool**: Updated for newer MySQL versions
2. **Database context**: Added `CREATE DATABASE IF NOT EXISTS test_tvf_db`
3. **Absolute paths**: Fixed SOURCE command paths
4. **Function conflicts**: Added DROP IF EXISTS before CREATE FUNCTION

### Running TVF Tests
```bash
# Complete test with build, install, and run
./scripts/run_tvf_test.sh

# Or manually
./scripts/build_tvf.sh
./scripts/install_tvf.sh
mysql -u root < tests/test_tvf_join.sql
```

## Related Files

- **README.md**: User documentation (updated with TVF info)
- **TVF_TEST_README.md**: Detailed TVF documentation
- **CMakeLists.txt**: Build configuration
- **tests/test_udf.sql**: chDB UDF tests
- **tests/test_tvf_join.sql**: TVF simulation tests

This project demonstrates both a simple chDB integration and a creative workaround for table-valued functions in MySQL.