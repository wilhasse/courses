# MySQL chDB Plugin Solution Summary

## Current Situation

1. **libchdb.so exists** at `/home/cslog/chdb/libchdb.so` (756MB)
2. **Python module has issues** with undefined symbol `resolve_affinity`
3. **clickhouse binaries** exist as symlinks in `/home/cslog/chdb/buildlib/programs/`
4. **ClickHouse data** exists at `/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data`

## Working Solution

Since the direct approach with libchdb.so has the Python module issue, here's the working approach:

### Option 1: Build and Test with Existing Plugin

The plugin has already been built successfully (as shown in your earlier output). To complete the setup:

```bash
# 1. Install the functions in MySQL
mysql -u root -pteste << 'EOF'
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_get_customer_id;
DROP FUNCTION IF EXISTS ch_get_customer_name;
DROP FUNCTION IF EXISTS ch_get_customer_city;
DROP FUNCTION IF EXISTS ch_get_customer_age;
DROP FUNCTION IF EXISTS ch_query_scalar;

CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_id RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_name RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_city RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_get_customer_age RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';
EOF
```

### Option 2: Build with libchdb.so Directly

```bash
# Build the libchdb version
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

g++ -shared -fPIC -o mysql_chdb_tvf_libchdb.so \
    ../src/chdb_tvf_libchdb.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -ldl \
    -std=c++11

sudo cp mysql_chdb_tvf_libchdb.so /usr/lib/mysql/plugin/
```

### Option 3: Debug the Current Issue

To understand why the functions return NULL:

1. **Check if the clickhouse binary works**:
```bash
# Test with full path to see actual error
/home/cslog/chdb/buildlib/programs/clickhouse-local --version 2>&1

# Or try the chl symlink
/home/cslog/chdb/buildlib/programs/chl --version 2>&1
```

2. **Check MySQL process permissions**:
```bash
# See what user MySQL runs as
ps aux | grep mysqld

# Check if that user can access the data directory
sudo -u mysql ls -la /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data
```

3. **Create a simple test plugin** to verify basic functionality:
```cpp
// test_simple.cpp
#include <mysql.h>
#include <cstring>

extern "C" {

bool test_simple_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    return 0;
}

void test_simple_deinit(UDF_INIT *initid) {}

long long test_simple(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    return 42;
}

}
```

Build and test:
```bash
g++ -shared -fPIC -o test_simple.so test_simple.cpp $(mysql_config --cflags)
sudo cp test_simple.so /usr/lib/mysql/plugin/
mysql -u root -pteste -e "CREATE FUNCTION test_simple RETURNS INTEGER SONAME 'test_simple.so';"
mysql -u root -pteste -e "SELECT test_simple();"
```

## Most Likely Issues

1. **Permission Problem**: MySQL process (usually runs as `mysql` user) cannot access `/home/cslog/` directories
2. **Binary Path Issue**: The clickhouse binary might not be where expected
3. **Library Dependencies**: The clickhouse binary might have missing dependencies

## Quick Fix

Create a wrapper script that MySQL can execute:

```bash
# Create wrapper script
sudo tee /usr/local/bin/mysql_chdb_query.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/home/cslog/chdb:$LD_LIBRARY_PATH
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="$1" \
    --format=TabSeparated 2>/dev/null
EOF

sudo chmod +x /usr/local/bin/mysql_chdb_query.sh
sudo chmod +r /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data -R
```

Then update the plugin to use this wrapper instead of calling the binary directly.

## Testing Data Access

To verify the data is accessible:

```bash
# As your user
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT COUNT(*) FROM mysql_import.customers" 2>&1

# As mysql user (if different)
sudo -u mysql /home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT COUNT(*) FROM mysql_import.customers" 2>&1
```

The second command will likely show the permission issue if that's the problem.