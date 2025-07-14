#!/bin/bash

echo "=== ClickHouse TVF Test Script ==="
echo

# First check if clickhouse_data exists
if [ ! -d "/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" ]; then
    echo "ERROR: ClickHouse data directory not found!"
    echo "Please run the feed_data_v2 program first to populate the data."
    echo
    echo "Run these commands:"
    echo "  cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example"
    echo "  ./feed_data_v2"
    exit 1
fi

# Check if chDB binary exists
if [ ! -f "/home/cslog/chdb/buildlib/programs/clickhouse-local" ]; then
    echo "ERROR: chDB binary not found!"
    echo "Please build chDB first."
    exit 1
fi

# Test if we can query the data
echo "Testing ClickHouse data access..."
/home/cslog/chdb/buildlib/programs/clickhouse local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT COUNT(*) FROM mysql_import.customers" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Cannot access ClickHouse data!"
    exit 1
fi

echo "ClickHouse data is accessible."
echo

# Build the plugin
echo "Building the plugin..."
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin
./scripts/build_chdb_tvf.sh

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Install the functions
echo
echo "Installing the functions..."
./scripts/install_chdb_tvf.sh

if [ $? -ne 0 ]; then
    echo "Installation failed!"
    exit 1
fi

# Run the tests
echo
echo "Running tests..."
mysql -u root -pteste < tests/test_chdb_tvf.sql

echo
echo "=== Test Complete ==="