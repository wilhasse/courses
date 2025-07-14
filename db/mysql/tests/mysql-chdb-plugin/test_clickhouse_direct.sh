#!/bin/bash

echo "Testing ClickHouse binary directly..."
echo

# Test 1: Check if binary exists
echo "Test 1: Checking if clickhouse-local exists..."
if [ -f "/home/cslog/chdb/buildlib/programs/clickhouse-local" ]; then
    echo "✓ Binary exists"
    ls -la /home/cslog/chdb/buildlib/programs/clickhouse-local
else
    echo "✗ Binary not found!"
    exit 1
fi

echo
echo "Test 2: Testing simple query without data path..."
/home/cslog/chdb/buildlib/programs/clickhouse-local --query="SELECT 1 as test" 2>&1

echo
echo "Test 3: Testing with data path..."
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT COUNT(*) FROM mysql_import.customers" 2>&1

echo
echo "Test 4: List databases..."
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SHOW DATABASES" 2>&1

echo
echo "Test 5: List tables in mysql_import database..."
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SHOW TABLES FROM mysql_import" 2>&1

echo
echo "Test 6: Describe customers table..."
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="DESCRIBE mysql_import.customers" 2>&1

echo
echo "Test 7: Select first 5 customers..."
/home/cslog/chdb/buildlib/programs/clickhouse-local \
    --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
    --query="SELECT * FROM mysql_import.customers LIMIT 5" \
    --format=TabSeparated 2>&1
