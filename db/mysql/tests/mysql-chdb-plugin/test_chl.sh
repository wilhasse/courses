#!/bin/bash

echo "Testing chl (clickhouse-local) binary..."

# Test if chl exists and is executable
if [ -x "/home/cslog/chdb/buildlib/programs/chl" ]; then
    echo "✓ chl exists and is executable"
    
    # Test simple query
    echo
    echo "Test 1: Simple query"
    /home/cslog/chdb/buildlib/programs/chl --query="SELECT 'Hello from chl'" 2>&1 | head -5
    
    # Test with data path
    echo
    echo "Test 2: Query with data path"
    /home/cslog/chdb/buildlib/programs/chl \
        --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
        --query="SELECT COUNT(*) FROM mysql_import.customers" 2>&1 | head -5
    
    # Test listing databases
    echo
    echo "Test 3: Show databases"
    /home/cslog/chdb/buildlib/programs/chl \
        --path="/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data" \
        --query="SHOW DATABASES" 2>&1 | head -10
else
    echo "✗ chl not found or not executable"
    
    # Try to find what exists
    echo
    echo "Looking for any clickhouse executable..."
    find /home/cslog/chdb/buildlib/programs -name "clickhouse*" -type f -executable 2>/dev/null | head -10
    
    # Check if it's a symlink issue
    echo
    echo "Checking symlink targets..."
    ls -la /home/cslog/chdb/buildlib/programs/ch* | head -10
fi
