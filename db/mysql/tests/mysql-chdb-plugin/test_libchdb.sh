#!/bin/bash

echo "Testing libchdb.so directly..."
echo

# Test 1: Check if libchdb.so exists
echo "Test 1: Checking if libchdb.so exists..."
if [ -f "/home/cslog/chdb/libchdb.so" ]; then
    echo "✓ libchdb.so exists"
    ls -la /home/cslog/chdb/libchdb.so
else
    echo "✗ libchdb.so not found!"
    echo "Build it with: cd /home/cslog/chdb && make buildlib"
    exit 1
fi

# Test 2: Check if buildlib directory exists
echo
echo "Test 2: Checking buildlib directory..."
if [ -d "/home/cslog/chdb/buildlib" ]; then
    echo "✓ buildlib directory exists"
    echo "Contents of buildlib/programs:"
    ls -la /home/cslog/chdb/buildlib/programs/ | head -20
else
    echo "✗ buildlib directory not found!"
fi

# Test 3: Python test of chDB
echo
echo "Test 3: Testing chDB with Python..."
python3 -c "
import sys
sys.path.insert(0, '/home/cslog/chdb')
try:
    import chdb
    # Test simple query
    result = chdb.query('SELECT 1 as test', 'CSV')
    print('✓ Simple query works:', result)
    
    # Test with data path
    result = chdb.query(
        \"SELECT COUNT(*) FROM mysql_import.customers\",
        'CSV',
        path='/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data'
    )
    print('✓ Query with data path:', result)
except Exception as e:
    print('✗ Error:', e)
" 2>&1

# Test 4: Check chdb command
echo
echo "Test 4: Testing chdb command-line..."
if command -v chdb &> /dev/null; then
    echo "✓ chdb command found"
    chdb "SELECT 'Hello from chdb'" Pretty 2>&1 | head -5
else
    # Try running it directly
    if [ -f "/home/cslog/chdb/chdb/__main__.py" ]; then
        echo "Running chdb via Python module..."
        cd /home/cslog/chdb
        python3 -m chdb "SELECT 'Hello from chdb'" Pretty 2>&1 | head -5
    fi
fi
