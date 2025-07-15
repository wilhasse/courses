#!/bin/bash

# Simple test script for chDB API UDF

echo "=== Testing chDB API Connection ==="
echo

# Check if API server is running
echo "1. Checking API server on port 8125..."
if nc -z localhost 8125 2>/dev/null; then
    echo "   ✓ API server is running"
else
    echo "   ✗ API server not found!"
    echo "   Please start: cd ../mysql-to-chdb-example && ./chdb_api_server_simple -d /chdb/data/"
    exit 1
fi

# Check if function exists
echo
echo "2. Checking MySQL function..."
FUNC_EXISTS=$(mysql -u root -N -e "SELECT COUNT(*) FROM mysql.func WHERE name = 'chdb_api_query';" 2>/dev/null || echo "0")
if [ "$FUNC_EXISTS" -eq "1" ]; then
    echo "   ✓ Function chdb_api_query exists"
else
    echo "   ✗ Function not found!"
    echo "   Please run: ./scripts/install_chdb_api.sh"
    exit 1
fi

# Test queries
echo
echo "3. Running test queries..."
echo

echo "   ClickHouse version:"
mysql -u root -N -e "SELECT CAST(chdb_api_query('SELECT version()') AS CHAR);" | sed 's/^/   /'

echo
echo "   Simple math (1+1):"
mysql -u root -N -e "SELECT CAST(chdb_api_query('SELECT 1 + 1') AS CHAR);" | sed 's/^/   /'

echo
echo "   Current date:"
mysql -u root -N -e "SELECT CAST(chdb_api_query('SELECT today()') AS CHAR);" | sed 's/^/   /'

echo
echo "   Historico table count:"
mysql -u root -N -e "SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico') AS CHAR);" 2>/dev/null | sed 's/^/   /'

echo
echo "=== All tests completed ==="