#!/bin/bash

# Simple installation script for chDB API UDF plugin
# This plugin connects to the chDB API server running on port 8125

set -e

echo "=== Installing chDB API UDF Plugin ==="
echo

# Check if API server is running
echo "Checking if chDB API server is running on port 8125..."
if nc -z localhost 8125 2>/dev/null; then
    echo "✓ API server is running"
else
    echo "⚠ WARNING: API server not detected on port 8125"
    echo "Please start the server with: ./chdb_api_server_simple -d /chdb/data/"
    echo
fi

# Build the plugin
echo "Building chdb_api_udf plugin..."
g++ -shared -fPIC -o chdb_api_udf.so src/chdb_api_udf.cpp -I/usr/include/mysql

# Copy to MySQL plugin directory
echo "Installing plugin to MySQL..."
sudo cp chdb_api_udf.so /usr/lib/mysql/plugin/

# Create the function
echo "Creating MySQL function..."
mysql -u root -e "DROP FUNCTION IF EXISTS chdb_api_query;"
mysql -u root -e "CREATE FUNCTION chdb_api_query RETURNS STRING SONAME 'chdb_api_udf.so';"

# Test the function
echo
echo "Testing the function..."
echo "Running: SELECT chdb_api_query('SELECT version()')"
RESULT=$(mysql -u root -N -e "SELECT CAST(chdb_api_query('SELECT version()') AS CHAR);")
echo "ClickHouse version: $RESULT"

echo
echo "=== Installation Complete! ==="
echo
echo "Usage examples:"
echo "  SELECT CAST(chdb_api_query('SELECT 1') AS CHAR);"
echo "  SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico') AS CHAR);"
echo "  SELECT CAST(chdb_api_query('SELECT today()') AS CHAR);"
echo
echo "Note: Always use CAST(... AS CHAR) to convert binary output to readable text"