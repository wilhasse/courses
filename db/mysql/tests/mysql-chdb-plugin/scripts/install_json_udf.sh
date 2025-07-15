#!/bin/bash

# Installation script for chDB API JSON UDF plugin
# This plugin automatically adds FORMAT JSON to queries

set -e

echo "=== Installing chDB API JSON UDF Plugin ==="
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
echo "Building chdb_api_json_udf plugin..."
g++ -shared -fPIC -o chdb_api_json_udf.so src/chdb_api_json_udf.cpp -I/usr/include/mysql

# Copy to MySQL plugin directory
echo "Installing plugin to MySQL..."
sudo cp chdb_api_json_udf.so /usr/lib/mysql/plugin/

# Check if function already exists
echo "Checking if function already exists..."
FUNC_EXISTS=$(mysql -u root -N -e "SELECT COUNT(*) FROM mysql.func WHERE name = 'chdb_api_query_json';" 2>/dev/null || echo "0")

if [ "$FUNC_EXISTS" -eq "1" ]; then
    echo "Function chdb_api_query_json already exists."
    echo "To reinstall, manually remove it first:"
    echo "  mysql -u root -e \"DELETE FROM mysql.func WHERE name='chdb_api_query_json'; FLUSH PRIVILEGES;\""
    echo "  sudo systemctl restart mysql"
    echo "  Then run this script again"
else
    echo "Creating MySQL function..."
    mysql -u root -e "CREATE FUNCTION chdb_api_query_json RETURNS STRING SONAME 'chdb_api_json_udf.so';"
fi

# Test the function
echo
echo "Testing the function..."
echo "Running: SELECT chdb_api_query_json('SELECT version()')"
RESULT=$(mysql -u root -N -e "SELECT chdb_api_query_json('SELECT version()');")
echo "Result (JSON format):"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

echo
echo "=== Installation Complete! ==="
echo
echo "The chdb_api_query_json function automatically adds FORMAT JSON to queries."
echo
echo "Usage examples:"
echo "  -- Simple query (returns JSON)"
echo "  SELECT chdb_api_query_json('SELECT 1 AS num, \"hello\" AS msg');"
echo
echo "  -- Query table data as JSON"
echo "  SELECT chdb_api_query_json('SELECT * FROM mysql_import.historico LIMIT 5');"
echo
echo "  -- Use with JSON_TABLE (MySQL 8.0.19+)"
echo "  SELECT jt.*"
echo "  FROM JSON_TABLE("
echo "    chdb_api_query_json('SELECT ID_CONTR, SEQ, CODIGO FROM mysql_import.historico LIMIT 5'),"
echo "    '\$.data[*]' COLUMNS ("
echo "      ID_CONTR INT PATH '\$.ID_CONTR',"
echo "      SEQ INT PATH '\$.SEQ',"
echo "      CODIGO INT PATH '\$.CODIGO'"
echo "    )"
echo "  ) AS jt;"