#!/bin/bash

# Installation script for IP-configurable chDB API UDF plugins
# These allow specifying the API server address at runtime

set -e

echo "=== Installing IP-Configurable chDB API UDF Plugins ==="
echo
echo "These functions allow you to specify the API server address:"
echo "  - chdb_api_query_remote('host:port', 'SQL')     - Query any server"
echo "  - chdb_api_query_local('SQL')                   - Query localhost:8125"
echo "  - chdb_api_query_json_remote('host:port', 'SQL') - JSON format"
echo "  - chdb_api_query_json_local('SQL')              - JSON format on localhost"
echo

# Build the plugins
echo "Building IP-configurable plugins..."
g++ -shared -fPIC -o chdb_api_ip_udf.so src/chdb_api_ip_udf.cpp -I/usr/include/mysql
g++ -shared -fPIC -o chdb_api_ip_json_udf.so src/chdb_api_ip_json_udf.cpp -I/usr/include/mysql

# Copy to MySQL plugin directory
echo "Installing plugins to MySQL..."
sudo cp chdb_api_ip_udf.so /usr/lib/mysql/plugin/
sudo cp chdb_api_ip_json_udf.so /usr/lib/mysql/plugin/

# Check if functions already exist
echo "Checking existing functions..."
EXISTING=$(mysql -u root -N -e "SELECT COUNT(*) FROM mysql.func WHERE name IN ('chdb_api_query_remote', 'chdb_api_query_local', 'chdb_api_query_json_remote', 'chdb_api_query_json_local', 'chdb_query_remote', 'chdb_query_local', 'chdb_query_json_remote', 'chdb_query_json_local');" 2>/dev/null || echo "0")

if [ "$EXISTING" -gt "0" ]; then
    echo "Some functions already exist. Removing them first..."
    mysql -u root -e "DELETE FROM mysql.func WHERE name IN ('chdb_api_query_remote', 'chdb_api_query_local', 'chdb_api_query_json_remote', 'chdb_api_query_json_local', 'chdb_query_remote', 'chdb_query_local', 'chdb_query_json_remote', 'chdb_query_json_local'); FLUSH PRIVILEGES;" 2>/dev/null || true
    echo "Restarting MySQL..."
    sudo systemctl restart mysql
    sleep 2
fi

# Create the functions
echo "Creating MySQL functions..."
mysql -u root -e "
CREATE FUNCTION chdb_api_query_remote RETURNS STRING SONAME 'chdb_api_ip_udf.so';
CREATE FUNCTION chdb_api_query_local RETURNS STRING SONAME 'chdb_api_ip_udf.so';
CREATE FUNCTION chdb_api_query_json_remote RETURNS STRING SONAME 'chdb_api_ip_json_udf.so';
CREATE FUNCTION chdb_api_query_json_local RETURNS STRING SONAME 'chdb_api_ip_json_udf.so';
"

# Test the functions
echo
echo "Testing functions..."

# Test local function (if server is running on localhost)
if nc -z localhost 8125 2>/dev/null; then
    echo "Testing localhost connection..."
    mysql -u root -e "SELECT CAST(chdb_api_query_local('SELECT version()') AS CHAR) AS version;"
else
    echo "No server detected on localhost:8125, skipping localhost test"
fi

echo
echo "=== Installation Complete! ==="
echo
echo "Usage examples:"
echo
echo "1. Query localhost server (default port 8125):"
echo "   SELECT CAST(chdb_api_query_local('SELECT 1') AS CHAR);"
echo
echo "2. Query remote server:"
echo "   SELECT CAST(chdb_api_query_remote('192.168.1.100:8125', 'SELECT 1') AS CHAR);"
echo "   SELECT CAST(chdb_api_query_remote('myserver.com:8125', 'SELECT 1') AS CHAR);"
echo
echo "3. Query with JSON format (for JSON_TABLE):"
echo "   SELECT jt.*"
echo "   FROM JSON_TABLE("
echo "     CONVERT(chdb_api_query_json_remote('192.168.1.100:8125', 'SELECT * FROM table LIMIT 10') USING utf8mb4),"
echo "     '\$.data[*]' COLUMNS (...)"
echo "   ) AS jt;"
echo
echo "4. Use different ports:"
echo "   SELECT CAST(chdb_api_query_remote('server1:8125', 'SELECT 1') AS CHAR);"
echo "   SELECT CAST(chdb_api_query_remote('server2:9000', 'SELECT 1') AS CHAR);"
echo
echo "Note: The server address can be:"
echo "  - IP address: '192.168.1.100:8125'"
echo "  - Hostname: 'myserver.local:8125'"
echo "  - Just host (uses default port 8125): '192.168.1.100'"