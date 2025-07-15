#!/bin/bash

# Script to rebuild UDFs with custom result size limit

if [ $# -ne 1 ]; then
    echo "Usage: $0 <size_in_mb>"
    echo "Example: $0 20  # Sets limit to 20MB"
    exit 1
fi

SIZE_MB=$1
SIZE_BYTES=$((SIZE_MB * 1024 * 1024))

echo "=== Rebuilding UDFs with ${SIZE_MB}MB limit ==="
echo

# Update the source files
echo "Updating source files..."
sed -i "s/#define MAX_RESULT_SIZE [0-9]*/#define MAX_RESULT_SIZE $SIZE_BYTES/" src/chdb_api_udf.cpp
sed -i "s/#define MAX_RESULT_SIZE [0-9]*/#define MAX_RESULT_SIZE $SIZE_BYTES/" src/chdb_api_json_udf.cpp

# Show the change
echo "Updated MAX_RESULT_SIZE to $SIZE_BYTES bytes (${SIZE_MB}MB)"
grep "MAX_RESULT_SIZE" src/chdb_api_udf.cpp

# Remove old functions
echo
echo "Removing old functions..."
mysql -u root -e "DELETE FROM mysql.func WHERE name IN ('chdb_api_query', 'chdb_api_query_json'); FLUSH PRIVILEGES;" 2>/dev/null || true

# Restart MySQL
echo "Restarting MySQL..."
sudo systemctl restart mysql
sleep 2

# Build plugins
echo
echo "Building plugins..."
g++ -shared -fPIC -o chdb_api_udf.so src/chdb_api_udf.cpp -I/usr/include/mysql
g++ -shared -fPIC -o chdb_api_json_udf.so src/chdb_api_json_udf.cpp -I/usr/include/mysql

# Install plugins
echo "Installing plugins..."
sudo cp chdb_api_udf.so /usr/lib/mysql/plugin/
sudo cp chdb_api_json_udf.so /usr/lib/mysql/plugin/

# Create functions
echo "Creating functions..."
mysql -u root -e "CREATE FUNCTION chdb_api_query RETURNS STRING SONAME 'chdb_api_udf.so';"
mysql -u root -e "CREATE FUNCTION chdb_api_query_json RETURNS STRING SONAME 'chdb_api_json_udf.so';"

# Test
echo
echo "Testing..."
mysql -u root -e "SELECT CAST(chdb_api_query('SELECT version()') AS CHAR) AS version;"

echo
echo "=== Complete! ==="
echo "UDFs rebuilt with ${SIZE_MB}MB limit ($SIZE_BYTES bytes)"
echo
echo "Note: Keep in mind that very large results may impact MySQL performance."
echo "Recommended limits:"
echo "  - 10MB for general use"
echo "  - 50MB for large analytical queries"
echo "  - 100MB maximum (use with caution)"