#!/bin/bash

echo "=== Diagnosing MySQL chDB API Connection Issue ==="
echo

# Check if API server is running
echo "1. Checking if API server is running..."
if nc -z 127.0.0.1 8125 2>/dev/null; then
    echo "   ✓ Server is listening on port 8125"
else
    echo "   ✗ Server is NOT running on port 8125"
    echo "   Start it with:"
    echo "     cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example"
    echo "     ./chdb_api_server_simple"
    exit 1
fi

# Build simple test client
echo
echo "2. Building test client..."
cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example
g++ -o test_simple_client test_simple_client.cpp

# Test direct connection
echo
echo "3. Testing direct connection to API server..."
./test_simple_client "SELECT 1"

# Check MySQL function status
echo
echo "4. Checking MySQL UDF status..."
mysql -u root -pteste -e "SHOW FUNCTION STATUS WHERE Name LIKE 'chdb%'" 2>/dev/null

# Build UDF with debug
echo
echo "5. Building UDF with debug logging..."
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin
mkdir -p build
cd build
g++ -Wall -O2 -fPIC -shared -I/usr/include/mysql ../src/chdb_api_functions.cpp -o chdb_api_functions.so

# Copy and install
echo
echo "6. Installing UDF..."
sudo cp chdb_api_functions.so /usr/lib/mysql/plugin/

# Clear debug log
> /tmp/mysql_chdb_api_debug.log

# Test MySQL function
echo
echo "7. Testing MySQL function..."
mysql -u root -pteste -e "SELECT chdb_query('SELECT 1') AS result" 2>/dev/null

# Show debug log
echo
echo "8. Debug log contents:"
echo "---------------------"
if [ -f /tmp/mysql_chdb_api_debug.log ]; then
    cat /tmp/mysql_chdb_api_debug.log
else
    echo "No debug log found"
fi

echo
echo "To monitor debug log in real-time:"
echo "  tail -f /tmp/mysql_chdb_api_debug.log"