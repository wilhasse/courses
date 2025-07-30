#!/bin/bash

# Set LD_LIBRARY_PATH for LMDB
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH

echo "MySQL Passthrough Test Script"
echo "================================"

# Function to test connection (default port)
test_connection() {
    local port=${1:-3306}
    echo "Testing connection to MySQL server on port $port..."
    mysql -h localhost -P $port -u root -e "SELECT 'Connection successful' as status;" 2>/dev/null
    return $?
}

# Function to run test queries
run_test_queries() {
    local port=${1:-3306}
    echo "Running test queries on port $port..."
    
    # Test 1: Show databases
    echo -e "\n1. SHOW DATABASES:"
    mysql -h localhost -P $port -u root -e "SHOW DATABASES;" 2>/dev/null
    
    # Test 2: Create test database
    echo -e "\n2. CREATE DATABASE test_passthrough:"
    mysql -h localhost -P $port -u root -e "CREATE DATABASE IF NOT EXISTS test_passthrough;" 2>/dev/null
    
    # Test 3: Use database and create table
    echo -e "\n3. CREATE TABLE:"
    mysql -h localhost -P $port -u root -e "
    USE test_passthrough;
    CREATE TABLE IF NOT EXISTS test_table (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100),
        value INT
    );" 2>/dev/null
    
    # Test 4: Insert data
    echo -e "\n4. INSERT DATA:"
    mysql -h localhost -P $port -u root -e "
    USE test_passthrough;
    INSERT INTO test_table (name, value) VALUES 
    ('test1', 100),
    ('test2', 200),
    ('test3', 300);" 2>/dev/null
    
    # Test 5: Select data
    echo -e "\n5. SELECT DATA:"
    mysql -h localhost -P $port -u root -e "
    USE test_passthrough;
    SELECT * FROM test_table;" 2>/dev/null
    
    # Test 6: Cleanup
    echo -e "\n6. CLEANUP:"
    mysql -h localhost -P $port -u root -e "DROP DATABASE IF EXISTS test_passthrough;" 2>/dev/null
}

# Check if server is already running
if pgrep -f "bin/mysql-server" > /dev/null; then
    echo "MySQL server is already running. Stopping it first..."
    pkill -f "bin/mysql-server"
    sleep 2
fi

# Test with MySQL passthrough (if MySQL is available)
echo -e "\n=== Testing MySQL Passthrough Mode ==="
echo "Starting server with MySQL backend..."

# Check if MySQL service is available
if command -v mysql &> /dev/null; then
    # Try to start MySQL service if not running
    sudo systemctl start mysql 2>/dev/null || sudo service mysql start 2>/dev/null || true
    sleep 2
fi

# Start our server in background with MySQL backend on port 3311
./bin/mysql-server --storage mysql --port 3311 > mysql_passthrough.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID on port 3311"

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 5

# Update test connection function for different port
test_mysql_passthrough() {
    mysql -h localhost -P 3311 -u root -e "SELECT 'Connection successful' as status;" 2>/dev/null
}

# Test connection
if test_mysql_passthrough; then
    echo "✅ Connection successful!"
    run_test_queries 3311
else
    echo "❌ Connection failed. MySQL might not be available."
    echo "Server logs:"
    tail -n 20 mysql_passthrough.log
fi

# Stop server
echo -e "\nStopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# Test with LMDB backend as fallback
echo -e "\n=== Testing LMDB Backend Mode (Fallback) ==="
echo "Starting server with LMDB backend..."

# Start server with LMDB backend on port 3312
./bin/mysql-server --storage lmdb --port 3312 > lmdb_test.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 5

# Test connection
if test_connection 3312; then
    echo "✅ Connection successful!"
    run_test_queries 3312
else
    echo "❌ Connection failed."
    echo "Server logs:"
    tail -n 20 lmdb_test.log
fi

# Stop server
echo -e "\nStopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo -e "\n=== Test Complete ==="
echo "Check mysql_passthrough.log and lmdb_test.log for detailed server output."