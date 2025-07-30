#!/bin/bash

# Set LD_LIBRARY_PATH for LMDB
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH

echo "Simple MySQL Server Test"
echo "========================"

# Kill any existing processes
pkill -f "bin/mysql-server" 2>/dev/null

# Start server with LMDB backend on port 3312
echo "Starting server with LMDB backend on port 3312..."
./bin/mysql-server --storage lmdb --port 3312 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to fully start
echo "Waiting for server to initialize..."
sleep 8

# Test connection
echo -e "\nTesting connection..."
if mysql -h 127.0.0.1 -P 3312 -u root -e "SELECT 'Server is running!' as Status;" 2>/dev/null; then
    echo "✅ Connection successful!"
    
    # Run some basic tests
    echo -e "\n=== Running Basic Tests ==="
    
    echo -e "\n1. Show databases:"
    mysql -h 127.0.0.1 -P 3312 -u root -e "SHOW DATABASES;"
    
    echo -e "\n2. Use testdb and show tables:"
    mysql -h 127.0.0.1 -P 3312 -u root -e "USE testdb; SHOW TABLES;"
    
    echo -e "\n3. Query users table:"
    mysql -h 127.0.0.1 -P 3312 -u root -e "USE testdb; SELECT * FROM users LIMIT 3;"
    
    echo -e "\n4. Create new database:"
    mysql -h 127.0.0.1 -P 3312 -u root -e "CREATE DATABASE IF NOT EXISTS test_new;"
    mysql -h 127.0.0.1 -P 3312 -u root -e "SHOW DATABASES;"
    
else
    echo "❌ Connection failed!"
    echo "Server logs:"
    tail -n 20 lmdb_test.log
fi

# Stop server
echo -e "\nStopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo -e "\n=== Test Complete ==="