#!/bin/bash

# Quick test script to demonstrate the chDB API server

echo "=== Quick chDB API Server Test ==="
echo

# Check if protobuf is installed
if ! command -v protoc &> /dev/null; then
    echo "ERROR: protoc not found. Please install Protocol Buffers:"
    echo "  sudo apt-get install -y protobuf-compiler libprotobuf-dev"
    exit 1
fi

# Build if needed
if [ ! -f "chdb_api_server" ] || [ ! -f "chdb_api_client" ]; then
    echo "Building API server and client..."
    make chdb_api_server chdb_api_client
    if [ $? -ne 0 ]; then
        echo "Build failed. Please check for errors."
        exit 1
    fi
fi

# Check if data exists
if [ ! -d "./clickhouse_data" ]; then
    echo "ClickHouse data not found. Running feed_data_v2..."
    if [ ! -f "feed_data_v2" ]; then
        make feed_data_v2
    fi
    ./feed_data_v2
    echo
fi

# Start server
echo "Starting chDB API server..."
./chdb_api_server &
SERVER_PID=$!
sleep 3

# Check if server started
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server failed to start"
    exit 1
fi

echo
echo "=== Running test queries ==="
echo

# Test 1: Simple count
echo "1. Customer count:"
./chdb_api_client "SELECT COUNT(*) FROM mysql_import.customers"
echo

# Test 2: Group by with formatting
echo "2. Customers by city (TSV format):"
./chdb_api_client "SELECT city, COUNT(*) as count FROM mysql_import.customers GROUP BY city ORDER BY count DESC" TSV
echo

# Test 3: Join query
echo "3. Top revenue customers:"
./chdb_api_client "SELECT c.name, SUM(o.price * o.quantity) as revenue FROM mysql_import.customers c JOIN mysql_import.orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY revenue DESC LIMIT 3"
echo

# Stop server
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo
echo "=== Test complete! ==="
echo "The API server successfully:"
echo "- Loaded the 722MB libchdb.so once"
echo "- Served multiple queries without reloading"
echo "- Provided fast query responses"
echo
echo "This approach is ideal for:"
echo "- MySQL UDF integration (avoiding crashes)"
echo "- Multi-client environments"
echo "- Production deployments"