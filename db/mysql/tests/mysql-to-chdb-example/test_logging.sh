#!/bin/bash

echo "Testing Enhanced Logging for chDB API Servers"
echo "============================================="
echo

# Kill any existing servers
pkill -f chdb_api_server_simple
pkill -f chdb_api_server

# Test simple server
echo "1. Testing chdb_api_server_simple (port 8125)..."
./chdb_api_server_simple -p 8125 &
SERVER_PID=$!
sleep 2

# Run some test queries
echo "   Running test queries..."
./chdb_api_client_simple 8125 "SELECT 'Hello World' as greeting"
./chdb_api_client_simple 8125 "SELECT COUNT(*) FROM system.tables"
./chdb_api_client_simple 8125 "SELECT now() as current_time, version() as version"

echo
echo "2. Checking log file contents..."
echo "   Contents of chdb_api_server_simple.log:"
echo "   ========================================"
if [ -f chdb_api_server_simple.log ]; then
    tail -n 10 chdb_api_server_simple.log
else
    echo "   Log file not found!"
fi

# Kill simple server
kill $SERVER_PID 2>/dev/null
sleep 1

echo
echo "3. Testing chdb_api_server (protocol buffers, port 8126)..."
./chdb_api_server -p 8126 &
SERVER_PID=$!
sleep 2

# Run some test queries
echo "   Running test queries..."
./chdb_api_client 8126 "SELECT 'Protocol Buffer Test' as message" JSON
./chdb_api_client 8126 "SELECT number, number*2 as double FROM system.numbers LIMIT 5" CSV

echo
echo "4. Checking log file contents..."
echo "   Contents of chdb_api_server.log:"
echo "   ================================="
if [ -f chdb_api_server.log ]; then
    tail -n 10 chdb_api_server.log
else
    echo "   Log file not found!"
fi

# Kill protobuf server
kill $SERVER_PID 2>/dev/null

echo
echo "Test completed! Log files:"
echo "- chdb_api_server_simple.log"
echo "- chdb_api_server.log"