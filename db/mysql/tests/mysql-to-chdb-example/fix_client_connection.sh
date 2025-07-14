#!/bin/bash

echo "=== Fixing chDB API Client Connection Issue ==="
echo

# The issue is that the original client uses "localhost" which may not resolve correctly
# We need to use 127.0.0.1 instead

echo "The client was trying to connect to 'localhost' but inet_pton() requires an IP address."
echo "We've already updated the client to use '127.0.0.1' instead."
echo

echo "To fix the issue, you need to:"
echo "1. Install protobuf if not already installed:"
echo "   sudo apt-get install -y protobuf-compiler libprotobuf-dev"
echo
echo "2. Generate the protobuf files:"
echo "   protoc --cpp_out=. chdb_api.proto"
echo
echo "3. Rebuild the client:"
echo "   make chdb_api_client"
echo
echo "4. Run the client:"
echo "   ./chdb_api_client \"SELECT COUNT(*) FROM mysql_import.customers\""
echo

echo "Alternative: Use netcat to test the connection:"
echo "   nc -v 127.0.0.1 8125"
echo

echo "If the server is running on a different host or port, specify it:"
echo "   ./chdb_api_client \"SELECT 1\" CSV 127.0.0.1 8125"