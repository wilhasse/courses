#!/bin/bash

# Build script for chDB API Server with Protocol Buffers

echo "=== chDB API Server Build Script ==="
echo

# Check for protobuf
if ! command -v protoc &> /dev/null; then
    echo "ERROR: protoc (Protocol Buffer compiler) not found!"
    echo
    echo "To install protobuf on Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y protobuf-compiler libprotobuf-dev"
    echo
    echo "To install protobuf on macOS:"
    echo "  brew install protobuf"
    echo
    exit 1
fi

echo "Found protoc: $(protoc --version)"

# Check for libprotobuf
if ! ldconfig -p | grep -q libprotobuf; then
    echo "WARNING: libprotobuf may not be installed"
    echo "Install with: sudo apt-get install libprotobuf-dev"
fi

# Build the API server
echo
echo "Building chDB API server..."
make chdb_api_server chdb_api_client

if [ $? -eq 0 ]; then
    echo
    echo "=== Build successful! ==="
    echo
    echo "To start the API server:"
    echo "  ./chdb_api_server"
    echo
    echo "To test with the client:"
    echo "  ./chdb_api_client \"SELECT COUNT(*) FROM mysql_import.customers\""
    echo
    echo "To run performance tests:"
    echo "  ./test_performance.sh"
else
    echo
    echo "=== Build failed ==="
    echo "Please install required dependencies and try again"
fi