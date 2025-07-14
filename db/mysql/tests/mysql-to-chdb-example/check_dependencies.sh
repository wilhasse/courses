#!/bin/bash

echo "=== Checking dependencies for chDB API Server ==="
echo

# Check for protobuf compiler
echo -n "Checking for protoc... "
if command -v protoc &> /dev/null; then
    echo "FOUND ($(protoc --version))"
else
    echo "NOT FOUND"
    echo "  Install with: sudo apt-get install -y protobuf-compiler"
fi

# Check for protobuf library
echo -n "Checking for libprotobuf... "
if ldconfig -p 2>/dev/null | grep -q libprotobuf; then
    echo "FOUND"
else
    echo "NOT FOUND"
    echo "  Install with: sudo apt-get install -y libprotobuf-dev"
fi

# Check for C++ compiler
echo -n "Checking for g++... "
if command -v g++ &> /dev/null; then
    echo "FOUND ($(g++ --version | head -1))"
else
    echo "NOT FOUND"
    echo "  Install with: sudo apt-get install -y build-essential"
fi

# Check for chDB library
echo -n "Checking for libchdb.so... "
if [ -f "/home/cslog/chdb/libchdb.so" ]; then
    echo "FOUND"
else
    echo "NOT FOUND"
    echo "  Build with: cd /home/cslog/chdb && make build"
fi

# Check if protobuf files are generated
echo -n "Checking for generated protobuf files... "
if [ -f "chdb_api.pb.h" ] && [ -f "chdb_api.pb.cc" ]; then
    echo "FOUND"
else
    echo "NOT FOUND"
    echo "  Generate with: protoc --cpp_out=. chdb_api.proto"
fi

echo
echo "=== Summary ==="
if command -v protoc &> /dev/null && ldconfig -p 2>/dev/null | grep -q libprotobuf; then
    echo "All dependencies are installed!"
    echo
    echo "To build and run:"
    echo "1. Generate protobuf files: protoc --cpp_out=. chdb_api.proto"
    echo "2. Build server and client: make chdb_api_server chdb_api_client"
    echo "3. Run server: ./chdb_api_server"
    echo "4. Test client: ./chdb_api_client \"SELECT 1\""
else
    echo "Some dependencies are missing. Please install them first."
    echo
    echo "Quick install command for Ubuntu/Debian:"
    echo "sudo apt-get update && sudo apt-get install -y protobuf-compiler libprotobuf-dev"
fi