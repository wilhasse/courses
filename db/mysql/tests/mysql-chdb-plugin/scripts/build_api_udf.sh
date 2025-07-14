#!/bin/bash

# Build script for MySQL UDF functions that connect to chDB API server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Building chDB API UDF Functions ==="
echo "Project directory: $PROJECT_DIR"
echo

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Get MySQL plugin directory
PLUGIN_DIR=$(mysql_config --plugindir 2>/dev/null || echo "/usr/lib/mysql/plugin")
echo "MySQL plugin directory: $PLUGIN_DIR"

# Get MySQL include directory
MYSQL_INCLUDE=$(mysql_config --cflags 2>/dev/null | sed 's/-I//g' | awk '{print $1}')
if [ -z "$MYSQL_INCLUDE" ]; then
    MYSQL_INCLUDE="/usr/include/mysql"
fi
echo "MySQL include directory: $MYSQL_INCLUDE"

# Compile the UDF
echo
echo "Compiling chdb_api_functions.cpp..."
g++ -Wall -O2 -fPIC -shared \
    -I"$MYSQL_INCLUDE" \
    "$SRC_DIR/chdb_api_functions.cpp" \
    -o chdb_api_functions.so

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo
    echo "Generated: $BUILD_DIR/chdb_api_functions.so"
    echo
    echo "To install the UDF:"
    echo "  1. Copy to MySQL plugin directory:"
    echo "     sudo cp $BUILD_DIR/chdb_api_functions.so $PLUGIN_DIR/"
    echo
    echo "  2. Register the functions in MySQL:"
    echo "     mysql -u root -p < $PROJECT_DIR/scripts/install_api_udf.sql"
    echo
    echo "Note: Make sure the chDB API server is running on port 8125"
else
    echo "Build failed!"
    exit 1
fi