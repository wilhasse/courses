#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "Building MySQL Table-Valued Function Plugin..."

if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler not found. Please install build-essential."
    exit 1
fi

if [ ! -d "/usr/include/mysql" ]; then
    echo "Error: MySQL development headers not found. Please install libmysqlclient-dev."
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "Compiling test_tvf_plugin.cpp..."
g++ -shared -fPIC \
    -I/usr/include/mysql \
    -o "$BUILD_DIR/test_tvf_plugin.so" \
    "$PROJECT_ROOT/src/test_tvf_plugin.cpp"

if [ $? -eq 0 ]; then
    echo "Build successful! Plugin created at: $BUILD_DIR/test_tvf_plugin.so"
    echo ""
    echo "To install the plugin, run:"
    echo "  sudo $SCRIPT_DIR/install_tvf.sh"
else
    echo "Build failed!"
    exit 1
fi