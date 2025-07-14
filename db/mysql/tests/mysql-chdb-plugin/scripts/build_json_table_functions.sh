#!/bin/bash

# Build script for MySQL JSON table functions (MySQL 8.0.19+)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Building chDB JSON Table Functions (MySQL 8.0.19+) ==="
echo "Project directory: $PROJECT_DIR"
echo

# Check MySQL version
MYSQL_VERSION=$(mysql --version 2>/dev/null | grep -oP 'mysql\s+Ver\s+\K[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
echo "Detected MySQL version: $MYSQL_VERSION"

if [[ "$MYSQL_VERSION" != "unknown" ]]; then
    IFS='.' read -ra VERSION_PARTS <<< "$MYSQL_VERSION"
    MAJOR=${VERSION_PARTS[0]}
    MINOR=${VERSION_PARTS[1]}
    PATCH=${VERSION_PARTS[2]}
    
    if [[ $MAJOR -lt 8 ]] || [[ $MAJOR -eq 8 && $MINOR -eq 0 && $PATCH -lt 19 ]]; then
        echo "WARNING: JSON_TABLE requires MySQL 8.0.19+. You have $MYSQL_VERSION"
        echo "The functions will still work, but you may encounter issues with JSON_TABLE."
    else
        echo "✓ MySQL version supports JSON_TABLE"
    fi
fi

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

# Compile the JSON table functions
echo
echo "Compiling chdb_json_table_functions.cpp..."
g++ -Wall -O2 -fPIC -shared \
    -I"$MYSQL_INCLUDE" \
    "$SRC_DIR/chdb_json_table_functions.cpp" \
    -o chdb_json_table_functions.so

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo
    echo "Generated: $BUILD_DIR/chdb_json_table_functions.so"
    echo
    echo "To install the functions:"
    echo "  1. Copy to MySQL plugin directory:"
    echo "     sudo cp $BUILD_DIR/chdb_json_table_functions.so $PLUGIN_DIR/"
    echo
    echo "  2. Register the functions in MySQL:"
    echo "     mysql -u root -p < $PROJECT_DIR/scripts/install_json_table_functions.sql"
    echo
    echo "  3. Test JSON_TABLE functionality:"
    echo "     mysql -u root -p < $PROJECT_DIR/tests/test_json_table_functions.sql"
    echo
    echo "Example usage:"
    echo "  SELECT * FROM JSON_TABLE("
    echo "      chdb_customers_json(),"
    echo "      '\$[*]' COLUMNS ("
    echo "          id INT PATH '\$.id',"
    echo "          name VARCHAR(100) PATH '\$.name'"
    echo "      )"
    echo "  ) AS customers;"
else
    echo "Build failed!"
    exit 1
fi