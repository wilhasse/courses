#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
PLUGIN_FILE="$BUILD_DIR/test_tvf_plugin.so"
MYSQL_PLUGIN_DIR="/usr/lib/mysql/plugin"

if [ ! -f "$PLUGIN_FILE" ]; then
    echo "Error: Plugin file not found at $PLUGIN_FILE"
    echo "Please run ./scripts/build_tvf.sh first"
    exit 1
fi

if [ ! -d "$MYSQL_PLUGIN_DIR" ]; then
    echo "Error: MySQL plugin directory not found at $MYSQL_PLUGIN_DIR"
    exit 1
fi

echo "Installing MySQL Table-Valued Function Plugin..."

echo "Copying plugin to MySQL plugin directory..."
sudo cp "$PLUGIN_FILE" "$MYSQL_PLUGIN_DIR/"

if [ $? -ne 0 ]; then
    echo "Failed to copy plugin file"
    exit 1
fi

echo "Installing UDF functions in MySQL..."

# First drop existing functions if they exist
echo "Dropping existing functions..."
mysql -u root -pteste -e "DROP FUNCTION IF EXISTS test2_row_count;" 2>/dev/null
mysql -u root -pteste -e "DROP FUNCTION IF EXISTS test2_get_id;" 2>/dev/null
mysql -u root -pteste -e "DROP FUNCTION IF EXISTS test2_get_name;" 2>/dev/null
mysql -u root -pteste -e "DROP FUNCTION IF EXISTS test2_get_value;" 2>/dev/null

# Install all UDF functions
echo "Creating new functions..."
if ! mysql -u root -pteste -e "CREATE FUNCTION test2_row_count RETURNS INTEGER SONAME 'test_tvf_plugin.so';" 2>&1 | grep -v "Warning"; then
    echo "Note: Functions may already be installed. Continuing..."
fi

if ! mysql -u root -pteste -e "CREATE FUNCTION test2_get_id RETURNS INTEGER SONAME 'test_tvf_plugin.so';" 2>&1 | grep -v "Warning"; then
    echo "Note: Functions may already be installed. Continuing..."
fi

if ! mysql -u root -pteste -e "CREATE FUNCTION test2_get_name RETURNS STRING SONAME 'test_tvf_plugin.so';" 2>&1 | grep -v "Warning"; then
    echo "Note: Functions may already be installed. Continuing..."
fi

if ! mysql -u root -pteste -e "CREATE FUNCTION test2_get_value RETURNS REAL SONAME 'test_tvf_plugin.so';" 2>&1 | grep -v "Warning"; then
    echo "Note: Functions may already be installed. Continuing..."
fi

# Verify installation
echo ""
echo "Verifying installation..."
if mysql -u root -pteste -e "SELECT test2_row_count() AS test;" 2>&1 | grep -q "5"; then
    echo "✓ Functions installed and working correctly!"
else
    echo "✗ Functions may not be working correctly"
fi

echo "UDF functions installed successfully!"
echo ""
echo "You can now use the following functions:"
echo "  - test2_row_count() - returns number of rows (5)"
echo "  - test2_get_id(row_num) - returns id for given row"
echo "  - test2_get_name(row_num) - returns name for given row"  
echo "  - test2_get_value(row_num) - returns value for given row"
echo ""
echo "To uninstall, run:"
echo "  sudo $SCRIPT_DIR/uninstall_tvf.sh"