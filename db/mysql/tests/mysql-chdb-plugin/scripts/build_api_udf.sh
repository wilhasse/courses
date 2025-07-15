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
echo "Compiling chdb_api_udf.cpp..."
g++ -Wall -O2 -fPIC -shared \
    -I"$MYSQL_INCLUDE" \
    "$SRC_DIR/chdb_api_udf.cpp" \
    -o chdb_api_udf.so

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo
    echo "Generated: $BUILD_DIR/chdb_api_udf.so"
    echo
    
    # Check if function already exists
    echo "Checking if function already exists..."
    FUNC_EXISTS=$(mysql -u root -N -e "SELECT COUNT(*) FROM mysql.func WHERE name = 'chdb_api_query' AND dl = 'chdb_api_udf.so';" 2>/dev/null || echo "0")
    
    if [ "$FUNC_EXISTS" -eq "1" ]; then
        echo "Function chdb_api_query already exists."
        echo
        echo "⚠️  WARNING: Replacing a loaded plugin can crash MySQL!"
        echo "To update the plugin safely:"
        echo "  1. Drop the function: mysql -u root -e \"DELETE FROM mysql.func WHERE name='chdb_api_query'; FLUSH PRIVILEGES;\""
        echo "  2. Restart MySQL: sudo systemctl restart mysql"
        echo "  3. Run this script again"
        echo
        echo "Or just use the existing function if it's working."
        exit 0
    fi
    
    # Copy plugin to MySQL directory (requires sudo)
    echo "Copying plugin to MySQL plugin directory..."
    sudo cp "$BUILD_DIR/chdb_api_udf.so" "$PLUGIN_DIR/"
    
    if [ $? -eq 0 ]; then
        echo "Plugin copied successfully!"
        echo
        
        echo "Creating chdb_api_query function..."
        mysql -u root -e "CREATE FUNCTION chdb_api_query RETURNS STRING SONAME 'chdb_api_udf.so';"
        
        # Test the function
        echo "Testing the function..."
        mysql -u root < "$PROJECT_DIR/scripts/install_api_udf_safe.sql"
        
        if [ $? -eq 0 ]; then
            echo
            echo "=== Installation Complete! ==="
            echo "The chDB API UDF function is now ready to use."
            echo
            echo "Available function:"
            echo "  - chdb_api_query(sql) : Execute any ClickHouse SQL query"
            echo
            echo "Example usage:"
            echo "  SELECT CAST(chdb_api_query('SELECT version()') AS CHAR);"
            echo "  SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico') AS CHAR);"
            echo
            echo "Note: Make sure the chDB API server is running on port 8125"
        else
            echo "Failed to install UDF functions in MySQL"
            exit 1
        fi
    else
        echo "Failed to copy plugin to MySQL directory"
        exit 1
    fi
else
    echo "Build failed!"
    exit 1
fi
