#!/bin/bash

# Build script for ClickHouse TVF plugin

echo "Building ClickHouse TVF plugin..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the chdb_clickhouse_tvf plugin
echo "Compiling chdb_clickhouse_tvf.cpp..."
g++ -shared -fPIC -o mysql_chdb_clickhouse_tvf.so \
    ../src/chdb_clickhouse_tvf.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -std=c++11

if [ $? -eq 0 ]; then
    echo "Build successful! Plugin created: build/mysql_chdb_clickhouse_tvf.so"
    
    # Copy to MySQL plugin directory
    echo "Copying plugin to MySQL plugin directory..."
    sudo cp mysql_chdb_clickhouse_tvf.so /usr/lib/mysql/plugin/
    
    if [ $? -eq 0 ]; then
        echo "Plugin copied successfully!"
    else
        echo "Failed to copy plugin. You may need to run with sudo or check permissions."
    fi
else
    echo "Build failed!"
    exit 1
fi