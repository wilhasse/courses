#!/bin/bash

# Build the ClickHouse TVF plugin manually
echo "Building ClickHouse TVF plugin..."

cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

# Compile the plugin
g++ -shared -fPIC -o mysql_chdb_clickhouse_tvf.so \
    ../src/chdb_clickhouse_tvf.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -std=c++11

if [ $? -eq 0 ]; then
    echo "Build successful!"
    ls -la mysql_chdb_clickhouse_tvf.so
else
    echo "Build failed!"
    exit 1
fi
