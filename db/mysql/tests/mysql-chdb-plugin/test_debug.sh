#!/bin/bash

echo "Building debug version of ClickHouse TVF plugin..."

cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

# Compile the debug plugin
g++ -shared -fPIC -o mysql_chdb_clickhouse_tvf_debug.so \
    ../src/chdb_clickhouse_tvf_debug.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -std=c++11

if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Copy to MySQL plugin directory
    echo "Copying plugin to MySQL plugin directory..."
    sudo cp mysql_chdb_clickhouse_tvf_debug.so /usr/lib/mysql/plugin/
    
    # Clear any existing debug log
    rm -f /tmp/mysql_chdb_debug.log
    touch /tmp/mysql_chdb_debug.log
    chmod 666 /tmp/mysql_chdb_debug.log
    
    echo "Installing debug functions..."
    mysql teste -u root -pteste -f -e "DROP FUNCTION IF EXISTS ch_customer_count_debug;"
    mysql teste -u root -pteste -f -e "DROP FUNCTION IF EXISTS ch_query_scalar_debug;"
    mysql teste -u root -pteste -e "CREATE FUNCTION ch_customer_count_debug RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf_debug.so';"
    mysql teste -u root -pteste -e "CREATE FUNCTION ch_query_scalar_debug RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf_debug.so';"
    
    echo "\nTesting debug functions..."
    echo "Test 1: Customer count"
    mysql teste -u root -pteste -e "SELECT ch_customer_count_debug() AS count;"
    
    echo "\nTest 2: Simple query"
    mysql teste -u root -pteste -e "SELECT ch_query_scalar_debug('SELECT 1') AS result;"
    
    echo "\nDebug log contents:"
    cat /tmp/mysql_chdb_debug.log
    
else
    echo "Build failed!"
    exit 1
fi
