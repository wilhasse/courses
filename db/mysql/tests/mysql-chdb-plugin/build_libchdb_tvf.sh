#!/bin/bash

echo "Building ClickHouse TVF plugin using libchdb.so..."

# Check if libchdb.so exists
if [ ! -f "/home/cslog/chdb/libchdb.so" ]; then
    echo "ERROR: libchdb.so not found at /home/cslog/chdb/libchdb.so"
    echo "Please build chDB first with 'make buildlib' in the chDB directory"
    exit 1
fi

cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

# Compile the plugin
echo "Compiling chdb_tvf_libchdb.cpp..."
g++ -shared -fPIC -o mysql_chdb_tvf_libchdb.so \
    ../src/chdb_tvf_libchdb.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -ldl \
    -std=c++11

if [ $? -eq 0 ]; then
    echo "Build successful!"
    ls -la mysql_chdb_tvf_libchdb.so
    
    # Copy to MySQL plugin directory
    echo "Copying plugin to MySQL plugin directory..."
    sudo cp mysql_chdb_tvf_libchdb.so /usr/lib/mysql/plugin/
    
    echo ""
    echo "To install the functions, run:"
    echo "mysql -u root -pteste << EOF"
    echo "DROP FUNCTION IF EXISTS ch_customer_count;"
    echo "DROP FUNCTION IF EXISTS ch_get_customer_id;"
    echo "DROP FUNCTION IF EXISTS ch_get_customer_name;"
    echo "DROP FUNCTION IF EXISTS ch_get_customer_city;"
    echo "DROP FUNCTION IF EXISTS ch_get_customer_age;"
    echo "DROP FUNCTION IF EXISTS ch_query_scalar;"
    echo ""
    echo "CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_tvf_libchdb.so';"
    echo "CREATE FUNCTION ch_get_customer_id RETURNS INTEGER SONAME 'mysql_chdb_tvf_libchdb.so';"
    echo "CREATE FUNCTION ch_get_customer_name RETURNS STRING SONAME 'mysql_chdb_tvf_libchdb.so';"
    echo "CREATE FUNCTION ch_get_customer_city RETURNS STRING SONAME 'mysql_chdb_tvf_libchdb.so';"
    echo "CREATE FUNCTION ch_get_customer_age RETURNS INTEGER SONAME 'mysql_chdb_tvf_libchdb.so';"
    echo "CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_tvf_libchdb.so';"
    echo "EOF"
else
    echo "Build failed!"
    exit 1
fi
