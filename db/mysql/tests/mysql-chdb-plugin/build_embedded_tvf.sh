#!/bin/bash

echo "Building ClickHouse TVF plugin with embedded chDB (proper implementation)..."
echo

# Check if libchdb.so exists
if [ ! -f "/home/cslog/chdb/libchdb.so" ]; then
    echo "ERROR: libchdb.so not found at /home/cslog/chdb/libchdb.so"
    echo "Please build chDB first:"
    echo "  cd /home/cslog/chdb"
    echo "  make buildlib"
    exit 1
fi

echo "Found libchdb.so"
ls -lh /home/cslog/chdb/libchdb.so
echo

cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

# Compile the plugin
echo "Compiling chdb_tvf_embedded.cpp..."
g++ -shared -fPIC -o mysql_chdb_tvf_embedded.so \
    ../src/chdb_tvf_embedded.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -ldl \
    -std=c++11

if [ $? -eq 0 ]; then
    echo "Build successful!"
    ls -lh mysql_chdb_tvf_embedded.so
    
    # Copy to MySQL plugin directory
    echo
    echo "Copying plugin to MySQL plugin directory..."
    sudo cp mysql_chdb_tvf_embedded.so /usr/lib/mysql/plugin/
    
    echo
    echo "Installation complete!"
    echo
    echo "To install the functions in MySQL, run:"
    echo "----------------------------------------"
    cat << 'EOF'
mysql teste -u root -pteste << 'EOSQL'
-- Drop old functions if they exist
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_get_customer_id;
DROP FUNCTION IF EXISTS ch_get_customer_name;
DROP FUNCTION IF EXISTS ch_get_customer_city;
DROP FUNCTION IF EXISTS ch_get_customer_age;
DROP FUNCTION IF EXISTS ch_query_scalar;

-- Create new functions with embedded chDB
CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_id RETURNS INTEGER SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_name RETURNS STRING SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_city RETURNS STRING SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_age RETURNS INTEGER SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_tvf_embedded.so';

-- Test the functions
SELECT ch_customer_count() AS customer_count;
EOSQL
EOF
else
    echo "Build failed!"
    exit 1
fi
