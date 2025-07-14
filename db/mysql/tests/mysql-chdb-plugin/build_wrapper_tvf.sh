#!/bin/bash

echo "Building safe wrapper-based ClickHouse TVF plugin..."
echo

# First, restart MySQL if it crashed
echo "Checking MySQL status..."
if ! systemctl is-active --quiet mysql; then
    echo "MySQL is not running. Attempting to restart..."
    sudo systemctl restart mysql
    sleep 2
fi

cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin

# Build the helper program
echo "Building chdb_query_helper..."
g++ -o chdb_query_helper chdb_query_helper.cpp -ldl -std=c++11

if [ $? -ne 0 ]; then
    echo "Failed to build helper!"
    exit 1
fi

chmod +x chdb_query_helper

# Test the helper directly
echo
echo "Testing helper program..."
./chdb_query_helper "SELECT 1 as test"
echo

# Build the MySQL plugin
echo "Building MySQL plugin..."
cd build
g++ -shared -fPIC -o mysql_chdb_tvf_wrapper.so \
    ../src/chdb_tvf_wrapper.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -std=c++11

if [ $? -ne 0 ]; then
    echo "Failed to build plugin!"
    exit 1
fi

echo "Build successful! Copying to MySQL plugin directory..."
sudo cp mysql_chdb_tvf_wrapper.so /usr/lib/mysql/plugin/

echo
echo "Installation complete!"
echo
echo "To install the functions in MySQL, run:"
echo "----------------------------------------"
cat << 'EOF'
mysql teste -u root -pteste << 'EOSQL'
-- Drop old functions if they exist
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_query_scalar;

-- Create new wrapper-based functions
CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_tvf_wrapper.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_tvf_wrapper.so';

-- Test the functions
SELECT ch_customer_count() AS customer_count;
SELECT ch_query_scalar('SELECT COUNT(*) FROM mysql_import.orders') AS order_count;
EOSQL
EOF