#!/bin/bash

echo "=== Installing Wrapper-based UDF ==="
echo

# First ensure the wrapper plugin exists
if [ ! -f "build/mysql_chdb_tvf_wrapper.so" ]; then
    echo "Building wrapper plugin..."
    cd build
    g++ -shared -fPIC -o mysql_chdb_tvf_wrapper.so \
        ../src/chdb_tvf_wrapper.cpp \
        $(mysql_config --cflags) \
        $(mysql_config --libs) \
        -std=c++11
    cd ..
fi

# Copy to plugin directory
echo "Copying plugin to MySQL..."
sudo cp build/mysql_chdb_tvf_wrapper.so /usr/lib/mysql/plugin/

# Create the functions
echo "Creating functions in MySQL..."
mysql teste -u root -pteste << 'EOF'
-- Create wrapper functions
CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_tvf_wrapper.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_tvf_wrapper.so';

-- Test them
SELECT '=== Testing Functions ===' as info;
SELECT ch_customer_count() AS customer_count;
SELECT ch_query_scalar('SELECT AVG(age) FROM mysql_import.customers') AS avg_age;
SELECT ch_query_scalar('SELECT COUNT(*) FROM mysql_import.orders') AS order_count;
EOF

echo
echo "If you see results above, the integration is working!"