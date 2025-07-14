#!/bin/bash

echo "Rebuilding wrapper plugin..."
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

g++ -shared -fPIC -o mysql_chdb_tvf_wrapper.so \
    ../src/chdb_tvf_wrapper.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -std=c++11

if [ $? -eq 0 ]; then
    echo "Build successful!"
    sudo cp mysql_chdb_tvf_wrapper.so /usr/lib/mysql/plugin/
    
    # Test in MySQL
    echo
    echo "Testing in MySQL..."
    mysql teste -u root -pteste << 'EOF'
-- Drop and recreate to ensure fresh load
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_query_scalar;

CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_tvf_wrapper.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_tvf_wrapper.so';

-- Test
SELECT ch_customer_count() AS customer_count;
SELECT ch_query_scalar('SELECT AVG(age) FROM mysql_import.customers') AS avg_age;
EOF
else
    echo "Build failed!"
fi