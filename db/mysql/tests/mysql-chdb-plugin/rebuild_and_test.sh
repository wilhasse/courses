#!/bin/bash

echo "Rebuilding and testing chDB embedded plugin with debug logging..."

# Clear previous debug log
rm -f /tmp/mysql_chdb_embedded_debug.log
touch /tmp/mysql_chdb_embedded_debug.log
chmod 666 /tmp/mysql_chdb_embedded_debug.log

# Rebuild the plugin
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/build

echo "Compiling with debug logging..."
g++ -shared -fPIC -o mysql_chdb_tvf_embedded.so \
    ../src/chdb_tvf_embedded.cpp \
    $(mysql_config --cflags) \
    $(mysql_config --libs) \
    -ldl \
    -std=c++11

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful! Copying to MySQL plugin directory..."
sudo cp mysql_chdb_tvf_embedded.so /usr/lib/mysql/plugin/

echo
echo "Reinstalling functions..."
mysql teste -u root -pteste << 'EOF'
-- Drop and recreate functions to ensure fresh load
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_get_customer_id;
DROP FUNCTION IF EXISTS ch_get_customer_name;
DROP FUNCTION IF EXISTS ch_get_customer_city;
DROP FUNCTION IF EXISTS ch_get_customer_age;
DROP FUNCTION IF EXISTS ch_query_scalar;

CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_id RETURNS INTEGER SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_name RETURNS STRING SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_city RETURNS STRING SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_get_customer_age RETURNS INTEGER SONAME 'mysql_chdb_tvf_embedded.so';
CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_tvf_embedded.so';
EOF

echo
echo "Testing functions..."
echo "1. Simple test:"
mysql teste -u root -pteste -e "SELECT ch_customer_count() AS count;"

echo
echo "2. Query scalar test:"
mysql teste -u root -pteste -e "SELECT ch_query_scalar('SELECT 1') AS result;"

echo
echo "Debug log contents:"
cat /tmp/mysql_chdb_embedded_debug.log

echo
echo "If the log is empty, it might be a permission issue. Checking file permissions:"
ls -la /tmp/mysql_chdb_embedded_debug.log