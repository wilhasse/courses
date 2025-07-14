#!/bin/bash

echo "=== Rebuilding JSON functions with character set fix ==="

cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin
mkdir -p build
cd build

# Build the fixed version
echo "Building fixed JSON functions..."
g++ -Wall -O2 -fPIC -shared \
    -I/usr/include/mysql \
    ../src/chdb_json_table_functions_fixed.cpp \
    -o chdb_json_table_functions.so

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Install
echo "Installing functions..."
sudo cp chdb_json_table_functions.so /usr/lib/mysql/plugin/

# Reinstall in MySQL
mysql -u root -pteste <<EOF
DROP FUNCTION IF EXISTS chdb_customers_json;
DROP FUNCTION IF EXISTS chdb_customers_json_raw;
DROP FUNCTION IF EXISTS chdb_test_json;

CREATE FUNCTION chdb_customers_json RETURNS STRING SONAME 'chdb_json_table_functions.so';
CREATE FUNCTION chdb_customers_json_raw RETURNS STRING SONAME 'chdb_json_table_functions.so';
CREATE FUNCTION chdb_test_json RETURNS STRING SONAME 'chdb_json_table_functions.so';
EOF

echo "Functions rebuilt and installed!"
echo
echo "Test with:"
echo "  mysql -u root -pteste < test_json_simple.sql"