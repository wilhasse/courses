#!/bin/bash

echo "=== Rebuilding JSON functions with safe implementation ==="

cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin
mkdir -p build
cd build

# Build the safe version
echo "Building safe JSON functions..."
g++ -Wall -O2 -fPIC -shared \
    -I/usr/include/mysql \
    ../src/chdb_json_table_functions_safe.cpp \
    -o chdb_json_table_functions_safe.so

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Install
echo "Installing functions..."
sudo cp chdb_json_table_functions_safe.so /usr/lib/mysql/plugin/

# Check if MySQL is running first
if ! mysqladmin -u root -pteste ping > /dev/null 2>&1; then
    echo "Warning: MySQL server not responding, attempting to start..."
    sudo systemctl start mysql || sudo service mysql start
    sleep 3
fi

# Reinstall in MySQL with error handling
echo "Installing functions in MySQL..."
mysql -u root -pteste <<EOF || echo "Function installation failed"
-- Drop existing functions safely
DROP FUNCTION IF EXISTS chdb_test_json;
DROP FUNCTION IF EXISTS chdb_customers_json;

-- Create new functions
CREATE FUNCTION chdb_test_json RETURNS STRING SONAME 'chdb_json_table_functions_safe.so';
CREATE FUNCTION chdb_customers_json RETURNS STRING SONAME 'chdb_json_table_functions_safe.so';

-- Test basic function
SELECT 'Function installation test:' AS message;
SELECT chdb_test_json() AS test_result;
EOF

echo "Safe functions rebuilt and installed!"
echo
echo "Test with:"
echo "  mysql -u root -pteste -e \"SELECT chdb_test_json();\""
echo "  mysql -u root -pteste < test_json_simple.sql"