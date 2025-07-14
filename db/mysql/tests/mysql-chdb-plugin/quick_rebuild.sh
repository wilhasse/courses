#!/bin/bash

echo "Rebuilding MySQL chDB API UDF..."

# Build
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin
mkdir -p build
cd build
g++ -Wall -O2 -fPIC -shared -I/usr/include/mysql ../src/chdb_api_functions.cpp -o chdb_api_functions.so

# Install
sudo cp chdb_api_functions.so /usr/lib/mysql/plugin/

# Reinstall functions
mysql teste -u root -pteste <<EOF
DROP FUNCTION IF EXISTS chdb_query;
DROP FUNCTION IF EXISTS chdb_count;
DROP FUNCTION IF EXISTS chdb_sum;

CREATE FUNCTION chdb_query RETURNS STRING SONAME 'chdb_api_functions.so';
CREATE FUNCTION chdb_count RETURNS INTEGER SONAME 'chdb_api_functions.so';
CREATE FUNCTION chdb_sum RETURNS REAL SONAME 'chdb_api_functions.so';
EOF

echo "Done! UDF rebuilt and installed."
