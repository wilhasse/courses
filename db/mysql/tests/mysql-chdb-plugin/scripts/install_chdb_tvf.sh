#!/bin/bash

echo "Installing ClickHouse TVF functions..."

# Function to execute MySQL command
mysql_exec() {
    mysql teste -u root -pteste -e "$1" 2>/dev/null
}

# Drop existing functions if they exist
echo "Dropping existing functions if any..."
mysql_exec "DROP FUNCTION IF EXISTS ch_customer_count;"
mysql_exec "DROP FUNCTION IF EXISTS ch_get_customer_id;"
mysql_exec "DROP FUNCTION IF EXISTS ch_get_customer_name;"
mysql_exec "DROP FUNCTION IF EXISTS ch_get_customer_city;"
mysql_exec "DROP FUNCTION IF EXISTS ch_get_customer_age;"
mysql_exec "DROP FUNCTION IF EXISTS ch_query_scalar;"

# Create functions
echo "Creating ch_customer_count function..."
mysql_exec "CREATE FUNCTION ch_customer_count RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';"

echo "Creating ch_get_customer_id function..."
mysql_exec "CREATE FUNCTION ch_get_customer_id RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';"

echo "Creating ch_get_customer_name function..."
mysql_exec "CREATE FUNCTION ch_get_customer_name RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';"

echo "Creating ch_get_customer_city function..."
mysql_exec "CREATE FUNCTION ch_get_customer_city RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';"

echo "Creating ch_get_customer_age function..."
mysql_exec "CREATE FUNCTION ch_get_customer_age RETURNS INTEGER SONAME 'mysql_chdb_clickhouse_tvf.so';"

echo "Creating ch_query_scalar function..."
mysql_exec "CREATE FUNCTION ch_query_scalar RETURNS STRING SONAME 'mysql_chdb_clickhouse_tvf.so';"

echo "Installation complete!"

# Show installed functions
echo -e "\nInstalled functions:"
mysql -u root -pteste -e "SHOW FUNCTION STATUS WHERE Name LIKE 'ch_%';" 2>/dev/null
