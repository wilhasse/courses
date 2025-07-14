#!/bin/bash

echo "=== MySQL Cleanup and Restart ==="
echo

# Stop MySQL if running
echo "Stopping MySQL..."
sudo systemctl stop mysql

# Remove problematic plugins
echo "Removing problematic plugins..."
sudo rm -f /usr/lib/mysql/plugin/mysql_chdb_tvf_embedded.so
sudo rm -f /usr/lib/mysql/plugin/mysql_chdb_clickhouse_tvf.so
sudo rm -f /usr/lib/mysql/plugin/mysql_chdb_clickhouse_tvf_debug.so

# Start MySQL
echo "Starting MySQL..."
sudo systemctl start mysql

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
sleep 3

# Check status
echo "MySQL status:"
sudo systemctl status mysql | head -10

# Clean up any remaining functions
echo
echo "Cleaning up MySQL functions..."
mysql teste -u root -pteste << 'EOF' 2>/dev/null || echo "Skipping function cleanup"
-- Remove all ch_ functions
DROP FUNCTION IF EXISTS ch_customer_count;
DROP FUNCTION IF EXISTS ch_get_customer_id;
DROP FUNCTION IF EXISTS ch_get_customer_name;
DROP FUNCTION IF EXISTS ch_get_customer_city;
DROP FUNCTION IF EXISTS ch_get_customer_age;
DROP FUNCTION IF EXISTS ch_query_scalar;
DROP FUNCTION IF EXISTS chdb_columns;
DROP FUNCTION IF EXISTS chdb_to_utf8;

-- Show remaining functions
SELECT name, dl FROM mysql.func WHERE name LIKE 'ch%';
EOF

echo
echo "Cleanup complete. MySQL should be running clean now."
echo
echo "To test the wrapper approach, run:"
echo "./build_wrapper_tvf.sh"