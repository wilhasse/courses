#!/bin/bash

set -e

MYSQL_PLUGIN_DIR="/usr/lib/mysql/plugin"

echo "Uninstalling MySQL Table-Valued Function Plugin..."

echo "Uninstalling UDFs from MySQL..."
mysql -u root -pteste -e "
DROP FUNCTION IF EXISTS test2_row_count;
DROP FUNCTION IF EXISTS test2_get_id;
DROP FUNCTION IF EXISTS test2_get_name;
DROP FUNCTION IF EXISTS test2_get_value;
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Warning: Failed to uninstall UDFs from MySQL (they may not be installed)"
fi

echo "Removing plugin file..."
sudo rm -f "$MYSQL_PLUGIN_DIR/test_tvf_plugin.so"

if [ $? -eq 0 ]; then
    echo "Plugin uninstalled successfully!"
else
    echo "Failed to remove plugin file"
    exit 1
fi