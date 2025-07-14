#!/bin/bash

# Fix debug log permissions

echo "Fixing debug log permissions..."

# Create log file with proper permissions
sudo touch /tmp/mysql_chdb_api_debug.log
sudo chmod 666 /tmp/mysql_chdb_api_debug.log

echo "Debug log permissions fixed."
echo "The MySQL process should now be able to write to the log."

# Test write
echo "[$(date)] Debug log test write" >> /tmp/mysql_chdb_api_debug.log

if [ $? -eq 0 ]; then
    echo "✓ Log file is writable"
else
    echo "✗ Log file is still not writable"
fi