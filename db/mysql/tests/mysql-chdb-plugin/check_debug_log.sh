#!/bin/bash

# Script to check debug logs for MySQL chDB API UDF

LOG_FILE="/tmp/mysql_chdb_api_debug.log"

echo "=== MySQL chDB API Debug Log ==="
echo

if [ ! -f "$LOG_FILE" ]; then
    echo "Debug log file not found: $LOG_FILE"
    echo "The log will be created when the UDF is used."
    exit 1
fi

echo "Last 50 lines of debug log:"
echo "----------------------------"
tail -n 50 "$LOG_FILE"

echo
echo "To monitor in real-time:"
echo "  tail -f $LOG_FILE"
echo
echo "To clear the log:"
echo "  > $LOG_FILE"