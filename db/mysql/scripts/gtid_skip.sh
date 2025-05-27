#!/bin/bash

# Function to show usage
usage() {
    echo "Usage: $0 <SERVER_UUID> <START_GTID> <END_GTID> [MYSQL_USER] [MYSQL_PASSWORD] [MYSQL_HOST] [MYSQL_PORT]"
    echo ""
    echo "Required parameters:"
    echo "  SERVER_UUID   - The MySQL server UUID"
    echo "  START_GTID    - Starting GTID number"
    echo "  END_GTID      - Ending GTID number"
    echo ""
    echo "Optional parameters (with defaults):"
    echo "  MYSQL_USER    - MySQL username (default: root)"
    echo "  MYSQL_PASSWORD- MySQL password (default: empty)"
    echo "  MYSQL_HOST    - MySQL host (default: localhost)"
    echo "  MYSQL_PORT    - MySQL port (default: 3306)"
    echo ""
    echo "Example:"
    echo "  $0 'eee60942-fc2f-11ee-85a9-00155d011e08' 367413595 367414789"
    echo "  $0 'eee60942-fc2f-11ee-85a9-00155d011e08' 367413595 367414789 myuser mypass"
    echo "  $0 'eee60942-fc2f-11ee-85a9-00155d011e08' 367413595 367414789 myuser mypass 192.168.1.100 3306"
    exit 1
}

# Check if minimum required parameters are provided
if [ $# -lt 3 ]; then
    echo "Error: Missing required parameters"
    usage
fi

# Required parameters
SERVER_UUID="$1"
START_GTID="$2"
END_GTID="$3"

# Optional parameters with defaults
MYSQL_USER="${4:-root}"
MYSQL_PASSWORD="${5:-}"
MYSQL_HOST="${6:-localhost}"
MYSQL_PORT="${7:-3306}"

# Validate GTID range
if [ "$START_GTID" -gt "$END_GTID" ]; then
    echo "Error: START_GTID ($START_GTID) cannot be greater than END_GTID ($END_GTID)"
    exit 1
fi

# Calculate total GTIDs to skip
TOTAL_GTIDS=$((END_GTID - START_GTID + 1))

echo "=== GTID Skip Configuration ==="
echo "Server UUID: $SERVER_UUID"
echo "GTID Range: $START_GTID to $END_GTID"
echo "Total GTIDs to skip: $TOTAL_GTIDS"
echo "MySQL Host: $MYSQL_HOST:$MYSQL_PORT"
echo "MySQL User: $MYSQL_USER"
echo "================================"
echo ""

# Confirm before proceeding
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Create temporary SQL file
SQL_FILE="/tmp/skip_gtids_$(date +%s).sql"

echo "Creating SQL script..."
echo "STOP SLAVE;" > $SQL_FILE

echo "Generating GTID skip commands..."
for ((i=$START_GTID; i<=$END_GTID; i++)); do
    echo "SET GTID_NEXT='$SERVER_UUID:$i'; BEGIN; COMMIT;" >> $SQL_FILE
    
    # Show progress for large ranges
    if [ $((i % 1000)) -eq 0 ]; then
        echo "Progress: $((i - START_GTID + 1))/$TOTAL_GTIDS GTIDs processed..."
    fi
done

echo "SET GTID_NEXT='AUTOMATIC';" >> $SQL_FILE
echo "START SLAVE;" >> $SQL_FILE
echo "SHOW SLAVE STATUS\\G" >> $SQL_FILE

echo ""
echo "Executing GTID skip script..."

# Build MySQL command with password handling
if [ -n "$MYSQL_PASSWORD" ]; then
    mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" -h"$MYSQL_HOST" -P"$MYSQL_PORT" < "$SQL_FILE"
else
    mysql -u"$MYSQL_USER" -h"$MYSQL_HOST" -P"$MYSQL_PORT" < "$SQL_FILE"
fi

# Check if MySQL command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "GTID skip operation completed successfully!"
else
    echo ""
    echo "Error: GTID skip operation failed!"
    echo "SQL file preserved at: $SQL_FILE"
    exit 1
fi

# Clean up
rm "$SQL_FILE"
echo "Temporary files cleaned up."