#!/bin/bash

echo "=== Direct Test of Wrapper Components ==="
echo

# Test the helper directly
echo "1. Testing chdb_query_helper directly:"
if [ -f "./chdb_query_helper" ]; then
    echo "   Query: SELECT COUNT(*) FROM mysql_import.customers"
    ./chdb_query_helper "SELECT COUNT(*) FROM mysql_import.customers"
    echo
    
    echo "   Query: SELECT 1+1 as result"
    ./chdb_query_helper "SELECT 1+1 as result"
    echo
else
    echo "   ERROR: chdb_query_helper not found. Build it first with:"
    echo "   g++ -o chdb_query_helper chdb_query_helper.cpp -ldl -std=c++11"
fi

echo
echo "2. Testing via command line (simulating what the plugin does):"
echo '   echo "SELECT COUNT(*) FROM mysql_import.customers" | ./chdb_query_helper'
echo "SELECT COUNT(*) FROM mysql_import.customers" | ./chdb_query_helper 2>&1

echo
echo "3. If MySQL won't accept UDFs, we can create a stored procedure wrapper:"
cat << 'EOF'

-- Alternative: Use a stored procedure that calls our helper via sys_exec
-- (requires sys_exec UDF or similar)

DELIMITER //
CREATE PROCEDURE query_clickhouse(IN query_text VARCHAR(1000))
BEGIN
    DECLARE result TEXT;
    -- This would need sys_exec or similar UDF
    -- SET result = sys_exec(CONCAT('./chdb_query_helper "', query_text, '"'));
    SELECT CONCAT('Would execute: ./chdb_query_helper "', query_text, '"') as command;
END//
DELIMITER ;

-- Usage:
CALL query_clickhouse('SELECT COUNT(*) FROM mysql_import.customers');
EOF

echo
echo "4. Python alternative (if you have PyMySQL):"
cat << 'EOF'
# query_chdb.py
import subprocess
import pymysql

def query_chdb(query):
    result = subprocess.run(['./chdb_query_helper', query], 
                          capture_output=True, text=True)
    return result.stdout.strip()

# Example usage
count = query_chdb("SELECT COUNT(*) FROM mysql_import.customers")
print(f"Customer count: {count}")
EOF