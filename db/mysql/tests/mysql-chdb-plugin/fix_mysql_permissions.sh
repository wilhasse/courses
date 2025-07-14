#!/bin/bash

echo "=== Fixing MySQL Permissions ==="
echo

# Check MySQL data directory
echo "1. Checking MySQL data directory:"
MYSQL_DATA_DIR=$(mysql teste -u root -pteste -NBe "SELECT @@datadir" 2>/dev/null || echo "/var/lib/mysql")
echo "Data directory: $MYSQL_DATA_DIR"

# Check ownership
echo
echo "2. Current ownership:"
ls -la $MYSQL_DATA_DIR/mysql.* | grep func

# Fix ownership if needed
echo
echo "3. Fixing ownership (requires sudo):"
sudo chown mysql:mysql $MYSQL_DATA_DIR/mysql.*
sudo chmod 660 $MYSQL_DATA_DIR/mysql.func*

# Check if MySQL is in read-only mode
echo
echo "4. Checking read_only status:"
mysql teste -u root -pteste -e "SHOW VARIABLES LIKE '%read_only%';" 2>/dev/null

# Try to disable read_only if needed
echo
echo "5. Attempting to disable read_only mode:"
mysql teste -u root -pteste -e "SET GLOBAL read_only = 0; SET GLOBAL super_read_only = 0;" 2>/dev/null

# Check disk space
echo
echo "6. Checking disk space:"
df -h $MYSQL_DATA_DIR

# Try to flush privileges
echo
echo "7. Flushing privileges:"
mysql teste -u root -pteste -e "FLUSH PRIVILEGES;" 2>/dev/null

# Test creating a simple function
echo
echo "8. Testing function creation:"
mysql teste -u root -pteste << 'EOF'
-- First drop if exists
DROP FUNCTION IF EXISTS test_func;

-- Try to create a simple function
DELIMITER //
CREATE FUNCTION test_func() RETURNS INT DETERMINISTIC
BEGIN
    RETURN 42;
END//
DELIMITER ;

-- Test it
SELECT test_func() as result;

-- Clean up
DROP FUNCTION test_func;
EOF

echo
echo "If the test function worked, try creating the UDF again."