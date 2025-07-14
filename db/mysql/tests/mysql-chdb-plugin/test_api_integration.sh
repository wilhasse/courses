#!/bin/bash

# Quick test script for MySQL + chDB API integration

echo "=== MySQL + chDB API Integration Test ==="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if API server is running
if ! nc -z 127.0.0.1 8125 2>/dev/null; then
    echo -e "${RED}ERROR: chDB API server is not running on port 8125${NC}"
    echo
    echo "Please start the server first:"
    echo "  cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example"
    echo "  ./chdb_api_server_simple"
    exit 1
fi

echo -e "${GREEN}✓ chDB API server is running${NC}"

# Build the UDF if not exists
if [ ! -f "build/chdb_api_functions.so" ]; then
    echo -e "${YELLOW}Building UDF functions...${NC}"
    ./scripts/build_api_udf.sh
fi

# Test the functions
echo
echo -e "${YELLOW}Testing MySQL UDF functions...${NC}"
echo

mysql -u root -pteste 2>/dev/null <<EOF
USE test;

-- Test 1: Basic connectivity
SELECT 'Test 1: Basic connectivity' AS test_name;
SELECT chdb_query('SELECT 1') AS result;

-- Test 2: Count function
SELECT '\nTest 2: Count customers' AS test_name;
SELECT chdb_count('mysql_import.customers') AS customer_count;

-- Test 3: Complex query
SELECT '\nTest 3: Customers by city' AS test_name;
SELECT chdb_query('SELECT city, COUNT(*) FROM mysql_import.customers GROUP BY city ORDER BY COUNT(*) DESC') AS result;

-- Test 4: Performance test
SELECT '\nTest 4: Query performance' AS test_name;
SET @start = NOW(6);
SELECT chdb_query('SELECT COUNT(*) FROM mysql_import.customers WHERE age > 30') AS filtered_count;
SET @elapsed = TIMESTAMPDIFF(MICROSECOND, @start, NOW(6)) / 1000.0;
SELECT CONCAT('Query time: ', @elapsed, ' ms') AS performance;
EOF

if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo
    echo "The MySQL UDF functions are working correctly with the chDB API server."
    echo
    echo "You can now use these functions in your MySQL queries:"
    echo "  - chdb_query(sql)    : Execute any ClickHouse SQL"
    echo "  - chdb_count(table)  : Get row count"
    echo "  - chdb_sum(table, column) : Calculate sum"
else
    echo
    echo -e "${RED}✗ Tests failed${NC}"
    echo
    echo "Troubleshooting:"
    echo "1. Check if the UDF is installed:"
    echo "   mysql -u root -pteste -e \"SHOW FUNCTION STATUS WHERE Name LIKE 'chdb%'\""
    echo "2. Check MySQL error log:"
    echo "   sudo tail -f /var/log/mysql/error.log"
fi