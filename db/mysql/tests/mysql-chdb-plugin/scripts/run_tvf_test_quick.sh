#!/bin/bash

# Quick test script that only runs the queries without rebuilding/reinstalling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== MySQL UDF Table-Valued Function Quick Test ==="
echo ""
echo "Running test queries (assuming functions are already installed)..."
echo ""
echo "=================================================="
echo "TEST RESULTS:"
echo "=================================================="

# Run the test and show output, filtering warnings
mysql -u root -pteste < "$PROJECT_ROOT/tests/test_tvf_join.sql" 2>&1 | grep -v "Warning"

echo ""
echo "=================================================="
echo "Test completed!"
echo "=================================================="