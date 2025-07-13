#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== MySQL UDF Table-Valued Function Simulation Test ==="
echo ""

echo "1. Building the plugin..."
"$SCRIPT_DIR/build_tvf.sh"

echo ""
echo "2. Installing the plugin..."
"$SCRIPT_DIR/install_tvf.sh"

echo ""
echo "3. Running the test queries..."
echo ""
echo "=================================================="
echo "TEST RESULTS:"
echo "=================================================="
mysql -u root -pteste < "$PROJECT_ROOT/tests/test_tvf_join.sql" 2>&1 | grep -v "Warning"

echo ""
echo "=================================================="
echo "Test completed!"
echo "=================================================="
echo ""
echo "To uninstall the plugin, run: $SCRIPT_DIR/uninstall_tvf.sh"