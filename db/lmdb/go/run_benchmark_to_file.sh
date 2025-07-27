#!/bin/bash

# LMDB Benchmark Runner - Save Results to File
# Generates benchmark results file for comparison

set -e

RESULTS_FILE="lmdb_benchmark_results.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "🚀 LMDB Go Benchmark Runner - Saving to $RESULTS_FILE"
echo "====================================================="
echo

# Check if LMDB library is set up
if [ ! -d "lmdb-lib/lib" ]; then
    echo "⚠️  LMDB C library not found. Setting up..."
    ./setup-lmdb.sh
    echo "✅ LMDB C library setup complete"
    echo
fi

# Set environment variables
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"

# Clean up any existing benchmark databases
echo "🧹 Cleaning up previous benchmark data..."
rm -rf ./benchmark_testdb
rm -rf ./testdb

echo "📊 Running LMDB benchmarks and saving to $RESULTS_FILE..."
echo

# Create results file with header
cat > "$RESULTS_FILE" <<EOF
LMDB Benchmark Results
======================
Generated: $TIMESTAMP
Host: $(hostname)
Go Version: $(go version)
LMDB Library: $(pwd)/lmdb-lib/lib/liblmdb.so

EOF

# Run standalone benchmark and append to file
echo "Running standalone LMDB benchmarks..." | tee -a "$RESULTS_FILE"
echo "=====================================" | tee -a "$RESULTS_FILE"
echo | tee -a "$RESULTS_FILE"

go run ./cmd/benchmark_runner.go 2>&1 | tee -a "$RESULTS_FILE"

echo | tee -a "$RESULTS_FILE"
echo "Benchmark completed at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RESULTS_FILE"

echo
echo "✅ LMDB benchmark results saved to: $RESULTS_FILE"
echo
echo "📝 To compare with B+ Tree benchmarks:"
echo "   cd /home/cslog/BPlusTree3"
echo "   ./scripts/run_all_benchmarks.py -v > btree_results.txt"
echo
echo "🔍 View results:"
echo "   cat $RESULTS_FILE"
echo