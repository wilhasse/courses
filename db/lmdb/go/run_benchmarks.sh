#!/bin/bash

# LMDB Benchmark Runner
# Equivalent to B+ Tree benchmark runner for fair comparison

set -e

echo "🚀 LMDB Go Benchmark Runner"
echo "=================================="
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

echo "🔧 Environment configured:"
echo "   CGO_CFLAGS: $CGO_CFLAGS"
echo "   CGO_LDFLAGS: $CGO_LDFLAGS"
echo "   LD_LIBRARY_PATH: $(echo $LD_LIBRARY_PATH | cut -d: -f1)"
echo

# Clean up any existing benchmark databases
echo "🧹 Cleaning up previous benchmark data..."
rm -rf ./benchmark_testdb
rm -rf ./testdb

echo "📊 Running LMDB benchmarks..."
echo

# Option 1: Run the standalone comparison runner (equivalent to Zig benchmark format)
if [ "$1" = "--standalone" ] || [ "$1" = "-s" ]; then
    echo "Running standalone benchmark (Zig-style output)..."
    go run ./cmd/benchmark_runner.go

# Option 2: Run Go benchmark tests (Go benchmark format)
elif [ "$1" = "--gobench" ] || [ "$1" = "-g" ]; then
    echo "Running Go benchmark tests..."
    go test -bench=BenchmarkComparison ./benchmark -benchtime=3s -timeout=30m

# Option 3: Run specific benchmark
elif [ "$1" = "--specific" ] || [ "$1" = "-sp" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --specific <benchmark_name>"
        echo "Available benchmarks:"
        echo "  SequentialInsert"
        echo "  RandomInsert" 
        echo "  Lookup"
        echo "  Iteration"
        echo "  RangeQuery"
        echo "  Comparison"
        exit 1
    fi
    echo "Running specific benchmark: $2..."
    go test -bench="Benchmark$2" ./benchmark -benchtime=3s -timeout=10m

# Option 4: Run all benchmarks (default)
else
    echo "Running comprehensive benchmark suite..."
    echo
    
    echo "1️⃣  Running standalone comparison (for cross-language comparison):"
    echo "   (Results in µs - comparable with Zig/Rust/Go B+ Tree benchmarks)"
    echo
    go run ./cmd/benchmark_runner.go
    
    echo
    echo "2️⃣  Running Go benchmark tests (for detailed profiling):"
    echo "   (Go standard benchmark format)"
    echo
    go test -bench=BenchmarkComparison ./benchmark -benchtime=1s -timeout=15m | head -50
fi

echo
echo "✅ LMDB benchmarks complete!"
echo
echo "📝 To compare with B+ Tree benchmarks:"
echo "   cd /home/cslog/BPlusTree3"
echo "   ./scripts/run_all_benchmarks.py -v"
echo
echo "🔍 Benchmark options:"
echo "   $0                    # Run all benchmarks"
echo "   $0 --standalone       # Run standalone comparison only"
echo "   $0 --gobench          # Run Go benchmark tests only"
echo "   $0 --specific <name>  # Run specific benchmark"
echo