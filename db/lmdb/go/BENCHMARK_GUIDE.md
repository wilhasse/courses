# LMDB vs B+ Tree Benchmark Comparison Guide

## 🎯 Overview

This benchmark suite provides direct performance comparison between **LMDB** (Lightning Memory-Mapped Database) and **in-memory B+ Tree** implementations across multiple languages (Zig, Go, Rust).

### Why This Comparison Matters

- **LMDB**: Persistent, ACID-compliant, memory-mapped database with B+ tree structure
- **B+ Tree**: In-memory data structure optimized for ordered access
- **Fair Comparison**: Both use B+ tree structures internally, allowing meaningful performance analysis

## 🚀 Quick Start

### Run LMDB Benchmarks
```bash
# Setup and run all LMDB benchmarks
./run_benchmarks.sh

# Or step by step:
./setup-lmdb.sh              # Setup LMDB C library
./run_benchmarks.sh --standalone  # Run comparison benchmarks
```

### Run B+ Tree Benchmarks (for comparison)
```bash
cd /home/cslog/BPlusTree3
./scripts/run_all_benchmarks.py -v
```

## 📊 Benchmark Operations

Both benchmark suites test the same operations with identical parameters:

### Operations Tested
1. **Sequential Insert**: Insert keys 0, 1, 2, ... N
2. **Random Insert**: Insert keys in random order
3. **Random Lookup**: Random key lookups in populated structure
4. **Full Iteration**: Complete traversal of all entries
5. **Range Query**: Retrieve subset of entries (1000 items)

### Data Sizes
- 100, 1,000, 10,000, 100,000 elements

### Metrics
- **Time**: Microseconds (µs) for direct comparison
- **Throughput**: Operations per second
- **Latency**: Microseconds per operation

## 🔧 Benchmark Commands

### LMDB Benchmarks

```bash
# Complete benchmark suite
./run_benchmarks.sh

# Standalone comparison (Zig-style output for cross-language comparison)
./run_benchmarks.sh --standalone

# Go benchmark format (detailed profiling)
./run_benchmarks.sh --gobench

# Specific operation
./run_benchmarks.sh --specific SequentialInsert
./run_benchmarks.sh --specific Lookup
./run_benchmarks.sh --specific RangeQuery
```

### B+ Tree Benchmarks

```bash
cd /home/cslog/BPlusTree3

# All languages (Zig, Go, Rust)
./scripts/run_all_benchmarks.py -v

# Specific language
./scripts/run_all_benchmarks.py --go-only
./scripts/run_all_benchmarks.py --zig-only
./scripts/run_all_benchmarks.py --rust-only
```

## 📈 Expected Performance Characteristics

### LMDB Advantages
- **Persistence**: Data survives application restarts
- **ACID Transactions**: Consistency guarantees
- **Memory Efficiency**: Memory-mapped, zero-copy reads
- **Concurrent Readers**: Multiple processes can read simultaneously
- **Production Ready**: Battle-tested in real applications

### In-Memory B+ Tree Advantages
- **Speed**: No disk I/O overhead
- **Simplicity**: Pure in-memory operations
- **Language Optimization**: Highly optimized per language

### Expected Results

Based on structure and use case:

**LMDB Performance:**
- Sequential Insert: ~10-100 µs/op (depends on batch size)
- Random Insert: ~50-200 µs/op 
- Lookup: ~1-10 µs/op
- Iteration: ~0.01-0.1 µs/op
- Range Query: ~10-100 µs/op

**In-Memory B+ Tree Performance:**
- Sequential Insert: ~0.1-10 µs/op
- Random Insert: ~1-50 µs/op
- Lookup: ~0.1-1 µs/op  
- Iteration: ~0.001-0.01 µs/op
- Range Query: ~1-10 µs/op

**Performance Factors:**
- LMDB may be 2-10x slower due to persistence overhead
- LMDB excels in memory efficiency for large datasets
- B+ Tree excels in pure speed for in-memory workloads

## 🧪 Running Comprehensive Comparison

### 1. Setup Both Projects
```bash
# Setup LMDB benchmarks
cd /home/cslog/courses/db/lmdb/go
./setup-lmdb.sh

# Verify B+ Tree setup
cd /home/cslog/BPlusTree3
./scripts/check_and_run.py
```

### 2. Run LMDB Benchmarks
```bash
cd /home/cslog/courses/db/lmdb/go
./run_benchmarks.sh --standalone > lmdb_results.txt
```

### 3. Run B+ Tree Benchmarks  
```bash
cd /home/cslog/BPlusTree3
./scripts/run_all_benchmarks.py -v > btree_results.txt
```

### 4. Compare Results
```bash
# View LMDB results
cat /home/cslog/courses/db/lmdb/go/lmdb_results.txt

# View B+ Tree results
cat /home/cslog/BPlusTree3/btree_results.txt

# Side-by-side comparison
echo "=== LMDB Results ===" && cat lmdb_results.txt && echo && echo "=== B+ Tree Results ===" && cat btree_results.txt
```

## 🔍 Detailed Analysis

### Use LMDB When:
- **Persistence Required**: Data must survive application restarts
- **Large Datasets**: Memory-mapped access for datasets larger than RAM
- **ACID Properties**: Need transaction guarantees
- **Multiple Processes**: Concurrent reader access required
- **Production Reliability**: Need battle-tested database engine

### Use In-Memory B+ Tree When:
- **Maximum Speed**: Pure in-memory performance critical
- **Temporary Data**: Data doesn't need persistence
- **Embedded Use**: Part of larger application, no external dependencies
- **Custom Logic**: Need full control over data structure behavior

### Architecture Differences

**LMDB:**
- Persistent storage with memory-mapped files
- Copy-on-write transactions
- Operating system page cache integration
- C library with Go bindings (CGO overhead)

**In-Memory B+ Tree:**
- Pure language implementation
- Direct memory allocation
- No I/O or system calls
- Language-specific optimizations

## 🛠️ Troubleshooting

### LMDB Issues
```bash
# Build issues
source .envrc
go build

# Runtime library issues  
./run.sh  # Use provided script

# Permission issues
chmod +x setup-lmdb.sh run_benchmarks.sh
```

### B+ Tree Issues
```bash
cd /home/cslog/BPlusTree3

# Missing dependencies
./scripts/run_all_benchmarks.py --auto-install

# Specific language issues
./scripts/test_rust_install.py
zig version
go version
```

## 📋 Output Format

Both benchmark suites output results in comparable formats:

```
Operation Name                  Ops    Time(ms)   Ops/sec    µs/op
Sequential insertion         100000      45.23    2,211,234    0.45
Random insertion            100000      78.91    1,267,421    0.79
Random lookup (hit)         100000      12.34    8,103,727    0.12
Full iteration              100000       3.45   28,985,507    0.03
Range query (1000 items)       100      15.67      6,383      156.7
```

## 🎯 Use Case Recommendations

### Choose LMDB for:
- **Database Applications**: Key-value stores, caches with persistence
- **Configuration Storage**: Application settings that persist
- **Logging Systems**: High-throughput write-heavy workloads
- **Embedded Databases**: SQLite alternative for simpler use cases

### Choose In-Memory B+ Tree for:
- **Real-time Systems**: Ultra-low latency requirements
- **Gaming**: In-game data structures
- **Financial Trading**: Microsecond-sensitive applications  
- **Caching Layers**: Pure in-memory caches

### Hybrid Approach:
- Use B+ Tree for hot data (frequent access)
- Use LMDB for cold data (persistent storage)
- Combine both for tiered storage architecture

## 🔗 References

- [LMDB Documentation](http://www.lmdb.tech/doc/)
- [golmdb Go Bindings](https://pkg.go.dev/wellquite.org/golmdb)
- [B+ Tree Project](https://github.com/cslog/BPlusTree3)
- [LMDB Paper](https://www.symas.com/lmdb)

This benchmark comparison provides comprehensive performance analysis between persistent LMDB and in-memory B+ Tree implementations! 🚀