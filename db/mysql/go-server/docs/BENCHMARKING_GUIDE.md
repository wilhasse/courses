# Benchmarking Guide

This guide explains how to benchmark the MySQL server across different storage backends to measure performance improvements.

## Overview

The benchmarking system allows you to:
- Compare query performance across storage backends (MySQL passthrough, LMDB, chDB, Hybrid)
- Measure queries per second (QPS) and average query latency
- Identify optimal storage backend for different workload types
- Track performance improvements as optimizations are added

## Quick Start

```bash
# Run benchmarks
./benchmark.sh

# Analyze results
./analyze_benchmark.py
```

## Benchmark Script (`benchmark.sh`)

The benchmark script automatically:
1. Starts the server with each storage backend
2. Runs a series of test queries
3. Measures performance metrics
4. Saves results to `benchmark_results.csv`

### Test Queries

The benchmark includes various query types:
- **Point Query**: `SELECT * FROM users WHERE id = 1`
- **Full Table Scan**: `SELECT * FROM users`
- **Aggregation**: `SELECT COUNT(*) FROM products`
- **Range Query**: `SELECT * FROM products WHERE price > 50`
- **Complex JOIN**: Multi-table join with GROUP BY

### Configuration

Edit `benchmark.sh` to adjust:
```bash
ITERATIONS=100      # Number of times to run each query
PORT=3312          # Server port
HOST=127.0.0.1     # Server host
```

### Adding Storage Backends

To benchmark additional backends, modify the loop in `benchmark.sh`:
```bash
# Test each backend
for backend in lmdb mysql chdb hybrid; do
    # ... benchmark code ...
done
```

## Results Analysis (`analyze_benchmark.py`)

The analysis script provides:
- Performance comparison across backends
- Best performer identification (⭐)
- Relative performance percentages
- Average QPS summary
- Performance insights

### Sample Output

```
BENCHMARK RESULTS COMPARISON
================================================================================

Point query:
-----------
  lmdb       - QPS:  1234.56 | Avg:   0.81ms | Relative: 100.0% ⭐
  mysql      - QPS:   456.78 | Avg:   2.19ms | Relative:  37.0%

Full table scan:
---------------
  chdb       - QPS:   890.12 | Avg:   1.12ms | Relative: 100.0% ⭐
  lmdb       - QPS:   234.56 | Avg:   4.26ms | Relative:  26.3%
```

## Interpreting Results

### Queries Per Second (QPS)
- Higher is better
- Indicates throughput capacity
- Important for high-concurrency scenarios

### Average Query Time
- Lower is better
- Measured in milliseconds
- Critical for user-facing applications

### Relative Performance
- Percentage compared to best performer
- Helps identify performance gaps
- Guides optimization efforts

## Storage Backend Characteristics

### MySQL Passthrough
- **Baseline**: Direct forwarding to MySQL
- **Use Case**: Compatibility testing, baseline measurement
- **Expected**: Highest latency due to network overhead

### LMDB
- **Strength**: Excellent for point queries and small datasets
- **Use Case**: Transactional workloads, hot data
- **Expected**: Very fast for key-value lookups

### chDB (ClickHouse)
- **Strength**: Columnar storage, analytical queries
- **Use Case**: Aggregations, full table scans
- **Expected**: 10-100x faster for analytical workloads

### Hybrid
- **Strength**: Best of both worlds
- **Use Case**: Mixed workloads
- **Expected**: Optimal performance across query types

## Benchmarking Workflow

1. **Establish Baseline**
   ```bash
   # Test with MySQL passthrough
   ./benchmark.sh
   ```

2. **Test Optimizations**
   ```bash
   # Test with different backends
   vim benchmark.sh  # Enable other backends
   ./benchmark.sh
   ```

3. **Compare Results**
   ```bash
   ./analyze_benchmark.py
   ```

4. **Iterate**
   - Identify bottlenecks
   - Implement optimizations
   - Re-benchmark

## Advanced Benchmarking

### Custom Queries

Add custom queries to benchmark your specific workload:
```bash
QUERIES=(
    "YOUR_QUERY|Description"
    # ... more queries
)
```

### Load Testing

For concurrent load testing, use tools like:
- `mysqlslap` - MySQL's built-in tool
- `sysbench` - Advanced database benchmarking
- `wrk` - HTTP benchmarking (if using HTTP interface)

Example with mysqlslap:
```bash
mysqlslap \
    --host=127.0.0.1 \
    --port=3312 \
    --concurrency=10 \
    --iterations=100 \
    --query="SELECT * FROM testdb.users WHERE id = 1"
```

### Profiling

Enable profiling to identify bottlenecks:
```bash
# Run with debug mode
./bin/mysql-server --storage lmdb --debug

# Check execution traces in logs
tail -f debug.log
```

## Performance Optimization Tips

1. **Query Optimization**
   - Add appropriate indexes
   - Optimize JOIN order
   - Use query hints when available

2. **Storage Selection**
   - Use LMDB for OLTP workloads
   - Use chDB for OLAP workloads
   - Use Hybrid for mixed workloads

3. **Configuration Tuning**
   - Adjust connection pool sizes
   - Configure memory limits
   - Enable compression for chDB

4. **Monitoring**
   - Track cache hit rates
   - Monitor query patterns
   - Identify slow queries

## Continuous Benchmarking

For production environments:
1. Set up automated benchmarking in CI/CD
2. Track performance over time
3. Alert on performance regressions
4. Maintain performance baselines

## Troubleshooting

### Server Won't Start
- Check if port is already in use
- Verify LMDB library path: `export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH`
- Check logs: `tail benchmark_*.log`

### Inconsistent Results
- Ensure sufficient warm-up queries
- Close other applications
- Run multiple iterations
- Consider system load

### Missing Backends
- Verify backend is implemented
- Check dependencies (chDB requires Python)
- Review configuration files

## Next Steps

1. Run baseline benchmarks with current data
2. Identify performance bottlenecks
3. Implement optimizations
4. Measure improvements
5. Document findings

The benchmarking framework provides the foundation for data-driven performance optimization, supporting the "zero to hero" journey from simple MySQL passthrough to highly optimized hybrid storage.