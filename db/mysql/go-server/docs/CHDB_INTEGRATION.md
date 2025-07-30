# chDB Integration Guide

This guide explains how chDB (embedded ClickHouse) is integrated into the MySQL server for high-performance analytical queries.

## Overview

chDB brings ClickHouse's columnar storage and analytical query engine as an embedded library, providing:
- 100-1000x faster analytical queries
- Columnar data compression (10-50x)
- Vectorized query execution
- Native SQL support

## Architecture

### Three-Tier Storage System

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Hot Data      │     │  Analytical     │     │   Remote        │
│     LMDB        │     │     chDB        │     │     MySQL       │
│  (< 1M rows)    │     │  (> 10M rows)   │     │   (Source)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Storage Selection Logic

The hybrid storage system automatically routes tables based on:

1. **Table Size**
   - < 1M rows → LMDB (fast transactional)
   - > 10M rows → chDB (analytical)
   - 1M-10M rows → Decision based on other factors

2. **Access Patterns**
   - High modification rate → LMDB
   - Mostly reads with aggregations → chDB
   - Frequent point queries → LMDB

3. **Table Name Heuristics**
   - Names containing: `fact`, `events`, `logs`, `metrics` → chDB
   - Names containing: `dim`, `lookup`, `config` → LMDB

4. **Schema Analysis**
   - > 60% numeric columns → chDB
   - Many string/text columns → LMDB

## Installation

### Prerequisites

1. Install chDB library:
```bash
./install-chdb.sh
```

This script automatically detects CPU capabilities and installs the appropriate version (with or without AVX support).

### Build Configuration

The project requires CGO for LMDB. Build with:

```bash
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib -llmdb"
go build .
```

Or simply use:
```bash
make build
```

## Configuration

### Command Line Flags

```bash
# Select storage backend
./mysql-server-example --storage hybrid  # Default: uses all three tiers
./mysql-server-example --storage chdb    # chDB only
./mysql-server-example --storage lmdb    # LMDB only

# Configure thresholds
./mysql-server-example \
  --hot-data-threshold 500000 \      # Max rows for LMDB
  --analytical-threshold 5000000     # Min rows for chDB

# chDB specific settings
./mysql-server-example \
  --chdb-max-memory 8G \            # Memory limit
  --chdb-max-threads 8 \            # Parallelism
  --chdb-compression true           # Enable compression
```

### Environment Variables

```bash
export STORAGE_BACKEND=hybrid
export HOT_DATA_THRESHOLD=1000000
export ANALYTICAL_THRESHOLD=10000000
export CHDB_MAX_MEMORY=16G
export CHDB_MAX_THREADS=16
```

### Configuration File

Create `config.yaml` (future enhancement):
```yaml
storage:
  backend: hybrid
  lmdb_path: ./data
  chdb_path: ./chdb_data
  
hybrid:
  hot_data_threshold: 1000000
  analytical_threshold: 10000000
  auto_migration: true
  
chdb:
  max_memory: 8G
  max_threads: 8
  compression: true
  compression_method: lz4
```

## Usage Examples

### Creating Tables

Tables are automatically routed to the appropriate backend:

```sql
-- Small dimension table → LMDB
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(200)
);

-- Large fact table → chDB
CREATE TABLE events_log (
    timestamp DATETIME,
    user_id INT,
    event_type VARCHAR(50),
    value DECIMAL(10,2),
    metadata TEXT
);

-- Explicit routing (future enhancement)
CREATE TABLE sales_facts (
    date DATE,
    product_id INT,
    quantity INT,
    revenue DECIMAL(10,2)
) ENGINE = 'chdb';
```

### Analytical Queries

chDB excels at analytical queries:

```sql
-- Aggregations (100x faster in chDB)
SELECT 
    DATE(timestamp) as day,
    COUNT(*) as events,
    SUM(value) as total_value
FROM events_log
WHERE timestamp > DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(timestamp)
ORDER BY day;

-- Window functions
SELECT 
    user_id,
    value,
    AVG(value) OVER (PARTITION BY user_id ORDER BY timestamp 
                     ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) as moving_avg
FROM events_log;

-- Complex joins with aggregations
SELECT 
    u.name,
    COUNT(DISTINCT e.event_type) as unique_events,
    SUM(e.value) as total_value
FROM users u
JOIN events_log e ON u.id = e.user_id
GROUP BY u.id, u.name
HAVING total_value > 1000;
```

### Table Migration

Move tables between storage backends:

```sql
-- Check current storage backend
SELECT table_name, storage_backend, row_count
FROM information_schema.table_stats;

-- Migrate table to chDB (admin command)
CALL migrate_table('mydb', 'mytable', 'chdb');

-- View migration candidates
SELECT * FROM system.migration_candidates;
```

## Implementation Details

### ChDBStorage (`pkg/storage/chdb_storage.go`)

Implements the Storage interface with chDB backend:

- **Type Mapping**: Converts MySQL types to ClickHouse types
- **MergeTree Engine**: Uses for optimal analytical performance
- **Batch Operations**: Optimized for bulk inserts
- **Query Execution**: Direct SQL passthrough for analytics

### HybridStorage (`pkg/storage/hybrid_storage.go`)

Intelligent routing layer that:

- **Routes Queries**: Directs to appropriate backend
- **Tracks Metadata**: Table size, access patterns
- **Manages Migration**: Moves tables between backends
- **Optimizes Performance**: Based on workload patterns

### Table Metadata (`pkg/storage/table_metadata.go`)

Tracks detailed statistics:

- Row count and size
- Query patterns (SELECT/INSERT/UPDATE/DELETE)
- Join and aggregation frequency
- Access time patterns
- Performance metrics

## Performance Characteristics

### LMDB (Hot Data)
- **Point Queries**: < 0.1ms
- **Small Joins**: 1-10ms
- **Updates**: Fast, ACID compliant
- **Best For**: OLTP, dimension tables

### chDB (Analytical)
- **Aggregations**: 100-1000x faster than row storage
- **Large Scans**: Columnar compression
- **Parallel Processing**: Uses all CPU cores
- **Best For**: OLAP, fact tables, time-series

### Hybrid Benefits
- **Automatic Optimization**: Tables migrate based on usage
- **Best of Both Worlds**: Fast transactions + fast analytics
- **Transparent**: No application changes needed

## Troubleshooting

### Common Issues

1. **Build Errors**
   ```bash
   # CGO errors
   export CGO_ENABLED=1
   export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
   export CGO_LDFLAGS="-L$(pwd)/lmdb-lib -llmdb"
   ```

2. **chDB Installation**
   ```bash
   # Check if chDB is installed
   python3 -c "import chdb; print(chdb.__version__)"
   
   # Reinstall if needed
   ./install-chdb.sh
   ```

3. **Memory Issues**
   ```bash
   # Limit chDB memory usage
   ./mysql-server-example --chdb-max-memory 2G
   ```

### Performance Tuning

1. **Table Placement**
   ```sql
   -- Mark table as analytical
   CALL set_table_analytical('mydb', 'mytable', true);
   ```

2. **Memory Configuration**
   - Set `--chdb-max-memory` to 50-70% of available RAM
   - Use `--chdb-max-threads` = number of CPU cores

3. **Monitoring**
   ```sql
   -- View storage statistics
   SELECT * FROM system.storage_stats;
   
   -- Check query performance
   SELECT * FROM system.query_log ORDER BY duration_ms DESC LIMIT 10;
   ```

## Future Enhancements

1. **Materialized Views** in chDB for pre-aggregated data
2. **Automatic Data Tiering** based on age
3. **Query Result Caching** for repeated analytical queries
4. **Distributed chDB** for scale-out analytics
5. **Real-time Sync** from LMDB to chDB

## References

- [chDB Documentation](https://github.com/chdb-io/chdb)
- [ClickHouse SQL Reference](https://clickhouse.com/docs/en/sql-reference)
- [LMDB Integration Guide](LMDB_INTEGRATION.md)
- [Hybrid Query System](HYBRID_QUERY_SYSTEM.md)