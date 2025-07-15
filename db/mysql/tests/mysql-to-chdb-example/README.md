# MySQL to ClickHouse Data Transfer with chDB

This project demonstrates how to extract data from MySQL and load it into ClickHouse using the chDB library, with multiple approaches for different use cases.

## Overview

This project provides various tools to:
- Import MySQL data into ClickHouse (chDB) with high performance
- Convert between different ClickHouse storage engines for optimal query performance
- Serve ClickHouse queries via API servers
- Test and benchmark performance

## Main Components

### Data Import Tools

#### 1. **historico_log** (Recommended for All Dataset Sizes)
The main production-ready tool for importing MySQL tables into chDB using the Log engine. Now with built-in memory optimizations.

```bash
./historico_log <host> <user> <password> <database> [options]
```

Features:
- Row-based chunking (50,000 rows per chunk) for predictable memory usage
- Streaming results using `mysql_use_result()` to minimize memory footprint
- Imports both HISTORICO and HISTORICO_TEXTO tables
- Batch inserts (500 rows per INSERT) for optimal performance
- Progress tracking with ETA calculation
- Resume capability with offset option
- No memory growth issues even with billions of rows

Options:
- `--skip-texto`: Skip loading HISTORICO_TEXTO table (only load HISTORICO)
- `--row-count <count>`: Provide total row count to skip the slow COUNT(*) query
- `--offset <offset>`: Start processing from this row number (for resuming)

Examples:
```bash
# First run - will perform COUNT(*) query
./historico_log 172.16.120.10 appl_cslog mypassword cslog_siscom_prod --skip-texto

# Subsequent runs - skip COUNT(*) for faster startup
./historico_log 172.16.120.10 appl_cslog mypassword cslog_siscom_prod --skip-texto --row-count 32424049

# Resume from row 5,000,000 after interruption
./historico_log 172.16.120.10 appl_cslog mypassword cslog_siscom_prod --skip-texto --row-count 32424049 --offset 5000000

# Load both tables (HISTORICO and HISTORICO_TEXTO)
./historico_log 172.16.120.10 appl_cslog mypassword cslog_siscom_prod
```

Output Example:
```
[2024-01-15 14:30:45] Starting historico_log...
[2024-01-15 14:30:45] Using provided row count: 32424049
Total rows to process: 32424049

[2024-01-15 14:30:46] Processing chunk 1/649 (rows 0-50000 of 32424049)...
  HISTORICO: 50000 rows loaded for this chunk
  [2024-01-15 14:31:02] Chunk 1 completed in 16 seconds (avg: 3125 rows/sec)
  Progress: 0.2% - ETA: 172 minutes
...
```

Performance Tips:
- Use `--skip-texto` if you only need HISTORICO data (much faster)
- After first run, always use `--row-count` to skip the COUNT(*) query
- The tool processes exactly 50,000 rows per chunk for consistent memory usage
- Memory usage stays constant regardless of dataset size

#### 2. **historico_feeder**
Alternative implementation with advanced features:
- Configurable chunk size
- Test mode for debugging
- Automatic chDB library restart after many operations
- More detailed error handling

```bash
./historico_feeder <host> <user> <password> <database> [--test]
```

#### 3. **feed_data_v2** / **feed_data**
Original tools for importing sample customer/order data.

### Query and Conversion Tools

#### 1. **convert_to_mergetree**
Converts Log engine tables to MergeTree for better query performance.

```bash
./convert_to_mergetree
```

Converts:
- `historico` → `historico_mt`
- `historico_texto` → `historico_texto_mt`

Shows performance comparisons between engines.

#### 2. **test_performance**
Comprehensive performance testing tool comparing Log vs MergeTree engines.

```bash
./test_performance
```

Tests include:
- Range queries
- Point lookups
- Aggregations
- Text searches
- Join operations

#### 3. **execute_sql** (Interactive SQL Query Tool)
General-purpose SQL query executor for chDB/ClickHouse.

```bash
./execute_sql [options] "SQL query"
```

Options:
- `-f, --format <format>`: Output format (default: Pretty)
  - Pretty: Human-readable table
  - TSV: Tab-separated values
  - CSV: Comma-separated values
  - JSON: JSON format
  - Vertical: One value per line

Examples:
```bash
# Simple count
./execute_sql "SELECT COUNT(*) FROM mysql_import.historico"

# Export to CSV
./execute_sql -f CSV "SELECT * FROM mysql_import.historico LIMIT 1000" > export.csv

# Complex analytics with pretty output
./execute_sql "SELECT codigo, COUNT(*) as cnt FROM mysql_import.historico GROUP BY codigo ORDER BY cnt DESC LIMIT 10"

# Pipe to other tools
./execute_sql -f TSV "SELECT id_contr, seq FROM mysql_import.historico WHERE codigo = 51" | awk '{print $1}'
```

#### 4. **query_data_v2** / **query_data**
Query tools for the sample customer/order data.

### API Server Components

#### 1. **chdb_api_server_simple** (Recommended)
Lightweight HTTP API server for ClickHouse queries.

```bash
./chdb_api_server_simple  # Listens on port 8125
```

Usage:
```bash
curl -X POST http://localhost:8125/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT COUNT(*) FROM mysql_import.historico"}'
```

#### 2. **chdb_api_server** / **chdb_api_client**
Protocol Buffer-based API for high-performance applications.

## Prerequisites

1. **MySQL server** with access credentials
2. **MySQL client libraries**:
   ```bash
   sudo apt-get install libmysqlclient-dev  # Ubuntu/Debian
   ```
3. **C++ compiler** with C++17 support
4. **chDB library**:
   ```bash
   cd /home/cslog/chdb
   make build
   ```
5. **Protocol Buffers** (for API server):
   ```bash
   sudo apt-get install protobuf-compiler libprotobuf-dev
   ```

## Building

```bash
# Build all tools
make all

# Build specific tools
make historico_log
make convert_to_mergetree
make test_performance
```

## Typical Workflow

### 1. Import MySQL Data

#### First Time Import
```bash
# Import only HISTORICO (recommended for initial testing)
./historico_log 172.16.120.10 user password database --skip-texto

# Note the total row count from the output for future runs
```

#### Subsequent Imports (Much Faster)
```bash
# Skip the slow COUNT(*) query by providing row count
./historico_log 172.16.120.10 user password database --skip-texto --row-count 32424049
```

#### Resume After Interruption
```bash
# If import was interrupted at chunk 200 (10,000,000 rows), resume from there
./historico_log 172.16.120.10 user password database --skip-texto --row-count 32424049 --offset 10000000
```

#### Import Both Tables
```bash
# Import HISTORICO and HISTORICO_TEXTO (slower, needs more work for row-based approach)
./historico_log 172.16.120.10 user password database
```

Memory Usage: Stays constant at ~200-300MB regardless of table size!

### 2. Convert to MergeTree for Better Performance
```bash
./convert_to_mergetree
```

This creates MergeTree versions of your tables with "_mt" suffix for much faster queries.

### 3. Test Performance
```bash
./test_performance
```

Compare query performance between Log and MergeTree engines.

### 4. Query Your Data

#### Quick Status Check
```bash
# Check row counts
./execute_sql "SELECT 'historico' as table, COUNT(*) as rows FROM mysql_import.historico 
UNION ALL 
SELECT 'historico_texto', COUNT(*) FROM mysql_import.historico_texto"

# Check data range
./execute_sql "SELECT MIN(data) as earliest, MAX(data) as latest FROM mysql_import.historico"
```

#### Run Analytics
```bash
# Top 10 most frequent codes
./execute_sql "SELECT codigo, COUNT(*) as frequency FROM mysql_import.historico GROUP BY codigo ORDER BY frequency DESC LIMIT 10"

# Export specific data
./execute_sql -f CSV "SELECT * FROM mysql_import.historico WHERE codigo = 51 AND data >= '2024-01-01'" > codigo_51_2024.csv
```

### 5. Start API Server (Optional)
```bash
./chdb_api_server_simple
```

Now you can query via HTTP:
```bash
curl -X POST http://localhost:8125/query \
  -d '{"query": "SELECT COUNT(*) FROM mysql_import.historico_mt"}'
```

## Data Storage

- **Location**: `./clickhouse_data/`
- **Engines**:
  - **Log**: Simple, fast writes, slower queries
  - **MergeTree**: Indexed, slower writes, much faster queries

## Performance Tips

1. **For Import**: Use `historico_log` with Log engine
2. **For Queries**: Convert to MergeTree with `convert_to_mergetree`
3. **Batch Size**: Default 500-1000 rows per INSERT
4. **Chunk Size**: Default 10,000 rows per MySQL query

## Table Schemas

### HISTORICO
```sql
CREATE TABLE historico (
    id_contr Int32,
    seq UInt16,
    id_funcionario Int32,
    id_tel Int32,
    data DateTime,
    codigo UInt16,
    modo String
) ENGINE = Log  -- or MergeTree() ORDER BY (id_contr, seq)
```

### HISTORICO_TEXTO
```sql
CREATE TABLE historico_texto (
    id_contr Int32,
    seq UInt16,
    mensagem String,
    motivo String,
    autorizacao String
) ENGINE = Log  -- or MergeTree() ORDER BY (id_contr, seq)
```

## Troubleshooting

### "Failed to load libchdb.so"
```bash
cd /home/cslog/chdb && make build
export LD_LIBRARY_PATH=/home/cslog/chdb:$LD_LIBRARY_PATH
```

### MySQL Connection Issues
- Check credentials and network connectivity
- Ensure MySQL server allows remote connections
- Verify database and table names

### Performance Issues
- Use `convert_to_mergetree` for better query performance
- Adjust batch sizes in source code if needed
- Monitor disk I/O during import

## Clean Up

```bash
# Remove compiled binaries
make clean

# Remove imported data
make clean-data

# Remove everything
make clean-all
```

## MySQL Integration

For direct MySQL integration using UDF functions, see:
- [MySQL-chDB Plugin Documentation](../mysql-chdb-plugin/README.md)
- [Complete Integration Guide](../mysql-chdb-plugin/docs/COMPLETE_INTEGRATION_GUIDE.md)

## License

This project is for educational and development purposes.