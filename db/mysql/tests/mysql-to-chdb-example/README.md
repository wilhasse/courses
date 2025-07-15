# MySQL to ClickHouse Data Transfer with chDB

This project demonstrates how to extract data from MySQL and load it into ClickHouse using the chDB library, with multiple approaches for different use cases. The project now includes a high-performance Go implementation that solves common freezing issues encountered with large datasets.

## Overview

This project provides various tools to:
- Import MySQL data into ClickHouse (chDB) with high performance
- Convert between different ClickHouse storage engines for optimal query performance
- Serve ClickHouse queries via API servers
- Test and benchmark performance

## Installing libchdb (Required for All Tools)

The official chdb library must be installed before using any of the tools:

```bash
# Install using the official installer (recommended)
curl -sL https://lib.chdb.io | bash

# Update dynamic linker cache
sudo ldconfig

# Verify installation
ls -la /usr/local/lib/libchdb.so
```

This installs:
- `/usr/local/lib/libchdb.so` - The chdb shared library
- `/usr/local/include/chdb.h` - Header files

**Note**: The official release version is more stable than locally compiled versions.

## Main Components

### Data Import Tools

#### 1. **historico_loader_go** (Go Version - RECOMMENDED)
High-performance Go implementation that solves freezing issues with large datasets. Uses the MergeTree engine with optimized merge control.

```bash
./historico_loader_go [options]
```

Features:
- Native Go implementation with stable memory management
- Uses chdb-go library for better reliability
- STOP/START MERGES strategy prevents freezing
- Configurable data storage path
- 50,000 rows per chunk, 5,000 rows per batch
- Real-time progress tracking with ETA
- Resume capability with offset option
- Periodic data verification

Options:
- `-host`: MySQL host address (default: localhost)
- `-user`: MySQL username (default: root)
- `-password`: MySQL password
- `-database`: MySQL database name (required)
- `-row-count`: Total row count to skip COUNT(*) query
- `-offset`: Start from this row offset
- `-skip-texto`: Skip HISTORICO_TEXTO table
- `-chdb-path`: Path for chdb data storage (default: /tmp/chdb)

Examples:
```bash
# Basic usage
./historico_loader_go -host localhost -user root -password pass -database mydb

# Production usage with all optimizations
./historico_loader_go \
    -host 172.16.120.10 \
    -user appl_cslog \
    -password 'D981x@a' \
    -database cslog_siscom_prod \
    -row-count 300266692 \
    -skip-texto \
    -chdb-path /data/chdb

# Resume after interruption
./historico_loader_go ... -offset 10000000 -row-count 300266692

# Use SSD for better performance
./historico_loader_go ... -chdb-path /mnt/ssd/chdb
```

Performance:
- Initial speed: 100,000+ rows/second
- Sustained speed: 30,000-50,000 rows/second
- No freezing issues with proper merge control
- Memory usage: ~1-2GB constant

Building the Go version:
```bash
# Install dependencies
go mod download

# Build
go build -o historico_loader_go historico_loader.go
```

#### 2. **historico_log** (C++ Version - Log Engine)
The original C++ tool for importing MySQL tables into chDB using the Log engine. Works well but slower than MergeTree.

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

#### 3. **historico_feeder** (C++ Version - MergeTree Engine)
C++ implementation using MergeTree engine with STOP/START MERGES strategy:
- Configurable chunk size
- Test mode for debugging
- Automatic chDB library restart after many operations
- More detailed error handling

```bash
./historico_feeder <host> <user> <password> <database> [--test]
```

```bash
# Building C++ version with MergeTree support
g++ -o historico_feeder historico_feeder.cpp \
    -I/usr/include/mysql \
    -L/usr/lib/x86_64-linux-gnu \
    -lperconaserverclient -ldl -std=c++11
```

**Note**: The Go version (`historico_loader_go`) is recommended over C++ versions due to better stability and memory management.

#### 4. **feed_data_v2** / **feed_data**
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

1. **libchdb** (REQUIRED - see installation section above)
2. **MySQL server** with access credentials
3. **MySQL client libraries**:
   ```bash
   # For standard MySQL
   sudo apt-get install libmysqlclient-dev  # Ubuntu/Debian
   
   # For Percona Server (recommended)
   sudo apt-get install libperconaserverclient-dev
   ```
4. **For Go version**:
   - Go 1.21 or higher
   - Run `go mod download` to get dependencies
5. **For C++ versions**:
   - C++ compiler with C++17 support
   - Protocol Buffers (for API server only):
     ```bash
     sudo apt-get install protobuf-compiler libprotobuf-dev
     ```

## Building

### Go Version (Recommended)
```bash
# Install Go dependencies
go mod download

# Build the Go loader
go build -o historico_loader_go historico_loader.go
```

### C++ Versions
```bash
# Build all C++ tools
make all

# Build specific C++ tools
make historico_log
make historico_feeder
make convert_to_mergetree
make test_performance

# For Percona Server users
g++ -o historico_feeder historico_feeder.cpp \
    -I/usr/include/mysql \
    -L/usr/lib/x86_64-linux-gnu \
    -lperconaserverclient -ldl -std=c++11
```

## Typical Workflow

### 1. Import MySQL Data

#### Using Go Version (Recommended for Large Datasets)
```bash
# First time import - will perform COUNT(*) query
./historico_loader_go \
    -host 172.16.120.10 \
    -user user \
    -password password \
    -database database \
    -skip-texto \
    -chdb-path /data/chdb

# Subsequent imports - skip COUNT(*) for faster startup
./historico_loader_go \
    -host 172.16.120.10 \
    -user user \
    -password password \
    -database database \
    -skip-texto \
    -row-count 300266692 \
    -chdb-path /data/chdb

# Resume after interruption at row 10,000,000
./historico_loader_go ... -offset 10000000 -row-count 300266692
```

Performance with Go version:
- No freezing issues
- 30,000-50,000 rows/second sustained
- Direct MergeTree engine (no conversion needed)
- Memory usage: 1-2GB constant

#### Using C++ Log Version (Alternative)
```bash
# Import only HISTORICO
./historico_log 172.16.120.10 user password database --skip-texto

# Skip the slow COUNT(*) query by providing row count
./historico_log 172.16.120.10 user password database --skip-texto --row-count 32424049

# Resume after interruption
./historico_log 172.16.120.10 user password database --skip-texto --row-count 32424049 --offset 10000000
```

Memory Usage: Stays constant at ~200-300MB regardless of table size!

### 2. Convert to MergeTree for Better Performance (Only needed for C++ Log version)
```bash
# Not needed if you used historico_loader_go (already uses MergeTree)
./convert_to_mergetree
```

This creates MergeTree versions of your tables with "_mt" suffix for much faster queries.
**Note**: The Go version already uses MergeTree engine, so conversion is not needed.

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

### "libchdb.so: cannot open shared object file"
```bash
# Install the official library
curl -sL https://lib.chdb.io | bash
sudo ldconfig

# If still not found
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### MySQL Connection Issues
- Check credentials and network connectivity
- Ensure MySQL server allows remote connections
- Verify database and table names

### Performance Issues
- Use Go version (`historico_loader_go`) for best performance
- Place chdb data on SSD: `-chdb-path /mnt/ssd/chdb`
- Use `convert_to_mergetree` only if using C++ Log version
- Monitor disk I/O during import with `iostat -x 1`

### Freezing Issues with Large Datasets
- Use the Go version which implements proper STOP/START MERGES
- Avoid the C++ MergeTree version with very large datasets
- The freezing is caused by background merge operations blocking INSERTs

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