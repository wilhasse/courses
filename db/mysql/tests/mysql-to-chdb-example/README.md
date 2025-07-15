# MySQL to ClickHouse Data Transfer with chDB

This project demonstrates how to extract data from MySQL and load it into ClickHouse using the chDB library, with multiple approaches for different use cases. The project now includes a high-performance Go implementation that solves common freezing issues encountered with large datasets.

> **⚠️ CRITICAL PERFORMANCE NOTE**: We discovered that inserting data in PRIMARY KEY order can improve performance by **5x** and reduce storage by **100x**. See [CLICKHOUSE_OPTIMIZATION_INSIGHTS.md](CLICKHOUSE_OPTIMIZATION_INSIGHTS.md) for details that can save you hours of import time and hundreds of GB of storage.

## Overview

This project provides various tools to:
- Import MySQL data into ClickHouse (chDB) with high performance
- Convert between different ClickHouse storage engines for optimal query performance
- Serve ClickHouse queries via API servers
- Test and benchmark performance

## Quick Start Guide

```bash
# 1. Build the Go loader
go build -o historico_loader_go historico_loader.go

# 2. Import data (example with 10k rows for testing)
./historico_loader_go \
    -host your_mysql_host \
    -user your_user \
    -password 'your_password' \
    -database your_database \
    -row-count 10000 \
    -chdb-path /chdb/data

# 3. CRITICAL: Verify data exists!
./execute_sql -d /chdb/data "SELECT COUNT(*) FROM mysql_import.historico"

# 4. If count is 0, your data was deleted! Use the fixed loader and re-import.

# 5. Query your data
./execute_sql -d /chdb/data "SELECT * FROM mysql_import.historico LIMIT 10"
```

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

### API Server/Client Pairs

1. **Simple Binary Protocol** (Recommended)
   - `chdb_api_server_simple` + `chdb_api_client_simple`
   - Lightweight, no dependencies
   - Perfect for scripts and simple integrations

2. **Protocol Buffer API** (Advanced)
   - `chdb_api_server` + `chdb_api_client`
   - Structured communication with protobuf
   - Multiple output formats and better error handling

**⚠️ WARNING**: Clients and servers from different protocols are NOT compatible!

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

**⚠️ CRITICAL WARNING**: The chdb-go library's `Cleanup()` method DELETES ALL DATA! We've patched our loader to prevent this, but be aware that the session remains open. The data is safely persisted to disk.

**🚀 Performance Breakthrough**: The Go version uses keyset pagination which naturally maintains ORDER BY consistency. When data is inserted in the same order as the table's PRIMARY KEY, we observed:
- **5x faster imports** (99k vs 19k rows/sec)
- **100x smaller storage** (100MB vs 10GB for 16M rows)
- **No partitioning errors**
- **See [CLICKHOUSE_OPTIMIZATION_INSIGHTS.md](CLICKHOUSE_OPTIMIZATION_INSIGHTS.md) for details**

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
- `-d, --data <path>`: ClickHouse data path (default: ./clickhouse_data)
- `-h, --help`: Show help message

Examples:
```bash
# Simple count
./execute_sql "SELECT COUNT(*) FROM mysql_import.historico"

# Query from custom data path
./execute_sql -d /chdb/data "SELECT COUNT(*) FROM mysql_import.historico"

# Export to CSV
./execute_sql -f CSV "SELECT * FROM mysql_import.historico LIMIT 1000" > export.csv

# Complex analytics with pretty output
./execute_sql -d /chdb/data "SELECT codigo, COUNT(*) as cnt FROM mysql_import.historico GROUP BY codigo ORDER BY cnt DESC LIMIT 10"

# Pipe to other tools
./execute_sql -f TSV "SELECT id_contr, seq FROM mysql_import.historico WHERE codigo = 51" | awk '{print $1}'
```

#### 4. **query_data_v2** / **query_data**
Query tools for the sample customer/order data.

### API Server Components

**⚠️ IMPORTANT**: There are two different API implementations with incompatible protocols:
1. **Simple Binary Protocol** (lightweight, no dependencies)
2. **Protocol Buffers** (structured, requires protobuf library)

#### Simple Binary Protocol (Recommended for Most Use Cases)

##### **chdb_api_server_simple** - Simple Binary Server
Lightweight server using a simple binary protocol. No Protocol Buffers required.

```bash
./chdb_api_server_simple [options]
```

Options:
- `-p, --port <port>`: Port to listen on (default: 8125)
- `-d, --data <path>`: ClickHouse data path (default: ./clickhouse_data)
- `-h, --help`: Show help message

Examples:
```bash
# Start server with default settings
./chdb_api_server_simple

# Custom port and data path
./chdb_api_server_simple -p 8126 -d /chdb/data

# Use SSD for better performance
./chdb_api_server_simple --data /mnt/ssd/chdb
```

##### **chdb_api_client_simple** - Simple Binary Client
Matching client for the simple server. Uses the same binary protocol.

```bash
./chdb_api_client_simple [options] "SQL query"
```

Options:
- `-h, --host <host>`: Server host (default: 127.0.0.1)
- `-p, --port <port>`: Server port (default: 8125)
- `--help`: Show help message

Examples:
```bash
# Query local server
./chdb_api_client_simple "SELECT COUNT(*) FROM mysql_import.historico"

# Query remote server
./chdb_api_client_simple -h 192.168.1.10 "SELECT * FROM mysql_import.historico LIMIT 5"

# Custom port
./chdb_api_client_simple -p 8126 "SELECT version()"
```

**Simple Protocol Format:**
- Request: 4-byte size (network order) + query string
- Response: 4-byte size (network order) + result string (TSV format)

#### Protocol Buffer API (Advanced Use Cases)

##### **chdb_api_server** - Protocol Buffer Server
Advanced server using Google Protocol Buffers for structured communication.

```bash
./chdb_api_server [options]
```

Options:
- `-p, --port <port>`: Port to listen on (default: 8125)
- `-d, --data <path>`: ClickHouse data path (default: ./clickhouse_data)
- `-h, --help`: Show help message

Examples:
```bash
# Start protobuf server
./chdb_api_server

# Custom data path
./chdb_api_server -d /data/mysql_import

# Different port to avoid conflicts
./chdb_api_server -p 8200 -d /mnt/nvme/chdb
```

##### **chdb_api_client** - Protocol Buffer Client
Matching client for the protobuf server. Supports multiple output formats.

```bash
./chdb_api_client [options] "SQL query"
```

Features:
- Structured request/response format
- Multiple output formats (CSV, TSV, JSON, Pretty, etc.)
- Query statistics (rows read, bytes processed, elapsed time)
- Better error handling

Examples:
```bash
# Query protobuf server (must be running chdb_api_server, NOT simple)
./chdb_api_client "SELECT COUNT(*) FROM mysql_import.historico"

# Different output formats
./chdb_api_client -f JSON "SELECT * FROM mysql_import.historico LIMIT 5"
```

#### Choosing Between Simple and Protocol Buffer APIs

| Feature | Simple Binary | Protocol Buffers |
|---------|--------------|------------------|
| **Dependencies** | None | Requires protobuf library |
| **Protocol** | Simple 4-byte length prefix | Structured protobuf messages |
| **Output Formats** | TSV only | CSV, TSV, JSON, Pretty, etc. |
| **Performance** | Slightly faster | Slightly slower (serialization) |
| **Error Handling** | Basic | Advanced with error codes |
| **Use When** | Quick queries, scripts | Production apps, complex needs |

**⚠️ Protocol Compatibility:**
- `chdb_api_server_simple` ↔ `chdb_api_client_simple` ✅
- `chdb_api_server` ↔ `chdb_api_client` ✅
- `chdb_api_server_simple` ↔ `chdb_api_client` ❌ (Protocol mismatch!)
- `chdb_api_server` ↔ `chdb_api_client_simple` ❌ (Protocol mismatch!)

#### Quick Test

```bash
# Terminal 1: Start simple server
./chdb_api_server_simple -d /chdb/data

# Terminal 2: Query with simple client
./chdb_api_client_simple "SELECT COUNT(*) FROM mysql_import.historico"

# Or use curl for simple HTTP-like testing
echo -n -e "\x00\x00\x00\x2FSELECT COUNT(*) FROM mysql_import.historico" | nc localhost 8125
```

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
make execute_sql

# Build API servers and clients
make chdb_api_server_simple chdb_api_client_simple  # Simple binary protocol
make chdb_api_server chdb_api_client                # Protocol Buffers

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

# CRITICAL: Verify data immediately after import completes!
./execute_sql -d /data/chdb "SELECT COUNT(*) FROM mysql_import.historico"
```

Performance with Go version:
- No freezing issues
- 30,000-50,000 rows/second sustained
- Direct MergeTree engine (no conversion needed)
- Memory usage: 1-2GB constant

**⚠️ ALWAYS verify your data exists after import - see step 3 below!**

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

### 3. CRITICAL: Verify Your Data After Import

**Before proceeding, ALWAYS verify your data survived the import process:**

```bash
# Check if data directory exists and has content
ls -la /chdb/data/
du -sh /chdb/data/

# Verify row count matches what was imported
./execute_sql -d /chdb/data "SELECT COUNT(*) as total_rows FROM mysql_import.historico"

# Check table structure
./execute_sql -d /chdb/data "DESCRIBE TABLE mysql_import.historico"

# Sample some data to ensure it's readable
./execute_sql -d /chdb/data "SELECT * FROM mysql_import.historico LIMIT 10"

# Check system tables for storage info
./execute_sql -d /chdb/data "
SELECT 
    database,
    table,
    formatReadableSize(sum(bytes_on_disk)) as disk_size,
    sum(rows) as total_rows,
    count() as parts_count
FROM system.parts 
WHERE database = 'mysql_import' AND active
GROUP BY database, table"
```

**If data is missing**: The import process may have called `Cleanup()` which deletes all data. You'll need to re-import using the fixed version of the loader.

### 4. Test Performance (Optional)
```bash
./test_performance
```

Compare query performance between Log and MergeTree engines.

### 5. Query Your Data

#### Quick Status Check

**⚠️ IMPORTANT**: Always verify your data after import! The chdb-go library's `Cleanup()` method can delete all data if not handled properly.

```bash
# CRITICAL: First verify your data exists
./execute_sql -d /chdb/data "SELECT COUNT(*) FROM mysql_import.historico"

# If using default path
./execute_sql "SELECT COUNT(*) FROM mysql_import.historico"

# Check both tables if you imported both
./execute_sql -d /chdb/data "SELECT 'historico' as table, COUNT(*) as rows FROM mysql_import.historico 
UNION ALL 
SELECT 'historico_texto', COUNT(*) FROM mysql_import.historico_texto"

# Verify data integrity - check first and last records
./execute_sql -d /chdb/data "SELECT 'First Row' as type, * FROM mysql_import.historico ORDER BY id_contr, seq LIMIT 1
UNION ALL
SELECT 'Last Row', * FROM mysql_import.historico ORDER BY id_contr DESC, seq DESC LIMIT 1"

# Check data range
./execute_sql -d /chdb/data "SELECT MIN(data) as earliest, MAX(data) as latest FROM mysql_import.historico"

# Verify storage location and size
du -sh /chdb/data/
```

#### Run Analytics
```bash
# Top 10 most frequent codes
./execute_sql "SELECT codigo, COUNT(*) as frequency FROM mysql_import.historico GROUP BY codigo ORDER BY frequency DESC LIMIT 10"

# Export specific data
./execute_sql -f CSV "SELECT * FROM mysql_import.historico WHERE codigo = 51 AND data >= '2024-01-01'" > codigo_51_2024.csv
```

### 6. Start API Server (Optional)
```bash
# Start with default data path
./chdb_api_server_simple

# Or start with custom data path (must match where you imported data)
./chdb_api_server_simple -d /data/chdb

# Run on different port if 8125 is in use
./chdb_api_server_simple -p 8126 -d /data/chdb
```

Now you can query via HTTP:
```bash
curl -X POST http://localhost:8125/query \
  -d '{"query": "SELECT COUNT(*) FROM mysql_import.historico_mt"}'

# Query specific columns
curl -X POST http://localhost:8125/query \
  -d '{"query": "SELECT id_contr, seq, data FROM mysql_import.historico LIMIT 10"}'
```

## Data Storage

- **Default Location**: `./clickhouse_data/`
- **Configurable**: Use `-chdb-path` for loaders or `-d/--data` for API servers
- **Engines**:
  - **Log**: Simple, fast writes, slower queries
  - **MergeTree**: Indexed, slower writes, much faster queries

**Important**: When using custom data paths, ensure all tools point to the same directory:
```bash
# Import data to custom location
./historico_loader_go ... -chdb-path /data/chdb

# Query from the same location
./chdb_api_server_simple -d /data/chdb

# Use execute_sql with same data path
./execute_sql -d /data/chdb "SELECT COUNT(*) FROM mysql_import.historico"
```

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
) ENGINE = MergeTree() 
ORDER BY (id_contr, seq)  -- Critical: INSERT data in this order for 100x better performance!
-- Avoid PARTITION BY unless needed for data lifecycle management
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

### API Connection Issues
If you get "Failed to execute query" when using API clients:

1. **Check Protocol Compatibility**:
   ```bash
   # Wrong - mixing protocols
   ./chdb_api_server_simple    # Simple protocol
   ./chdb_api_client           # Protobuf client - WON'T WORK!
   
   # Correct - matching protocols
   ./chdb_api_server_simple    # Simple protocol
   ./chdb_api_client_simple    # Simple client - WORKS!
   ```

2. **Verify Server is Running**:
   ```bash
   # Check if server is listening
   netstat -an | grep 8125
   telnet localhost 8125
   ```

3. **Test with Simple Client First**:
   ```bash
   # Simple client has better error messages
   ./chdb_api_client_simple "SELECT 1"
   ```

4. **Check Data Path**:
   ```bash
   # Make sure client queries match server's data path
   ./chdb_api_server_simple -d /chdb/data    # Server uses /chdb/data
   ./chdb_api_client_simple "SELECT COUNT(*) FROM mysql_import.historico"
   ```

### Partitioning Issues
If you get "Too many partitions for single INSERT block" error:
- Your data spans many months/years
- Remove `PARTITION BY toYYYYMM(data)` from table creation
- Or use coarser partitioning like `PARTITION BY toYear(data)`

### Resuming Interrupted Loads
The Go version supports smart resuming:

1. **Automatic Resume** (recommended):
   ```bash
   # Just run without -offset, it will auto-detect where to continue
   ./historico_loader_go -host ... -database ... -row-count 300266692
   ```

2. **Manual Resume with Offset**:
   ```bash
   # If you know approximately where you stopped
   ./historico_loader_go ... -offset 17250000 -row-count 300266692
   ```

**How Resume Works:**
- First tries to find last row from ClickHouse (instant)
- Falls back to MySQL OFFSET if needed (slow for first query only)
- Continues with keyset pagination at full speed

**Important:** Always specify `-row-count` when resuming to avoid slow COUNT(*) query

### Storage Management During Import

**Expected Storage Growth:**
- **During import**: 3-6x final size due to unmerged parts
- **Example**: 300M rows → ~200GB during import → ~10-20GB after optimization
- **Reason**: Each INSERT creates a separate part with metadata overhead

**Storage Optimization:**
- Automatic: `OPTIMIZE TABLE ... FINAL` runs at completion
- Manual: Run `OPTIMIZE TABLE mysql_import.historico FINAL` after import
- Timeline: 10-30 minutes for 300M rows optimization

**Monitoring Storage:**
```bash
# Check current storage usage
du -sh /chdb/data/mysql_import/historico

# Check parts and compression in chdb
./execute_sql -d /chdb/data "
SELECT 
    count() as parts,
    formatReadableSize(sum(bytes_on_disk)) as size,
    formatReadableSize(sum(data_compressed_bytes)) as compressed,
    round(sum(data_compressed_bytes) / sum(data_uncompressed_bytes), 2) as ratio
FROM system.parts 
WHERE database='mysql_import' AND table='historico' AND active"
```

### Data Loss Prevention and Recovery

**⚠️ CRITICAL ISSUE**: The chdb-go library's `Cleanup()` method DELETES ALL DATA!

**Prevention:**
1. **Use the fixed loader**: We've patched `historico_loader_go` to use `Close()` instead of `Cleanup()`
2. **Always verify after import**: Run count queries immediately after import completes
3. **Monitor the import output**: Look for "Session closed. Data preserved." message

**If Your Data Disappeared:**
```bash
# Check if any data remains
ls -la /chdb/data/
find /chdb -name "*.bin" -o -name "*.idx" 2>/dev/null | head -20

# Data is likely gone if directory is empty
# You'll need to re-import with the fixed loader
```

**Safe Import Process:**
```bash
# 1. Use the fixed loader
./historico_loader_go -host ... -database ... -chdb-path /chdb/data

# 2. Watch for the final message
# Should see: "Session closed. Data preserved."

# 3. IMMEDIATELY verify data
./execute_sql -d /chdb/data "SELECT COUNT(*) FROM mysql_import.historico"

# 4. Check storage
du -sh /chdb/data/
```

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