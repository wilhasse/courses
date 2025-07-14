# MySQL to ClickHouse Data Transfer with chDB

This project demonstrates how to extract data from MySQL and load it into ClickHouse using the real chDB library, with data persistence between program executions.

## Architecture

The project provides multiple approaches:

### V2 API (Deprecated but Stable) - RECOMMENDED
- **feed_data_v2**: Uses `query_stable_v2` API to load data from MySQL to ClickHouse
- **query_data_v2**: Uses `query_stable_v2` API to query persisted data

### Modern API (Currently has connection issues)
- **feed_data**: Uses modern `chdb_connect` API (connection issues being investigated)
- **query_data**: Uses modern `chdb_query` API

### API Server Approach (NEW - High Performance)
- **chdb_api_server**: Persistent server that loads libchdb.so once
- **chdb_api_client**: Client that communicates via Protocol Buffers
- Solves the 722MB library loading performance issue
- See [API Server Documentation](docs/api-server-approach.md)

## Files

- `common.h` - Shared structures and constants
- `feed_data_v2.cpp` - MySQL data extraction using v2 API (stable)
- `query_data_v2.cpp` - ClickHouse queries using v2 API (stable)
- `feed_data.cpp` - MySQL data extraction using modern API
- `query_data.cpp` - ClickHouse queries using modern API
- `chdb_api_server.cpp` - Protocol Buffer API server
- `chdb_api_client.cpp` - Protocol Buffer API client
- `chdb_api.proto` - Protocol Buffer schema
- `setup_mysql.sql` - SQL script to create sample MySQL database
- `test_performance.sh` - Performance comparison script
- `Makefile` - Build configuration

## Prerequisites

1. **MySQL server** (user: root, password: teste)
2. **MySQL client libraries**
3. **C++ compiler with C++17 support**
4. **chDB library** - Must be built first:
   ```bash
   cd /home/cslog/chdb
   make build
   ```
   This will create `libchdb.so` in the chdb directory.
5. **Protocol Buffers** (for API server):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install protobuf-compiler libprotobuf-dev
   
   # macOS
   brew install protobuf
   ```

## Building

```bash
# Build all versions
make

# Or build individually
make feed_data_v2 query_data_v2  # V2 API (recommended)
make feed_data query_data        # Modern API
```

## Running

### Using V2 API (Recommended)

#### Step 1: Set up MySQL database
```bash
mysql -u root -pteste < setup_mysql.sql
```

#### Step 2: Feed data from MySQL to ClickHouse
```bash
make run-feed-v2
# Or: ./feed_data_v2
```

#### Step 3: Query the persisted data
```bash
make run-query-v2
# Or: ./query_data_v2
```

#### Run both steps
```bash
make run-all-v2
```

### Using Modern API (Experimental)

```bash
make run-all  # This currently has connection issues
```

### Using API Server (High Performance)

#### Step 1: Build the API server and client
```bash
make chdb_api_server chdb_api_client
```

#### Step 2: Start the API server
```bash
# In one terminal
./chdb_api_server
# Server loads libchdb.so once and listens on port 8125
```

#### Step 3: Use the client to query
```bash
# In another terminal
./chdb_api_client "SELECT COUNT(*) FROM mysql_import.customers"
./chdb_api_client "SELECT * FROM mysql_import.orders LIMIT 5" TSV
```

#### Performance Comparison
```bash
# Run automated performance test
./test_performance.sh
```

## Data Persistence

Data is persisted to the `./clickhouse_data` directory using chDB's persistence features. This allows:
- Running feed_data once to load data
- Running query_data multiple times without reloading
- Data survives program restarts

## Cleaning

```bash
# Clean build artifacts
make clean

# Clean persisted data
make clean-data

# Clean everything
make clean-all
```

## Sample Queries

Both query programs demonstrate:
1. Customer count by city
2. Top customers by revenue
3. Monthly order statistics
4. Customer demographics
5. Pretty formatted output

## Technical Details

### V2 API (query_stable_v2)
- Uses command-line style arguments
- More mature and stable
- Returns `local_result_v2` structure with error handling
- Suitable for production use

### Modern API (chdb_connect/chdb_query)
- Cleaner API design
- Currently investigating connection initialization issues
- Will be the preferred API once issues are resolved

## Troubleshooting

If you get "Failed to load libchdb.so":
1. Ensure chDB is built: `cd /home/cslog/chdb && make build`
2. Check that libchdb.so exists in the chdb directory
3. The programs will search for libchdb.so in multiple locations

## API Comparison

| Feature | V2 API | Modern API |
|---------|---------|------------|
| Function | `query_stable_v2()` | `chdb_connect()`, `chdb_query()` |
| Stability | Stable | Under development |
| Connection | Per-query | Persistent |
| Error Handling | `error_message` field | `chdb_result_error()` |
| Recommended | Yes (for now) | Future

## MySQL Integration

This project now includes full MySQL integration! You can query ClickHouse data directly from MySQL using UDF functions.

### Quick MySQL Setup

1. **Start the API server** (this directory):
   ```bash
   ./chdb_api_server_simple
   ```

2. **Install MySQL UDF** (in mysql-chdb-plugin directory):
   ```bash
   cd ../mysql-chdb-plugin
   ./scripts/build_api_udf.sh
   sudo cp build/chdb_api_functions.so /usr/lib/mysql/plugin/
   mysql -u root -pteste < scripts/install_api_udf.sql
   ```

3. **Query from MySQL**:
   ```sql
   -- Count customers
   SELECT chdb_count('mysql_import.customers');
   
   -- Analytics query
   SELECT CAST(chdb_query('
       SELECT city, COUNT(*) as cnt 
       FROM mysql_import.customers 
       GROUP BY city
   ') AS CHAR);
   ```

For complete MySQL integration documentation, see:
- [mysql-chdb-plugin/docs/COMPLETE_INTEGRATION_GUIDE.md](../mysql-chdb-plugin/docs/COMPLETE_INTEGRATION_GUIDE.md)
- [mysql-chdb-plugin/docs/API_UDF_GUIDE.md](../mysql-chdb-plugin/docs/API_UDF_GUIDE.md)