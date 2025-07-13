# Implementation Guide: MySQL to ClickHouse Data Transfer with chDB

This guide documents the complete process of implementing a C++ application that transfers data from MySQL to ClickHouse using the chDB library.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Implementation Process](#implementation-process)
4. [API Evolution](#api-evolution)
5. [Key Learnings](#key-learnings)
6. [Code Structure](#code-structure)
7. [Building chDB](#building-chdb)
8. [Troubleshooting](#troubleshooting)

## Overview

This project demonstrates how to:
- Extract data from a MySQL database
- Load it into ClickHouse using chDB (embedded ClickHouse)
- Persist the data between program executions
- Query the data using ClickHouse's analytical capabilities

### Why chDB?

chDB is an embedded SQL OLAP engine powered by ClickHouse that runs in-process. Unlike traditional ClickHouse, it doesn't require a separate server, making it ideal for:
- Embedded analytics
- Local data processing
- Applications that need OLAP capabilities without server infrastructure

## Architecture Design

### Initial Design Goals

1. **Separation of Concerns**: Split data loading and querying into separate executables
2. **Data Persistence**: Enable data to survive between program executions
3. **Real chDB Integration**: Use the actual chDB C/C++ API, not simulations

### Final Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│     MySQL       │     │   feed_data_v2   │     │  ClickHouse     │
│   Database      │────▶│   (Extractor)    │────▶│   (via chDB)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  query_data_v2   │────▶│ Persisted Data  │
                        │   (Analyzer)     │     │ ./clickhouse_data│
                        └──────────────────┘     └─────────────────┘
```

## Implementation Process

### Step 1: Project Setup

Created the basic project structure:
```bash
mysql-to-chdb-example/
├── common.h              # Shared structures and constants
├── setup_mysql.sql       # MySQL sample data
├── feed_data.cpp        # Data extraction (modern API)
├── query_data.cpp       # Data querying (modern API)
├── feed_data_v2.cpp     # Data extraction (v2 API)
├── query_data_v2.cpp    # Data querying (v2 API)
└── Makefile             # Build configuration
```

### Step 2: Understanding chDB's C API

Initially attempted to use the modern API from `chdb.h`:

```cpp
// Modern API structure
typedef struct chdb_connection_* chdb_connection;
typedef struct chdb_result_* chdb_result;

// Connection functions
chdb_connection* chdb_connect(int argc, char** argv);
void chdb_close_conn(chdb_connection* conn);

// Query functions
chdb_result* chdb_query(chdb_connection conn, const char* query, const char* format);
void chdb_destroy_query_result(chdb_result* result);
```

### Step 3: Discovering API Compatibility Issues

The modern API returned NULL connections, indicating initialization issues. Investigation revealed:
- The modern C API is primarily designed for Python bindings
- Additional initialization may be required
- The deprecated v2 API is more stable for direct C usage

### Step 4: Implementing with V2 API

Switched to the deprecated but stable v2 API:

```cpp
// V2 API structure
struct local_result_v2 {
    char* buf;
    size_t len;
    void* _vec;
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    char* error_message;
};

// V2 API functions
struct local_result_v2* query_stable_v2(int argc, char** argv);
void free_result_v2(struct local_result_v2* result);
```

### Step 5: Dynamic Library Loading

Used `dlopen` for runtime loading of `libchdb.so`:

```cpp
void* chdb_handle = dlopen("/home/cslog/chdb/libchdb.so", RTLD_LAZY);
query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
```

This approach provides:
- No compile-time dependency on chDB
- Flexible library location
- Clear error messages if library is missing

## API Evolution

### Why Two API Versions?

1. **V2 API (Deprecated but Stable)**
   - Works reliably for C/C++ applications
   - Command-line style interface
   - Proven stability

2. **Modern API (Future Direction)**
   - Cleaner design
   - Connection-oriented
   - Currently has initialization issues in C/C++

### API Usage Comparison

#### V2 API Usage:
```cpp
// Build command-line arguments
std::vector<char*> argv;
argv.push_back("clickhouse");
argv.push_back("--multiquery");
argv.push_back("--output-format=CSV");
argv.push_back("--path=./clickhouse_data");
argv.push_back("--query=SELECT 1");

// Execute query
auto result = query_stable_v2(argv.size(), argv.data());
if (result && !result->error_message) {
    std::cout << result->buf << std::endl;
}
free_result_v2(result);
```

#### Modern API Usage (when working):
```cpp
// Create connection
char* argv[] = {"clickhouse", "--path=./clickhouse_data"};
auto conn = chdb_connect(2, argv);

// Execute query
auto result = chdb_query(conn, "SELECT 1", "CSV");
if (!chdb_result_error(result)) {
    std::cout << chdb_result_buffer(result) << std::endl;
}
chdb_destroy_query_result(result);
chdb_close_conn(conn);
```

## Key Learnings

### 1. Data Persistence

chDB supports data persistence through the `--path` parameter:
```cpp
"--path=./clickhouse_data"  // Data persists in this directory
```

This enables:
- Stateful operations across program runs
- Separation of data loading and querying
- Database-like persistence without a server

### 2. Query Execution Pattern

Every query with v2 API requires full argument setup:
```cpp
executeQuery(const std::string& query, const std::string& format = "CSV") {
    std::vector<char*> argv;
    argv.push_back("clickhouse");
    argv.push_back("--multiquery");
    argv.push_back("--output-format=" + format);
    argv.push_back("--path=" + CHDB_PATH);
    argv.push_back("--query=" + query);
    return query_stable_v2(argv.size(), argv.data());
}
```

### 3. Error Handling

Always check for errors:
```cpp
if (result->error_message) {
    std::cerr << "Error: " << result->error_message << std::endl;
} else {
    // Process result
}
```

### 4. Memory Management

- Results must be freed with `free_result_v2()`
- Dynamic loading requires `dlclose()` at cleanup
- Use RAII patterns for automatic cleanup

## Code Structure

### Common Header (common.h)
```cpp
// Shared data structures
struct Customer {
    int id;
    std::string name;
    std::string email;
    int age;
    std::string city;
    std::string created_at;
};

// Configuration constants
const std::string CHDB_PATH = "./clickhouse_data";
const std::string MYSQL_HOST = "localhost";
const std::string MYSQL_USER = "root";
const std::string MYSQL_PASSWORD = "teste";
```

### Data Flow

1. **MySQL → Memory**: Fetch data using MySQL C API
2. **Memory → ClickHouse**: Insert using chDB queries
3. **Persistence**: Data saved in `./clickhouse_data`
4. **Querying**: Analytical queries on persisted data

## Building chDB

### Prerequisites
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    cmake ninja-build python3-pip \
    llvm-15 clang-15 lld-15
```

### Build Process
```bash
cd /home/cslog/chdb
make clean
make build

# This creates libchdb.so in the chdb directory
ls -la libchdb.so
```

### Build Time
- Full build can take 30-60 minutes
- Requires significant RAM (8GB+ recommended)
- Creates a large library (~500MB)

## Troubleshooting

### Common Issues and Solutions

#### 1. "Failed to load libchdb.so"
**Cause**: Library not built or not in expected location
**Solution**: 
```bash
cd /home/cslog/chdb && make build
# Verify library exists
ls -la /home/cslog/chdb/libchdb.so
```

#### 2. "Invalid or closed connection" (Modern API)
**Cause**: Connection initialization issues
**Solution**: Use v2 API until modern API is fixed

#### 3. MySQL Connection Failures
**Cause**: Wrong credentials or MySQL not running
**Solution**: 
```bash
# Test MySQL connection
mysql -u root -pteste -e "SELECT 1"
```

#### 4. Segmentation Faults
**Cause**: Usually NULL pointer access
**Solution**: Always check return values:
```cpp
if (result && result->buf) {
    // Safe to use result
}
```

### Performance Considerations

1. **Batch Inserts**: Current implementation inserts row-by-row
   - Consider using batch inserts for large datasets
   - Use ClickHouse's native formats for better performance

2. **Memory Usage**: chDB loads ClickHouse engine in-process
   - Monitor memory usage for large datasets
   - Consider streaming for very large transfers

3. **Query Optimization**: ClickHouse excels at analytical queries
   - Use appropriate indexes (ORDER BY clause)
   - Leverage ClickHouse's columnar storage

## Future Improvements

1. **Batch Operations**: Implement bulk inserts for better performance
2. **Error Recovery**: Add retry logic and better error handling
3. **Configuration**: Move hardcoded values to config file
4. **Modern API**: Investigate and fix connection issues
5. **Streaming**: Implement streaming for large datasets

## Conclusion

This implementation successfully demonstrates:
- ✅ MySQL to ClickHouse data transfer
- ✅ Data persistence between executions
- ✅ Analytical query capabilities
- ✅ Real chDB library integration

The v2 API, while deprecated, provides a stable foundation for C/C++ applications using chDB. The modern API shows promise but requires further investigation for proper C/C++ usage.

## References

- [chDB GitHub Repository](https://github.com/chdb-io/chdb)
- [ClickHouse Documentation](https://clickhouse.com/docs)
- [MySQL C API Documentation](https://dev.mysql.com/doc/c-api/8.0/en/)