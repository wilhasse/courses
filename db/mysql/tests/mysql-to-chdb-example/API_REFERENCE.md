# chDB C/C++ API Reference

This document provides a reference for using chDB's C/C++ APIs in your applications.

## Table of Contents
1. [V2 API (Stable)](#v2-api-stable)
2. [Modern API (Experimental)](#modern-api-experimental)
3. [Dynamic Loading](#dynamic-loading)
4. [Query Formats](#query-formats)
5. [Examples](#examples)

## V2 API (Stable)

### Structures

```cpp
struct local_result_v2 {
    char* buf;              // Query result buffer
    size_t len;             // Length of result
    void* _vec;             // Internal vector (don't access directly)
    double elapsed;         // Query execution time in seconds
    uint64_t rows_read;     // Number of rows read
    uint64_t bytes_read;    // Number of bytes read
    char* error_message;    // Error message (NULL if no error)
};
```

### Functions

#### query_stable_v2
```cpp
struct local_result_v2* query_stable_v2(int argc, char** argv);
```
Executes a query with command-line style arguments.

**Parameters:**
- `argc`: Number of arguments
- `argv`: Array of argument strings

**Returns:** Pointer to result structure (must be freed)

**Example:**
```cpp
char* argv[] = {
    "clickhouse",
    "--multiquery",
    "--output-format=CSV",
    "--path=./data",
    "--query=SELECT 1"
};
auto result = query_stable_v2(5, argv);
```

#### free_result_v2
```cpp
void free_result_v2(struct local_result_v2* result);
```
Frees memory allocated by query_stable_v2.

**Parameters:**
- `result`: Result structure to free

### Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--multiquery` | Enable multiple queries | Required for most uses |
| `--output-format` | Output format | `--output-format=CSV` |
| `--path` | Data directory path | `--path=./clickhouse_data` |
| `--query` | SQL query to execute | `--query=SELECT 1` |
| `--verbose` | Enable verbose output | For debugging |
| `--log-level` | Set log level | `--log-level=trace` |

## Modern API (Experimental)

### Types

```cpp
typedef struct chdb_connection_* chdb_connection;
typedef struct chdb_result_* chdb_result;
```

### Connection Functions

#### chdb_connect
```cpp
chdb_connection* chdb_connect(int argc, char** argv);
```
Creates a new connection to chDB.

**Parameters:**
- `argc`: Number of arguments
- `argv`: Connection arguments (e.g., `--path`)

**Returns:** Connection handle or NULL on failure

#### chdb_close_conn
```cpp
void chdb_close_conn(chdb_connection* conn);
```
Closes a connection and releases resources.

### Query Functions

#### chdb_query
```cpp
chdb_result* chdb_query(chdb_connection conn, const char* query, const char* format);
```
Executes a query on the connection.

**Parameters:**
- `conn`: Active connection
- `query`: SQL query string
- `format`: Output format (e.g., "CSV", "JSON")

**Returns:** Query result handle

#### chdb_destroy_query_result
```cpp
void chdb_destroy_query_result(chdb_result* result);
```
Frees query result resources.

### Result Access Functions

#### chdb_result_buffer
```cpp
char* chdb_result_buffer(chdb_result* result);
```
Gets pointer to result data.

#### chdb_result_length
```cpp
size_t chdb_result_length(chdb_result* result);
```
Gets length of result data.

#### chdb_result_error
```cpp
const char* chdb_result_error(chdb_result* result);
```
Gets error message (NULL if no error).

## Dynamic Loading

### Loading the Library

```cpp
#include <dlfcn.h>

// Load library
void* handle = dlopen("/path/to/libchdb.so", RTLD_LAZY);
if (!handle) {
    std::cerr << "Error: " << dlerror() << std::endl;
    return false;
}

// Load function pointers
typedef struct local_result_v2* (*query_stable_v2_fn)(int, char**);
query_stable_v2_fn query_stable_v2 = 
    (query_stable_v2_fn)dlsym(handle, "query_stable_v2");

// Use the function
// ...

// Clean up
dlclose(handle);
```

### Function Pointer Types

```cpp
// V2 API
typedef struct local_result_v2* (*query_stable_v2_fn)(int argc, char** argv);
typedef void (*free_result_v2_fn)(struct local_result_v2* result);

// Modern API
typedef chdb_connection* (*chdb_connect_fn)(int argc, char** argv);
typedef void (*chdb_close_conn_fn)(chdb_connection* conn);
typedef chdb_result* (*chdb_query_fn)(chdb_connection conn, const char* query, const char* format);
```

## Query Formats

chDB supports multiple output formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| `CSV` | Comma-separated values | Default, parsing |
| `TSV` | Tab-separated values | Parsing |
| `JSON` | JSON format | APIs |
| `JSONCompact` | Compact JSON | APIs |
| `Pretty` | Human-readable table | Display |
| `PrettyCompact` | Compact table | Display |
| `Parquet` | Apache Parquet | Data exchange |

## Examples

### Basic Query Execution (V2 API)

```cpp
class ChDBWrapper {
private:
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    std::string db_path;

public:
    ChDBWrapper(const std::string& path) : db_path(path) {
        // Load functions (see Dynamic Loading section)
    }

    std::string executeQuery(const std::string& query, 
                           const std::string& format = "CSV") {
        std::vector<char*> argv;
        std::vector<std::string> args;
        
        // Build arguments
        args.push_back("clickhouse");
        args.push_back("--multiquery");
        args.push_back("--output-format=" + format);
        args.push_back("--path=" + db_path);
        args.push_back("--query=" + query);
        
        // Convert to char*
        for (auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        // Execute
        auto result = query_stable_v2(argv.size(), argv.data());
        
        std::string output;
        if (result) {
            if (result->error_message) {
                output = "Error: " + std::string(result->error_message);
            } else if (result->buf) {
                output = std::string(result->buf, result->len);
            }
            free_result_v2(result);
        }
        
        return output;
    }
};
```

### Creating Tables

```cpp
// Create database
wrapper.executeQuery("CREATE DATABASE IF NOT EXISTS mydb");

// Create table
wrapper.executeQuery(R"(
    CREATE TABLE IF NOT EXISTS mydb.events (
        timestamp DateTime,
        user_id UInt32,
        event_type String,
        value Float64
    ) ENGINE = MergeTree()
    ORDER BY (timestamp, user_id)
)");
```

### Inserting Data

```cpp
// Single insert
wrapper.executeQuery(
    "INSERT INTO mydb.events VALUES "
    "('2024-01-01 10:00:00', 123, 'click', 1.0)"
);

// Batch insert
std::stringstream query;
query << "INSERT INTO mydb.events VALUES ";
for (int i = 0; i < 1000; i++) {
    if (i > 0) query << ",";
    query << "('2024-01-01 10:00:00', " 
          << i << ", 'click', " << (i * 0.1) << ")";
}
wrapper.executeQuery(query.str());
```

### Analytical Queries

```cpp
// Aggregation
auto result = wrapper.executeQuery(R"(
    SELECT 
        event_type,
        COUNT(*) as count,
        AVG(value) as avg_value
    FROM mydb.events
    GROUP BY event_type
    ORDER BY count DESC
)", "Pretty");

// Time series
auto result = wrapper.executeQuery(R"(
    SELECT 
        toStartOfHour(timestamp) as hour,
        COUNT(*) as events_per_hour
    FROM mydb.events
    WHERE timestamp >= today()
    GROUP BY hour
    ORDER BY hour
)");
```

### Error Handling Pattern

```cpp
template<typename Func>
bool executeWithRetry(Func func, int max_retries = 3) {
    for (int i = 0; i < max_retries; i++) {
        try {
            auto result = func();
            if (result && !result->error_message) {
                return true;
            }
            if (result && result->error_message) {
                std::cerr << "Attempt " << (i+1) 
                          << " failed: " << result->error_message 
                          << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
        
        if (i < max_retries - 1) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    return false;
}
```

## Best Practices

1. **Always Check Errors**: Both APIs can return error states
2. **Free Resources**: Always free results to prevent memory leaks
3. **Use RAII**: Wrap API calls in RAII classes for automatic cleanup
4. **Batch Operations**: Group inserts for better performance
5. **Connection Reuse**: With modern API, reuse connections when possible

## Performance Tips

1. **Output Formats**: Use binary formats for large results
2. **Batch Size**: Insert 1000-10000 rows at a time
3. **Indexes**: Design tables with appropriate ORDER BY
4. **Partitioning**: Use partitions for time-series data
5. **Compression**: ClickHouse compresses data automatically

## Migration Guide (V2 to Modern API)

When the modern API becomes stable, migration will involve:

```cpp
// V2 API
auto result = query_stable_v2(argc, argv);
if (result && !result->error_message) {
    process(result->buf);
}
free_result_v2(result);

// Modern API equivalent
auto conn = chdb_connect(2, conn_argv);
auto result = chdb_query(conn, query, "CSV");
if (!chdb_result_error(result)) {
    process(chdb_result_buffer(result));
}
chdb_destroy_query_result(result);
chdb_close_conn(conn);
```