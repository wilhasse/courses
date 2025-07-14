# Embedded vs External chDB Implementation

## The Problem

The original `chdb_clickhouse_tvf.cpp` was trying to execute an external binary using `popen()`:

```cpp
// WRONG APPROACH - External execution
std::string execute_chdb_query(const std::string& query) {
    std::string cmd = std::string(CHDB_BINARY) + 
                     " --path='" + CLICKHOUSE_DATA_PATH + 
                     "' --query=\"" + query + "\" --format=TabSeparated 2>/dev/null";
    
    FILE* pipe = popen(cmd.c_str(), "r");  // External process!
    // ...
}
```

This approach has several problems:
1. Requires an external clickhouse binary to exist
2. Creates a new process for each query (slow)
3. Has permission issues when MySQL tries to execute external programs
4. Is not how chDB is designed to be used

## The Correct Solution

The proper approach (like in `query_data_v2.cpp`) is to embed chDB directly:

```cpp
// CORRECT APPROACH - Embedded library
struct local_result_v2* executeQuery(const std::string& query) {
    // Load libchdb.so dynamically
    void* chdb_handle = dlopen("/home/cslog/chdb/libchdb.so", RTLD_LAZY);
    
    // Get function pointers
    query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
    
    // Call the function directly (no external process!)
    return query_stable_v2(argv.size(), argv.data());
}
```

## Implementation Files

1. **chdb_clickhouse_tvf.cpp** - Original (incorrect) implementation using popen()
2. **chdb_tvf_libchdb.cpp** - First attempt at embedded version
3. **chdb_tvf_embedded.cpp** - Correct embedded implementation matching query_data_v2.cpp

## Building and Using the Correct Version

```bash
# 1. Build the embedded version
chmod +x build_embedded_tvf.sh
./build_embedded_tvf.sh

# 2. Install functions in MySQL
mysql -u root -pteste < install_embedded_functions.sql

# 3. Test
mysql -u root -pteste -e "SELECT ch_customer_count();"
```

## Key Differences

| Aspect | External (Wrong) | Embedded (Correct) |
|--------|-----------------|-------------------|
| Performance | Slow (new process each query) | Fast (in-process) |
| Dependencies | Needs clickhouse binary | Only needs libchdb.so |
| Permissions | MySQL needs execute permissions | Just needs to load .so |
| Memory | Separate process memory | Shared process memory |
| Error handling | Parse stdout/stderr | Direct error codes |

## Why This Matters

chDB is designed as an **embedded** analytical engine. It's meant to run inside your application process, not as a separate server or command-line tool. This is what makes it unique compared to regular ClickHouse.

The mysql-to-chdb-example project demonstrates this correctly:
- `feed_data_v2.cpp` - Embeds chDB to load data
- `query_data_v2.cpp` - Embeds chDB to query data

Both use `dlopen()` to load `libchdb.so` and call its functions directly, which is exactly what our MySQL plugin should do.