# MySQL Crash Solution: Wrapper Process Approach

## The Problem

When we tried to directly embed libchdb.so (722MB) into MySQL as a UDF plugin, it crashed MySQL server with:
```
ERROR 2013 (HY000): Lost connection to MySQL server during query
```

This happened because:
1. **libchdb.so is massive** - 722MB is huge for a MySQL plugin
2. **Symbol conflicts** - ClickHouse and MySQL may have conflicting symbols
3. **Memory/threading issues** - ClickHouse expects to control the process

## The Solution: Wrapper Process

Instead of embedding chDB directly into MySQL, we use a two-part approach:

### Part 1: Lightweight MySQL Plugin (`mysql_chdb_tvf_wrapper.so`)
- Small plugin that only handles MySQL UDF interface
- Uses `popen()` to execute queries via helper program
- No heavy dependencies, no risk of crashing MySQL

### Part 2: Helper Program (`chdb_query_helper`)
- Separate executable that loads libchdb.so
- Runs as a child process with its own memory space
- If it crashes, MySQL remains unaffected

## Architecture

```
MySQL Process
    ↓
mysql_chdb_tvf_wrapper.so (lightweight)
    ↓ (popen)
chdb_query_helper (separate process)
    ↓ (dlopen)
libchdb.so (722MB)
    ↓
ClickHouse Data
```

## Benefits

1. **Stability** - MySQL can't crash from chDB issues
2. **Isolation** - Each query runs in a fresh process
3. **Simplicity** - Easy to debug and maintain
4. **Compatibility** - Works with any MySQL version

## Performance Considerations

- Each query spawns a new process (slower than embedded)
- But queries against ClickHouse data are typically analytical (not high-frequency)
- The safety benefit outweighs the performance cost

## Usage

```sql
-- Same interface as before
SELECT ch_customer_count();
SELECT ch_query_scalar('SELECT AVG(age) FROM mysql_import.customers');
```

## Alternative Approaches

If you need better performance:
1. Use chDB's Python API and connect via MySQL's Python UDF support
2. Create a REST API service using chDB and query it from MySQL
3. Use MySQL's FEDERATED engine to connect to a chDB server

But for simple integration, the wrapper approach is the safest and most reliable.