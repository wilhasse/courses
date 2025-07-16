# Buffer Size Management Guide

## Overview

The chDB API UDF functions need to allocate memory buffers to receive responses from the ClickHouse server. This guide explains different approaches to handle varying response sizes.

## Current Limitation

By default, all chDB API UDF functions allocate a fixed buffer of 10MB. If your query returns more data than this, you'll get an error: "ERROR: Response too large".

## Solutions

### 1. Quick Fix: Increase Fixed Buffer Size

Use the provided script to increase the buffer size in all UDF files:

```bash
# Set buffer to 50MB
./scripts/update_buffer_size.sh 50

# Set buffer to 100MB
./scripts/update_buffer_size.sh 100

# Rebuild and reinstall (example for IP functions)
./scripts/install_ip_udf.sh
```

**Pros:**
- Simple and quick
- No code changes needed
- Works with existing functions

**Cons:**
- Wastes memory for small queries
- Still has a fixed limit
- Each connection uses the full buffer

### 2. Adaptive Buffer Function

Install the adaptive buffer version that grows dynamically:

```bash
./scripts/install_adaptive_udf.sh
```

Usage:
```sql
-- Automatically handles any size up to 1GB
SELECT CAST(chdb_api_query_adaptive('SELECT * FROM huge_table') AS CHAR);

-- Check buffer configuration
SELECT chdb_api_buffer_info();
```

**Pros:**
- Memory efficient (starts at 1MB)
- Grows automatically as needed
- Handles up to 1GB responses
- No manual configuration

**Cons:**
- Requires new function name
- Slight overhead for buffer management

### 3. Query Optimization (Recommended First)

Before increasing buffers, optimize your queries:

```sql
-- Add LIMIT
SELECT * FROM table LIMIT 10000

-- Use aggregation
SELECT COUNT(*), AVG(value) FROM table GROUP BY category

-- Select only needed columns
SELECT id, name FROM table  -- not SELECT *

-- Use ClickHouse sampling
SELECT * FROM table SAMPLE 0.1  -- 10% sample
```

## Memory Considerations

### Buffer Size Impact

| Buffer Size | Memory per Connection | 10 Connections | 100 Connections |
|------------|---------------------|----------------|-----------------|
| 10MB       | 10MB                | 100MB          | 1GB             |
| 50MB       | 50MB                | 500MB          | 5GB             |
| 100MB      | 100MB               | 1GB            | 10GB            |
| Adaptive   | 1MB-1GB (as needed) | 10MB-10GB      | 100MB-100GB     |

### Recommendations by Use Case

1. **Normal Queries (< 10MB results)**
   - Keep default 10MB buffer
   - Use standard functions

2. **Large Reports (10-50MB)**
   - Update buffer to 50MB: `./scripts/update_buffer_size.sh 50`
   - Or use adaptive function for efficiency

3. **Data Export (50MB+)**
   - Use adaptive function
   - Consider pagination or streaming
   - Use ClickHouse export features

4. **Analytics Dashboards**
   - Keep small buffers (10-20MB)
   - Use aggregation queries
   - Cache results in MySQL

## Advanced: Custom Buffer Sizes per Function

You can have different functions with different buffer sizes:

```cpp
// In chdb_api_small_udf.cpp
#define MAX_RESULT_SIZE 5242880   // 5MB for small queries

// In chdb_api_large_udf.cpp  
#define MAX_RESULT_SIZE 104857600 // 100MB for reports
```

Then install as separate functions:
```sql
CREATE FUNCTION chdb_small_query RETURNS STRING SONAME 'chdb_api_small_udf.so';
CREATE FUNCTION chdb_large_query RETURNS STRING SONAME 'chdb_api_large_udf.so';
```

## Monitoring Buffer Usage

To monitor actual response sizes:

```sql
-- Create a logging table
CREATE TABLE chdb_query_log (
    query_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query_text TEXT,
    response_size INT,
    execution_time FLOAT
);

-- Log queries (in your application)
SET @start = NOW(6);
SET @result = chdb_api_query_adaptive('SELECT ...');
INSERT INTO chdb_query_log (query_text, response_size, execution_time)
VALUES ('SELECT ...', LENGTH(@result), TIMESTAMPDIFF(MICROSECOND, @start, NOW(6))/1000000);
```

## Best Practices

1. **Start with query optimization** before increasing buffers
2. **Monitor actual usage** to right-size buffers
3. **Use adaptive function** for variable workloads
4. **Set connection limits** to control total memory usage
5. **Consider alternatives** for very large data:
   - Direct ClickHouse exports
   - Streaming solutions
   - Batch processing

## Troubleshooting

### "Response too large" errors
1. Check actual response size
2. Increase buffer or use adaptive function
3. Optimize query to return less data

### Out of memory errors
1. Too many connections with large buffers
2. Reduce buffer size or connection limit
3. Use adaptive function

### Slow performance
1. Network transfer of large data
2. Consider local ClickHouse access
3. Use aggregation to reduce data size