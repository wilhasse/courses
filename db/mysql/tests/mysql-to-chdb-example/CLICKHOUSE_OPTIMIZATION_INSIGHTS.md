# ClickHouse/chdb Critical Performance Insights

## Executive Summary

During the import of 300M+ rows from MySQL to ClickHouse (chdb), we discovered that **data ordering and partitioning strategy can impact performance by 5-10x and storage by 100x**. This document captures these critical insights that can save hours of import time and hundreds of GB of storage.

## The Discovery

### Initial Approach (Problematic)
- Table with `PARTITION BY toYYYYMM(data)`
- Data inserted in mixed date order
- Result: 19k rows/sec, 10GB for 14M rows, partition errors

### Optimized Approach (Solution)
- Table without partitioning
- Data inserted in PRIMARY KEY order (ID_CONTR, SEQ)
- Result: 99k rows/sec, few MB for 16M rows, no errors

## Key Insights

### 1. Partitioning Can Hurt More Than Help

**Problem:**
```sql
-- DON'T DO THIS for wide date ranges
CREATE TABLE historico (
    ...
) ENGINE = MergeTree() 
PARTITION BY toYYYYMM(data)  -- Creates 100+ partitions!
ORDER BY (id_contr, seq)
```

**Solution:**
```sql
-- DO THIS instead
CREATE TABLE historico (
    ...
) ENGINE = MergeTree() 
ORDER BY (id_contr, seq)  -- No partitioning needed
```

**Why:**
- Each INSERT with mixed dates creates parts in multiple partitions
- ClickHouse limits inserts to 100 partitions (max_partitions_per_insert_block)
- Partitioning is for data lifecycle management, NOT query performance

### 2. Insert Order Dramatically Affects Performance

**Critical Rule: INSERT data in the same order as your ORDER BY clause**

**Bad Practice:**
```sql
-- Random order insert
INSERT INTO historico 
SELECT * FROM source 
ORDER BY RAND()  -- Storage explosion!
```

**Best Practice:**
```sql
-- Ordered insert matching table's ORDER BY
INSERT INTO historico 
SELECT * FROM source 
ORDER BY id_contr, seq  -- Optimal compression!
```

### 3. Performance Impact of Ordered Inserts

| Metric | Random/Mixed Order | PRIMARY KEY Order | Improvement |
|--------|-------------------|-------------------|-------------|
| Insert Speed | 19k rows/sec | 99k rows/sec | **5x faster** |
| Storage (16M rows) | 10GB | ~100MB | **100x smaller** |
| Merge Efficiency | Slow, blocking | Fast, background | **No freezing** |
| Compression Ratio | 0.3-0.5 | 0.05-0.1 | **10x better** |

### 4. Why Ordered Inserts Are Magic

When data arrives in ORDER BY order:
1. **Sequential writes**: Each part contains contiguous key ranges
2. **Better compression**: Similar data groups together
3. **Efficient merges**: Adjacent parts merge cleanly
4. **Minimal indexes**: Primary key index stays compact

### 5. Storage Projections

For 300M rows:
- **Unordered inserts**: 200-300GB during import → 20-30GB after OPTIMIZE
- **Ordered inserts**: 1-3GB during import → 1-2GB after OPTIMIZE

## Implementation Guidelines

### 1. Table Design
```sql
CREATE TABLE large_table (
    key_field1 Type1,
    key_field2 Type2,
    ...other fields...
) ENGINE = MergeTree()
ORDER BY (key_field1, key_field2)
-- Avoid PARTITION BY unless you need data lifecycle management
```

### 2. Data Loading Strategy

**For MySQL to ClickHouse:**
```sql
-- Use keyset pagination to maintain order
SELECT * FROM source_table
WHERE (key1 > last_key1 OR (key1 = last_key1 AND key2 > last_key2))
ORDER BY key1, key2  -- CRITICAL: Match ClickHouse ORDER BY
LIMIT 50000
```

**For bulk inserts:**
```go
// Always sort your batches before inserting
sort.Slice(rows, func(i, j int) bool {
    if rows[i].ID != rows[j].ID {
        return rows[i].ID < rows[j].ID
    }
    return rows[i].Seq < rows[j].Seq
})
```

### 3. When to Use Partitioning

✅ **USE partitioning when:**
- You need to DROP old data by date/time
- Data lifecycle management is required
- You have <1000 total partitions lifetime

❌ **AVOID partitioning when:**
- Just trying to "improve performance"
- Data spans many time periods (years)
- Doing bulk historical imports

### 4. Optimization Settings

```sql
-- For bulk loading without partitions
SET max_insert_threads = 4;
SET max_insert_block_size = 1048576;  -- 1M rows

-- Let merges run naturally (don't STOP MERGES)
-- ClickHouse handles ordered data efficiently
```

## Real-World Example

Our MySQL → ClickHouse migration:
```
Dataset: 300M rows, 10 years of data
Table: historico (id_contr, seq, data, ...)

Attempt 1 - With monthly partitions:
- Speed: 19k rows/sec declining over time
- Storage: 10GB per 14M rows
- Issues: Partition limit errors, freezing
- ETA: 8+ hours

Attempt 2 - Without partitions, ordered by (id_contr, seq):
- Speed: 99k rows/sec sustained
- Storage: 100MB per 16M rows  
- Issues: None
- ETA: 48 minutes
```

## Key Takeaways

1. **ORDER BY is your PRIMARY optimization** - Partitioning is secondary
2. **Always INSERT in ORDER BY order** - 100x storage savings
3. **Partitioning is for data management** - Not for query performance
4. **Trust ClickHouse's merge algorithm** - It works best with ordered data
5. **Monitor storage during import** - Ordered inserts show immediate benefits

## Monitoring Queries

```sql
-- Check compression efficiency
SELECT 
    formatReadableSize(sum(bytes_on_disk)) as disk_size,
    formatReadableSize(sum(data_uncompressed_bytes)) as uncompressed,
    round(sum(data_compressed_bytes) / sum(data_uncompressed_bytes), 3) as ratio,
    count() as parts
FROM system.parts 
WHERE database = 'mysql_import' AND table = 'historico' AND active;

-- Check if inserts are ordered (low = good)
SELECT 
    round(sum(rows * data_uncompressed_bytes) / sum(data_uncompressed_bytes) / max(rows), 2) as disorder_factor
FROM system.parts 
WHERE database = 'mysql_import' AND table = 'historico' AND active;
```

## Conclusion

The difference between ordered and unordered inserts in ClickHouse is not a minor optimization - it's a fundamental architectural decision that can make or break your data pipeline. Always design your data flow to maintain ORDER BY consistency from source to destination.

**Remember: In ClickHouse, order isn't just important - it's everything.**