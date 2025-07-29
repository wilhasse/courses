# Hybrid Query System

The Hybrid Query System allows the go-mysql-server to cache specific tables in LMDB while keeping others in remote MySQL, and automatically perform cross-source joins when queries reference both cached and remote tables.

## Overview

The system consists of several components:

1. **Data Loader** (`data_loader.go`) - Loads tables from MySQL into LMDB cache
2. **SQL Parser** (`sql_parser.go`) - Analyzes queries to detect cached table references
3. **Query Rewriter** (`query_rewriter.go`) - Splits queries between MySQL and LMDB sources
4. **Join Executor** (`join_executor.go`) - Performs joins between results from different sources
5. **Hybrid Handler** (`hybrid_handler.go`) - Main orchestrator that integrates all components

## How It Works

### 1. Loading Data

First, you load a table (e.g., ACORDO_GM) from MySQL into LMDB:

```go
handler.LoadTable("mydb", "ACORDO_GM")
```

This copies all data from the MySQL table into LMDB for fast local access.

### 2. Query Analysis

When a query is executed, the system analyzes it to detect:
- Which tables are referenced
- Which tables are cached in LMDB
- Whether the query requires cross-source joins

### 3. Query Rewriting

If a query references both cached and remote tables, it's rewritten:

Original query:
```sql
SELECT a.*, b.status 
FROM ACORDO_GM a 
JOIN transactions b ON a.id = b.acordo_id 
WHERE a.status = 'ACTIVE'
```

Rewritten for MySQL (without ACORDO_GM):
```sql
SELECT b.status 
FROM transactions b 
WHERE 1=1
```

The ACORDO_GM data is fetched separately from LMDB.

### 4. Cross-Source Join

Results from MySQL and LMDB are joined in memory using the join conditions from the original query.

## Usage Example

```go
// Create hybrid handler
config := hybrid.Config{
    MySQLDSN: "root:password@tcp(remotehost:3306)/production",
    LMDBPath: "/var/lib/mysql-server/cache",
    Logger:   logger,
}

handler, err := hybrid.NewHybridHandler(config)
if err != nil {
    log.Fatal(err)
}
defer handler.Close()

// Load ACORDO_GM table into cache
err = handler.LoadTable("production", "ACORDO_GM")
if err != nil {
    log.Fatal(err)
}

// Execute a hybrid query
query := `
    SELECT a.id, a.name, t.amount 
    FROM ACORDO_GM a 
    JOIN transactions t ON a.id = t.acordo_id 
    WHERE t.date > '2024-01-01'
`

result, err := handler.ExecuteQuery(query, "production")
if err != nil {
    log.Fatal(err)
}

// Process results
for _, row := range result.Rows {
    fmt.Println(row)
}
```

## Integration with go-mysql-server

To integrate with the existing server, wrap your database provider:

```go
// Create original provider
originalProvider := provider.NewDatabaseProvider(storage, logger)

// Wrap with hybrid capabilities
hybridProvider := hybrid.NewHybridDatabaseProvider(originalProvider, hybridHandler)

// Use hybrid provider in engine
engine := sqle.NewDefault(hybridProvider)
```

## Admin Commands

The system provides admin commands for cache management:

- `HYBRID_LOAD_TABLE <database> <table>` - Load a table into cache
- `HYBRID_REFRESH_TABLE <database> <table>` - Refresh cached data
- `HYBRID_STATUS` - Show cache status
- `HYBRID_ENABLE` - Enable hybrid queries
- `HYBRID_DISABLE` - Disable hybrid queries

## Performance Considerations

1. **Cache Size**: LMDB can handle large datasets, but ensure adequate disk space
2. **Join Performance**: Cross-source joins are performed in memory, so be mindful of result set sizes
3. **Data Freshness**: Cached data is static until refreshed manually

## Limitations

1. Currently supports only SELECT queries with cached tables
2. Complex queries with subqueries may not be fully supported
3. Transactions across cached and remote tables are not supported
4. Only equi-joins are currently implemented

## Future Enhancements

1. Automatic cache refresh based on TTL
2. Support for more join types (LEFT JOIN, RIGHT JOIN)
3. Query result caching
4. Partial table caching (specific columns or rows)
5. Write-through cache for INSERT/UPDATE/DELETE

## Testing

Run integration tests:

```bash
INTEGRATION_TEST=true go test ./pkg/hybrid -v
```

Run benchmarks:

```bash
INTEGRATION_TEST=true go test ./pkg/hybrid -bench=. -benchmem
```