# Hybrid Query System - Complete Guide

This guide provides a comprehensive overview of the hybrid query system that enables caching remote MySQL tables in LMDB and performing transparent cross-source joins.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup and Configuration](#setup-and-configuration)
4. [Usage Examples](#usage-examples)
5. [How It Works](#how-it-works)
6. [Testing](#testing)
7. [Limitations and Future Work](#limitations-and-future-work)

## Overview

The Hybrid Query System allows the go-mysql-server to:
- Cache specific tables from remote MySQL servers into local LMDB storage
- Execute queries transparently across both cached and remote data
- Perform in-memory joins between data from different sources
- Improve query performance for frequently accessed tables

### Key Benefits

1. **Performance**: Frequently accessed tables are cached locally for faster access
2. **Transparency**: Applications don't need to know which tables are cached
3. **Flexibility**: Choose which tables to cache based on access patterns
4. **Scalability**: Reduce load on remote MySQL servers

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Hybrid Query Handler                      │
├─────────────────┬──────────────────┬───────────────────────┤
│   Data Loader   │   SQL Parser     │   Query Rewriter      │
│                 │                  │                        │
│ Loads tables    │ Analyzes queries │ Splits queries between│
│ from MySQL      │ to detect cached │ MySQL and LMDB        │
│ to LMDB         │ table references │ sources               │
└─────────────────┴──────────────────┴───────────────────────┘
                           │
                  ┌────────┴────────┐
                  │ Join Executor    │
                  │                  │
                  │ Performs in-     │
                  │ memory joins     │
                  └─────────────────┘
```

### Data Flow

1. **Load Phase**: Tables are copied from remote MySQL to local LMDB
2. **Query Phase**: 
   - SQL parser identifies which tables are cached
   - Query rewriter creates separate queries for each source
   - Join executor combines results

## Setup and Configuration

### 1. Create the Hybrid Handler

```go
import "mysql-server-example/pkg/hybrid"

config := hybrid.Config{
    MySQLDSN: "root:password@tcp(remote-host:3306)/database",
    LMDBPath: "./cache",
    Logger:   logger,
}

handler, err := hybrid.NewHybridHandler(config)
if err != nil {
    log.Fatal(err)
}
defer handler.Close()
```

### 2. Load Tables into Cache

```go
// Load a table from remote MySQL into LMDB
err = handler.LoadTable("database", "tablename")
if err != nil {
    log.Fatal(err)
}

// Register the table as cached
handler.SQLParser.RegisterCachedTable("database", "tablename")
```

### 3. Execute Queries

```go
// Execute any SQL query - the system handles routing automatically
result, err := handler.ExecuteQuery(
    "SELECT * FROM cached_table JOIN remote_table ON ...", 
    "database"
)
```

## Usage Examples

### Example 1: Simple Cached Table Query

```go
// Load employees table into cache
handler.LoadTable("testdb", "employees")

// Query cached table - executed from LMDB
result, _ := handler.ExecuteQuery(
    "SELECT * FROM employees WHERE department = 'Engineering'", 
    "testdb"
)
```

### Example 2: JOIN Between Cached and Remote Tables

```go
// employees is cached, employee_notes remains on remote MySQL
query := `
    SELECT 
        e.id,
        e.first_name,
        e.last_name,
        n.note,
        n.created_at
    FROM employees e
    JOIN employee_notes n ON e.id = n.emp_id
    WHERE e.department = 'Sales'
`

result, _ := handler.ExecuteQuery(query, "testdb")
```

### Example 3: Complex Multi-Table Query

```go
// Mix of cached and remote tables
query := `
    SELECT 
        e.name,
        d.department_name,
        p.project_name
    FROM employees e                    -- cached
    JOIN departments d ON e.dept_id = d.id  -- remote
    JOIN projects p ON e.id = p.lead_id     -- remote
    WHERE e.is_active = 1
`

result, _ := handler.ExecuteQuery(query, "testdb")
```

## How It Works

### 1. Query Analysis

When a query is submitted, the SQL parser:
- Identifies all table references
- Checks which tables are cached in LMDB
- Extracts join conditions
- Determines if query rewriting is needed

```go
analysis := handler.SQLParser.AnalyzeQuery(query, database)
// Returns:
// - HasCachedTable: true
// - CachedTables: [{database: "testdb", table: "employees"}]
// - RemoteTables: [{database: "testdb", table: "employee_notes"}]
// - JoinConditions: [{left: "e.id", right: "n.emp_id", operator: "="}]
```

### 2. Query Rewriting

If the query involves both cached and remote tables:
- Remove cached tables from the query
- Ensure join columns are included in SELECT
- Generate a query for MySQL with only remote tables

Original:
```sql
SELECT e.name, n.note 
FROM employees e 
JOIN employee_notes n ON e.id = n.emp_id
```

Rewritten for MySQL:
```sql
SELECT n.note, n.emp_id 
FROM employee_notes n
```

### 3. Parallel Execution

- Execute rewritten query on MySQL
- Retrieve all data from cached tables in LMDB
- Both operations can happen in parallel

### 4. In-Memory Join

The join executor:
- Uses extracted join conditions
- Performs hash or nested loop joins
- Combines results from both sources
- Returns unified result set

## Testing

### Running Tests

```bash
# Setup remote test data
mysql -h 10.1.0.7 -u root testdb < test/create_remote_test_tables.sql

# Run clean demo
make test-hybrid-clean

# Run comprehensive tests
make test-hybrid-working
make test-hybrid-final
```

### Test Scenarios

1. **Basic Caching**: Load and query cached tables
2. **Simple JOINs**: Join between one cached and one remote table
3. **Complex Queries**: Multiple tables with various join conditions
4. **Performance**: Compare cached vs non-cached query times

### Example Test Output

```
=== Hybrid Query System - Working Demo ===

1. Caching employees table from remote MySQL server (10.1.0.7)
   ✓ Successfully cached 15 employees

2. Query cached employees table:
   Employee #1: John Smith
   Employee #2: Sarah Johnson
   Employee #3: Michael Williams

3. JOIN cached employees with remote employee_notes:
   Found 5 employee notes:
   - Employee #1: Great performance this quarter
   - Employee #2: Completed certification program
   - Employee #3: Leading new project initiative
   - Employee #1: Promoted to senior position
   - Employee #4: Excellent teamwork
```

## Limitations and Future Work

### Current Limitations

1. **Query Types**: Only SELECT queries are supported for cached tables
2. **WHERE Clauses**: Complex WHERE conditions on cached tables in JOINs may not work
3. **ORDER BY**: Cannot order by cached table columns in cross-source queries
4. **Transactions**: No transaction support across sources
5. **Data Freshness**: Cached data is static until manually refreshed

### Future Enhancements

1. **Automatic Refresh**: TTL-based cache expiration
2. **Write-Through Cache**: Support INSERT/UPDATE/DELETE operations
3. **Query Optimization**: Smarter join algorithms based on data size
4. **Partial Caching**: Cache only specific columns or rows
5. **Statistics**: Track cache hit rates and performance metrics

## Performance Considerations

### When to Use

The hybrid system is beneficial when:
- Tables are frequently accessed but rarely updated
- Network latency to remote MySQL is high
- Remote MySQL server is under heavy load
- Specific tables are much smaller than the full database

### Cache Size Planning

```go
// Check cache statistics
stats := handler.GetStats()
fmt.Printf("Cached tables: %v\n", stats.CachedTables)

// Refresh cached data
handler.RefreshTable("database", "tablename")
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   ```
   Failed to connect to MySQL: dial tcp: connection refused
   ```
   - Check remote MySQL is accessible
   - Verify credentials and firewall rules

2. **LMDB Errors**
   ```
   Failed to create LMDB client: no such file or directory
   ```
   - Ensure cache directory exists
   - Check disk space and permissions

3. **Join Failures**
   ```
   No join conditions found - would produce cartesian product
   ```
   - Verify join conditions are properly specified
   - Check table aliases match

### Debug Mode

Enable debug logging to see query analysis and rewriting:

```go
logger := zerolog.New(os.Stdout).Level(zerolog.DebugLevel)
```

## API Reference

### HybridHandler Methods

- `LoadTable(database, table string) error` - Load table into cache
- `RefreshTable(database, table string) error` - Refresh cached data
- `IsTableCached(database, table string) bool` - Check if table is cached
- `ExecuteQuery(query, database string) (*QueryResult, error)` - Execute query
- `GetStats() Stats` - Get cache statistics

### Configuration Options

```go
type Config struct {
    MySQLDSN string  // Remote MySQL connection string
    LMDBPath string  // Local cache directory path
    Logger   Logger  // Zerolog logger instance
}
```

## Conclusion

The Hybrid Query System provides a powerful way to optimize database performance by intelligently caching frequently accessed tables while maintaining query compatibility. It's particularly useful for read-heavy workloads where certain reference tables are accessed repeatedly.