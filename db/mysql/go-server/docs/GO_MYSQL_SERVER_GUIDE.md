# go-mysql-server Complete Guide

This guide explains how to use go-mysql-server to build your own MySQL-compatible database server with a custom storage backend.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Interfaces](#core-interfaces)
4. [Implementation Steps](#implementation-steps)
5. [SQL Processing Flow](#sql-processing-flow)
6. [Storage Layer Design](#storage-layer-design)
7. [Running and Testing](#running-and-testing)
8. [Advanced Features](#advanced-features)

## Overview

go-mysql-server is a SQL engine that provides:
- MySQL wire protocol compatibility
- SQL parsing (using Vitess parser)
- Query optimization and execution
- Pluggable storage backends

You provide the storage layer, and go-mysql-server handles everything else.

## Architecture

```
┌─────────────────┐
│  MySQL Client   │
└────────┬────────┘
         │ MySQL Protocol
┌────────▼────────┐
│  go-mysql-server│
│     Server      │
└────────┬────────┘
         │
┌────────▼────────┐
│   SQL Engine    │
│  (Analyzer)     │
└────────┬────────┘
         │
┌────────▼────────┐
│ DatabaseProvider│
└────────┬────────┘
         │
┌────────▼────────┐
│    Database     │
└────────┬────────┘
         │
┌────────▼────────┐
│     Table       │
└────────┬────────┘
         │
┌────────▼────────┐
│  Storage Layer  │
│   (Your Code)   │
└─────────────────┘
```

## Core Interfaces

### 1. DatabaseProvider
Manages multiple databases and implements `sql.DatabaseProvider`:

```go
type DatabaseProvider interface {
    Database(ctx *sql.Context, name string) (sql.Database, error)
    HasDatabase(ctx *sql.Context, name string) bool
    AllDatabases(ctx *sql.Context) []sql.Database
}

// For CREATE/DROP DATABASE support:
type MutableDatabaseProvider interface {
    DatabaseProvider
    CreateDatabase(ctx *sql.Context, name string) error
    DropDatabase(ctx *sql.Context, name string) error
}
```

### 2. Database
Represents a single database and manages tables:

```go
type Database interface {
    Name() string
    GetTableInsensitive(ctx *sql.Context, name string) (sql.Table, bool, error)
    GetTableNames(ctx *sql.Context) ([]string, error)
}

// For CREATE/DROP TABLE support:
type TableCreator interface {
    CreateTable(ctx *sql.Context, name string, schema sql.PrimaryKeySchema, collation sql.CollationID, comment string) error
}

type TableDropper interface {
    DropTable(ctx *sql.Context, name string) error
}
```

### 3. Table
Represents a table and provides data access:

```go
type Table interface {
    String() string
    Schema() sql.Schema
    Collation() sql.CollationID
    Partitions(ctx *sql.Context) (sql.PartitionIter, error)
    PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error)
}

// For INSERT/UPDATE/DELETE support:
type UpdatableTable interface {
    Table
    Updater(ctx *sql.Context) sql.RowUpdater
    Inserter(ctx *sql.Context) sql.RowInserter
    Deleter(ctx *sql.Context) sql.RowDeleter
}
```

### 4. Partitions and Row Iteration
Tables return data through partitions:

```go
type Partition interface {
    Key() []byte
}

type PartitionIter interface {
    Next(ctx *sql.Context) (sql.Partition, error)
    Close(ctx *sql.Context) error
}

type RowIter interface {
    Next(ctx *sql.Context) (sql.Row, error)
    Close(ctx *sql.Context) error
}
```

## Implementation Steps

### Step 1: Create Storage Layer
```go
// pkg/storage/storage.go
type Storage interface {
    CreateDatabase(name string) error
    GetDatabase(name string) (*Database, error)
    
    CreateTable(database, table string, schema sql.Schema) error
    GetRows(database, table string) ([]sql.Row, error)
    InsertRow(database, table string, row sql.Row) error
    UpdateRow(database, table string, oldRow, newRow sql.Row) error
    DeleteRow(database, table string, row sql.Row) error
}
```

### Step 2: Implement Table
```go
// pkg/provider/table.go
type Table struct {
    name    string
    schema  sql.Schema
    storage Storage
    dbName  string
}

func (t *Table) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
    // Return single partition for simple implementation
    return sql.PartitionsToPartitionIter(&Partition{name: t.name}), nil
}

func (t *Table) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
    rows, err := t.storage.GetRows(t.dbName, t.name)
    if err != nil {
        return nil, err
    }
    return sql.RowsToRowIter(rows...), nil
}
```

### Step 3: Implement Database
```go
// pkg/provider/database.go
type Database struct {
    name    string
    storage Storage
}

func (db *Database) GetTableInsensitive(ctx *sql.Context, tblName string) (sql.Table, bool, error) {
    schema, err := db.storage.GetTableSchema(db.name, tblName)
    if err != nil {
        return nil, false, nil
    }
    
    table := &Table{
        name:    tblName,
        schema:  schema,
        storage: db.storage,
        dbName:  db.name,
    }
    return table, true, nil
}
```

### Step 4: Implement DatabaseProvider
```go
// pkg/provider/database_provider.go
type DatabaseProvider struct {
    storage Storage
}

func (p *DatabaseProvider) Database(ctx *sql.Context, name string) (sql.Database, error) {
    if !p.storage.HasDatabase(name) {
        return nil, sql.ErrDatabaseNotFound.New(name)
    }
    
    return &Database{
        name:    name,
        storage: p.storage,
    }, nil
}
```

### Step 5: Create Server
```go
// main.go
func main() {
    // Create storage backend
    storage := NewMemoryStorage()
    
    // Create database provider
    provider := NewDatabaseProvider(storage)
    
    // Create SQL engine
    engine := sqle.New(
        analyzer.NewBuilder(provider).Build(),
        &sqle.Config{},
    )
    
    // Create server config
    config := server.Config{
        Protocol: "tcp",
        Address:  "localhost:3306",
    }
    
    // Create session builder
    sessionBuilder := func(ctx context.Context, c *mysql.Conn, addr string) (sql.Session, error) {
        return sql.NewSession(addr, c.User), nil
    }
    
    // Create context factory
    contextFactory := func(ctx context.Context, options ...sql.ContextOption) *sql.Context {
        return sql.NewContext(ctx, options...)
    }
    
    // Create and start server
    s, err := server.NewServer(config, engine, contextFactory, sessionBuilder, nil)
    if err != nil {
        panic(err)
    }
    
    s.Start()
}
```

## SQL Processing Flow

### 1. Query Reception
```
MySQL Client → Wire Protocol → Handler → SQL Engine
```

### 2. Parsing
```
SQL String → Vitess Parser → AST
```

### 3. Analysis
```
AST → Analyzer → Optimized Plan
```

The analyzer applies rules in phases:
- **once-before**: Initial validation and transformation
- **default**: Main optimization rules (can run multiple times)
- **once-after**: Final optimizations
- **validation**: Validate the final plan
- **after-all**: Final transformations

### 4. Execution
```
Plan → Executor → Row Iterator → Results
```

## Storage Layer Design

### In-Memory Storage (Example)
```go
type MemoryStorage struct {
    databases map[string]*memoryDatabase
    mu        sync.RWMutex
}

type memoryDatabase struct {
    name   string
    tables map[string]*memoryTable
}

type memoryTable struct {
    name   string
    schema sql.Schema
    rows   []sql.Row
}
```

### Persistent Storage Options

1. **File-based**: Save tables as JSON/CSV files
2. **Embedded DB**: Use BoltDB, BadgerDB, or SQLite
3. **External DB**: PostgreSQL, MySQL as storage backend
4. **Cloud Storage**: S3, GCS for data files

## Running and Testing

### Basic Server
```bash
go run main.go
```

### Debug Server (with trace logging)
```bash
go run cmd/debug-server/main.go
```

### Connect with MySQL Client
```bash
mysql -h 127.0.0.1 -P 3306 -u root

# Test queries
USE testdb;
SELECT * FROM users;
SELECT * FROM products WHERE price > 50;
```

### Testing with Go
```go
func TestServer(t *testing.T) {
    // Create test storage
    storage := NewMemoryStorage()
    storage.CreateDatabase("test")
    
    // Create provider and engine
    provider := NewDatabaseProvider(storage)
    engine := sqle.New(analyzer.NewBuilder(provider).Build(), nil)
    
    // Execute query
    ctx := sql.NewContext(context.Background())
    _, iter, err := engine.Query(ctx, "SELECT 1")
    require.NoError(t, err)
    
    rows, err := sql.RowIterToRows(ctx, iter)
    require.NoError(t, err)
    require.Equal(t, []sql.Row{{1}}, rows)
}
```

## Advanced Features

### 1. Index Support
```go
type IndexedTable interface {
    sql.Table
    GetIndexes(ctx *sql.Context) ([]sql.Index, error)
}

type Index interface {
    ID() string
    Table() string
    Expressions() []sql.Expression
    IsUnique() bool
    IsSpatial() bool
    IsFullText() bool
}
```

### 2. Filter Pushdown
```go
type FilteredTable interface {
    sql.Table
    WithFilters(ctx *sql.Context, filters []sql.Expression) sql.Table
}

func (t *Table) WithFilters(ctx *sql.Context, filters []sql.Expression) sql.Table {
    // Return new table instance that applies filters during PartitionRows
    return &Table{
        name:    t.name,
        schema:  t.schema,
        storage: t.storage,
        dbName:  t.dbName,
        filters: filters, // Apply these during row iteration
    }
}
```

### 3. Projection Pushdown
```go
type ProjectedTable interface {
    sql.Table
    WithProjection(colNames []string) sql.Table
}
```

### 4. Statistics for Query Optimization
```go
type StatisticsTable interface {
    sql.Table
    NumRows(ctx *sql.Context) (uint64, error)
    DataLength(ctx *sql.Context) (uint64, error)
}
```

### 5. Transactions
```go
type TransactionalDatabase interface {
    sql.Database
    GetTransaction(ctx *sql.Context) sql.Transaction
}

type Transaction interface {
    Commit() error
    Rollback() error
}
```

### 6. Triggers and Stored Procedures
```go
type TriggerDatabase interface {
    sql.Database
    GetTriggers(ctx *sql.Context) ([]sql.TriggerDefinition, error)
    CreateTrigger(ctx *sql.Context, definition sql.TriggerDefinition) error
    DropTrigger(ctx *sql.Context, name string) error
}
```

## Best Practices

1. **Thread Safety**: Always use mutexes for concurrent access
2. **Memory Management**: Stream large results instead of loading all into memory
3. **Error Handling**: Return appropriate SQL errors (e.g., `sql.ErrTableNotFound`)
4. **Context Handling**: Respect context cancellation
5. **Testing**: Use enginetest package for comprehensive testing

## Debugging Tips

1. Enable analyzer debug mode:
```go
analyzer := analyzer.NewBuilder(provider).Build()
analyzer.Debug = true
analyzer.Verbose = true
```

2. Add logging to your storage layer:
```go
func (s *Storage) GetRows(database, table string) ([]sql.Row, error) {
    log.Printf("GetRows called for %s.%s", database, table)
    // ... implementation
}
```

3. Use the debug server to trace execution flow

## Common Pitfalls

1. **Not implementing RowIter correctly**: Always return `io.EOF` when done
2. **Forgetting thread safety**: Multiple queries can run concurrently
3. **Not handling NULL values**: Use `nil` for NULL in sql.Row
4. **Schema mismatches**: Ensure row data matches schema types
5. **Case sensitivity**: Table/database names are case-insensitive by default

## Performance Optimization

1. **Partition wisely**: Use partitions to parallelize large table scans
2. **Implement indexes**: Support index lookup for WHERE clauses
3. **Push down filters**: Filter rows in storage layer when possible
4. **Batch operations**: Implement batch insert/update methods
5. **Cache metadata**: Cache schema information to avoid repeated lookups

## Conclusion

go-mysql-server provides a powerful foundation for building MySQL-compatible databases. By implementing the storage interfaces, you get a full SQL engine with query optimization, MySQL compatibility, and more.

The key is understanding the interface boundaries and data flow. Start simple with in-memory storage, then gradually add features like persistence, indexes, and transactions as needed.