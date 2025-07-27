# go-mysql-server Quick Reference

## Quick Start

### 1. Minimal Server
```go
package main

import (
    "github.com/dolthub/go-mysql-server/server"
    "github.com/dolthub/go-mysql-server/sql"
    "github.com/dolthub/go-mysql-server/sql/analyzer"
    sqle "github.com/dolthub/go-mysql-server"
)

func main() {
    provider := createProvider() // Your DatabaseProvider
    engine := sqle.New(analyzer.NewBuilder(provider).Build(), nil)
    
    config := server.Config{
        Protocol: "tcp",
        Address:  "localhost:3306",
    }
    
    sessionBuilder := createSessionBuilder()
    contextFactory := sql.NewContext
    
    s, _ := server.NewServer(config, engine, contextFactory, sessionBuilder, nil)
    s.Start()
}
```

### 2. Connect and Test
```bash
mysql -h localhost -P 3306 -u root
```

## Essential Interfaces

### DatabaseProvider
```go
type MyDatabaseProvider struct {
    databases map[string]*MyDatabase
}

func (p *MyDatabaseProvider) Database(ctx *sql.Context, name string) (sql.Database, error)
func (p *MyDatabaseProvider) HasDatabase(ctx *sql.Context, name string) bool
func (p *MyDatabaseProvider) AllDatabases(ctx *sql.Context) []sql.Database
func (p *MyDatabaseProvider) CreateDatabase(ctx *sql.Context, name string) error
func (p *MyDatabaseProvider) DropDatabase(ctx *sql.Context, name string) error
```

### Database
```go
type MyDatabase struct {
    name   string
    tables map[string]*MyTable
}

func (d *MyDatabase) Name() string
func (d *MyDatabase) GetTableInsensitive(ctx *sql.Context, name string) (sql.Table, bool, error)
func (d *MyDatabase) GetTableNames(ctx *sql.Context) ([]string, error)
```

### Table
```go
type MyTable struct {
    name   string
    schema sql.Schema
    data   []sql.Row
}

func (t *MyTable) String() string
func (t *MyTable) Schema() sql.Schema
func (t *MyTable) Partitions(ctx *sql.Context) (sql.PartitionIter, error)
func (t *MyTable) PartitionRows(ctx *sql.Context, p sql.Partition) (sql.RowIter, error)
```

### RowIter
```go
type MyRowIter struct {
    rows  []sql.Row
    index int
}

func (i *MyRowIter) Next(ctx *sql.Context) (sql.Row, error) {
    if i.index >= len(i.rows) {
        return nil, io.EOF
    }
    row := i.rows[i.index]
    i.index++
    return row, nil
}

func (i *MyRowIter) Close(ctx *sql.Context) error {
    return nil
}
```

## Common Patterns

### Single Partition Table
```go
func (t *MyTable) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
    return sql.PartitionsToPartitionIter(
        &singlePartition{name: t.name},
    ), nil
}
```

### In-Memory Row Storage
```go
func (t *MyTable) PartitionRows(ctx *sql.Context, p sql.Partition) (sql.RowIter, error) {
    return sql.RowsToRowIter(t.rows...), nil
}
```

### Schema Definition
```go
schema := sql.Schema{
    {Name: "id", Type: types.Int32, Nullable: false, PrimaryKey: true},
    {Name: "name", Type: types.Text, Nullable: false},
    {Name: "age", Type: types.Int32, Nullable: true},
    {Name: "created_at", Type: types.Timestamp, Nullable: false},
}
```

### Context Factory
```go
contextFactory := func(ctx context.Context, opts ...sql.ContextOption) *sql.Context {
    return sql.NewContext(ctx, opts...)
}
```

### Session Builder
```go
sessionBuilder := func(ctx context.Context, conn *mysql.Conn, addr string) (sql.Session, error) {
    return sql.NewBaseSessionWithClientServer(
        "myserver",
        sql.Client{Address: addr, User: conn.User},
        conn.ConnectionID,
    ), nil
}
```

## SQL Type Mapping

| MySQL Type | go-mysql-server Type | Go Type |
|------------|---------------------|---------|
| TINYINT | types.Int8 | int8 |
| SMALLINT | types.Int16 | int16 |
| INT | types.Int32 | int32 |
| BIGINT | types.Int64 | int64 |
| FLOAT | types.Float32 | float32 |
| DOUBLE | types.Float64 | float64 |
| VARCHAR(n) | types.VarChar(n) | string |
| TEXT | types.Text | string |
| BLOB | types.Blob | []byte |
| DATE | types.Date | time.Time |
| DATETIME | types.Datetime | time.Time |
| TIMESTAMP | types.Timestamp | time.Time |
| BOOLEAN | types.Boolean | bool |
| JSON | types.JSON | string/interface{} |

## Advanced Features

### Updatable Table
```go
func (t *MyTable) Inserter(ctx *sql.Context) sql.RowInserter {
    return &myInserter{table: t}
}

func (t *MyTable) Updater(ctx *sql.Context) sql.RowUpdater {
    return &myUpdater{table: t}
}

func (t *MyTable) Deleter(ctx *sql.Context) sql.RowDeleter {
    return &myDeleter{table: t}
}
```

### Filter Pushdown
```go
func (t *MyTable) WithFilters(ctx *sql.Context, filters []sql.Expression) sql.Table {
    return &MyTable{
        name:    t.name,
        schema:  t.schema,
        data:    t.data,
        filters: filters,
    }
}
```

### Index Support
```go
func (t *MyTable) GetIndexes(ctx *sql.Context) ([]sql.Index, error) {
    return []sql.Index{
        &myIndex{
            id:     "PRIMARY",
            table:  t.name,
            unique: true,
            exprs:  []sql.Expression{expression.NewGetField(0, types.Int32, "id", false)},
        },
    }, nil
}
```

## Debugging

### Enable Analyzer Debug
```go
analyzer := analyzer.NewBuilder(provider).Build()
analyzer.Debug = true
analyzer.Verbose = true
```

### Log Queries
```go
type loggingProvider struct {
    sql.DatabaseProvider
}

func (p *loggingProvider) Database(ctx *sql.Context, name string) (sql.Database, error) {
    log.Printf("Database lookup: %s", name)
    return p.DatabaseProvider.Database(ctx, name)
}
```

### Trace Row Access
```go
type debugRowIter struct {
    sql.RowIter
    table string
}

func (i *debugRowIter) Next(ctx *sql.Context) (sql.Row, error) {
    row, err := i.RowIter.Next(ctx)
    if err == nil {
        log.Printf("Read row from %s: %v", i.table, row)
    }
    return row, err
}
```

## Error Handling

### Common Errors
```go
// Database not found
sql.ErrDatabaseNotFound.New(dbName)

// Table not found
sql.ErrTableNotFound.New(tableName)

// Column not found
sql.ErrColumnNotFound.New(colName)

// Duplicate entry
sql.ErrPrimaryKeyViolation.New(fmt.Sprintf("Duplicate entry '%v'", key))
```

### Context Cancellation
```go
func (i *MyRowIter) Next(ctx *sql.Context) (sql.Row, error) {
    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
        // Continue normal processing
    }
}
```

## Performance Tips

1. **Use Partitions**: Split large tables into partitions for parallel processing
2. **Implement Indexes**: Add index support for WHERE clause optimization
3. **Push Down Filters**: Filter rows at the storage layer
4. **Stream Results**: Use iterators instead of loading all data
5. **Cache Schemas**: Avoid repeated schema lookups

## Testing

### Unit Test Example
```go
func TestTableScan(t *testing.T) {
    table := &MyTable{
        name: "test",
        schema: sql.Schema{
            {Name: "id", Type: types.Int32},
            {Name: "name", Type: types.Text},
        },
        data: []sql.Row{
            {int32(1), "Alice"},
            {int32(2), "Bob"},
        },
    }
    
    ctx := sql.NewEmptyContext()
    partitions, _ := table.Partitions(ctx)
    partition, _ := partitions.Next(ctx)
    rows, _ := table.PartitionRows(ctx, partition)
    
    count := 0
    for {
        _, err := rows.Next(ctx)
        if err == io.EOF {
            break
        }
        count++
    }
    
    assert.Equal(t, 2, count)
}
```

### Integration Test Example
```go
func TestSQLQuery(t *testing.T) {
    provider := createTestProvider()
    engine := sqle.New(analyzer.NewBuilder(provider).Build(), nil)
    
    ctx := sql.NewEmptyContext()
    _, iter, err := engine.Query(ctx, "SELECT * FROM users WHERE id = 1")
    require.NoError(t, err)
    
    rows, err := sql.RowIterToRows(ctx, iter)
    require.NoError(t, err)
    require.Len(t, rows, 1)
}
```

## Resources

- [go-mysql-server GitHub](https://github.com/dolthub/go-mysql-server)
- [Documentation](https://docs.dolthub.com/sql-reference/sql-support)
- [Vitess Parser](https://vitess.io/docs/)
- [MySQL Protocol](https://dev.mysql.com/doc/internals/en/client-server-protocol.html)