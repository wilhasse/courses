# go-mysql-server Implementation Walkthrough

This document walks through the actual implementation in this project, explaining each component and how they work together.

## Project Structure

```
go-server/
├── main.go                    # Basic server entry point
├── cmd/
│   └── debug-server/
│       └── main.go           # Debug server with execution tracing
├── pkg/
│   ├── provider/
│   │   ├── database_provider.go  # Manages multiple databases
│   │   ├── database.go          # Database implementation
│   │   ├── table.go             # Table implementation
│   │   ├── partition.go         # Partition implementation
│   │   ├── iterators.go         # Row iterator implementations
│   │   └── session.go           # Session management
│   └── storage/
│       ├── storage.go           # Storage interface
│       ├── memory.go            # In-memory storage implementation
│       └── enhanced_storage.go  # Enhanced storage with features
└── docs/
    ├── sql_execution_flow.md    # How SQL queries are processed
    └── storage_abstraction.md   # Storage layer design
```

## Component Breakdown

### 1. Storage Layer (`pkg/storage/`)

The storage layer is the foundation - it stores and retrieves data.

#### Storage Interface
```go
type Storage interface {
    // Database operations
    CreateDatabase(name string) error
    DropDatabase(name string) error
    HasDatabase(name string) bool
    GetDatabaseNames() []string
    
    // Table operations
    CreateTable(database, tableName string, schema sql.Schema) error
    DropTable(database, tableName string) error
    GetTableNames(database string) ([]string, error)
    GetTableSchema(database, tableName string) (sql.Schema, error)
    
    // Row operations
    InsertRow(database, tableName string, row sql.Row) error
    UpdateRow(database, tableName string, oldRow, newRow sql.Row) error
    DeleteRow(database, tableName string, row sql.Row) error
    GetRows(database, tableName string) ([]sql.Row, error)
}
```

#### Memory Implementation
- Uses Go maps for storage
- Thread-safe with RWMutex
- Data structure:
  ```
  MemoryStorage
    └── databases (map)
          └── memoryDatabase
                ├── name
                └── tables (map)
                      └── memoryTable
                            ├── name
                            ├── schema
                            └── rows ([]sql.Row)
  ```

### 2. Provider Layer (`pkg/provider/`)

The provider layer implements go-mysql-server interfaces.

#### DatabaseProvider
- Entry point for SQL engine
- Manages database lifecycle
- Routes queries to correct database

```go
func (p *DatabaseProvider) Database(ctx *sql.Context, name string) (sql.Database, error) {
    // 1. Check if database exists in storage
    // 2. Create Database wrapper
    // 3. Return to SQL engine
}
```

#### Database
- Represents a single database
- Manages tables within database
- Handles table creation/deletion

```go
func (db *Database) GetTableInsensitive(ctx *sql.Context, tblName string) (sql.Table, bool, error) {
    // 1. Look up table schema in storage
    // 2. Create Table wrapper
    // 3. Return to SQL engine
}
```

#### Table
- Represents a single table
- Provides data through partitions
- Handles CRUD operations

```go
func (t *Table) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
    // Returns iterator over partitions (we use single partition)
}

func (t *Table) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
    // Returns iterator over rows in partition
}
```

### 3. Data Flow

#### SELECT Query Flow
```
1. MySQL Client sends: "SELECT * FROM users WHERE id > 1"
                                    ↓
2. Server receives query and creates context
                                    ↓
3. SQL Engine parses query using Vitess parser
                                    ↓
4. Analyzer optimizes query plan:
   - Resolve database/table references
   - Push down filters
   - Optimize joins
   - Select indexes
                                    ↓
5. Executor runs the plan:
   - Calls DatabaseProvider.Database("testdb")
   - Calls Database.GetTableInsensitive("users")
   - Calls Table.Partitions() → returns PartitionIter
   - For each partition:
     - Calls Table.PartitionRows() → returns RowIter
     - Iterates rows, applies WHERE filter
                                    ↓
6. Results sent back to client
```

#### INSERT Query Flow
```
1. MySQL Client sends: "INSERT INTO users VALUES (3, 'Charlie', 'charlie@example.com', NOW())"
                                    ↓
2. Parse and analyze query
                                    ↓
3. Executor:
   - Gets UpdatableTable
   - Calls Table.Inserter()
   - Calls RowInserter.Insert(row)
                                    ↓
4. Storage layer:
   - Validates row against schema
   - Appends to table's row slice
   - Returns success
```

### 4. Session Management

Each connection gets a session:

```go
type Session struct {
    baseSession *sql.BaseSession
    address     string
}

func NewSession(addr, user string) sql.Session {
    return &Session{
        baseSession: sql.NewBaseSessionWithClientServer(
            "go-mysql-server",
            sql.Client{Address: addr, User: user},
            1,
        ),
        address: addr,
    }
}
```

### 5. Debug Server Features

The debug server adds instrumentation:

#### Wrapped Components
- `DebugDatabaseProvider`: Logs database lookups
- `DebugDatabase`: Logs table lookups
- `DebugTable`: Logs partition and row access
- `debugRowIter`: Logs each row read

#### Analyzer Debugging
```go
analyzer := analyzer.NewBuilder(provider).Build()
analyzer.Debug = true    // Enables rule evaluation logs
analyzer.Verbose = true  // Shows plan transformations
```

## Key Concepts

### 1. Partitions
- Tables are divided into partitions
- Each partition can be processed independently
- Enables parallel query execution
- Simple implementation uses single partition

### 2. Row Iterators
- Lazy evaluation - rows fetched on demand
- Must return `io.EOF` when complete
- Support early termination via context

### 3. Schema Definition
```go
schema := sql.Schema{
    {Name: "id", Type: types.Int32, Nullable: false, PrimaryKey: true},
    {Name: "name", Type: types.Text, Nullable: false},
    {Name: "email", Type: types.Text, Nullable: false},
    {Name: "created_at", Type: types.Timestamp, Nullable: false},
}
```

### 4. Type System
go-mysql-server provides MySQL-compatible types:
- Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `Float32`, `Float64`
- String: `Text`, `Blob`, `VarChar(n)`
- Temporal: `Date`, `Datetime`, `Timestamp`
- Others: `Boolean`, `JSON`, `Geometry`

## Running the Servers

### Basic Server
```bash
# Terminal 1: Start server
go run main.go

# Terminal 2: Connect with MySQL client
mysql -h 127.0.0.1 -P 3306 -u root

# Try queries
mysql> USE testdb;
mysql> SELECT * FROM users;
mysql> INSERT INTO users VALUES (6, 'Frank', 'frank@example.com', NOW());
```

### Debug Server
```bash
# Terminal 1: Start debug server
go run cmd/debug-server/main.go

# Terminal 2: Connect and watch the logs
mysql -h 127.0.0.1 -P 3311 -u root

# Logs show:
# - Database/table lookups
# - Query analysis steps
# - Row scanning
# - Data being read
```

## Extending the Implementation

### 1. Add Persistence
```go
type FileStorage struct {
    baseDir string
    *MemoryStorage
}

func (s *FileStorage) InsertRow(database, table string, row sql.Row) error {
    // 1. Insert into memory
    err := s.MemoryStorage.InsertRow(database, table, row)
    
    // 2. Append to file
    file := filepath.Join(s.baseDir, database, table+".jsonl")
    // Write row as JSON line
    
    return err
}
```

### 2. Add Indexes
```go
type IndexedTable struct {
    *Table
    indexes map[string]*btree.BTree
}

func (t *IndexedTable) WithFilters(ctx *sql.Context, filters []sql.Expression) sql.Table {
    // Check if filter matches an index
    // Return optimized table that uses index
}
```

### 3. Add Transactions
```go
type TransactionalStorage struct {
    *MemoryStorage
    transactions map[uint32]*Transaction
}

type Transaction struct {
    id      uint32
    changes []Change
    mu      sync.Mutex
}
```

### 4. Add Statistics
```go
func (t *Table) NumRows(ctx *sql.Context) (uint64, error) {
    rows, err := t.storage.GetRows(t.dbName, t.name)
    return uint64(len(rows)), err
}
```

## Common Issues and Solutions

### Issue: Nil Pointer in Context Factory
```go
// Wrong:
s, err := server.NewServer(config, engine, nil, sessionFactory, nil)

// Correct:
contextFactory := func(ctx context.Context, options ...sql.ContextOption) *sql.Context {
    return sql.NewContext(ctx, options...)
}
s, err := server.NewServer(config, engine, contextFactory, sessionFactory, nil)
```

### Issue: Database Not Found
```go
// Always create default databases that MySQL expects
provider.CreateDatabase(ctx, "mysql")
provider.CreateDatabase(ctx, "information_schema")
```

### Issue: Type Mismatches
```go
// Ensure row data matches schema types
row := sql.Row{
    int32(1),                    // Must be int32, not int
    "Alice",                     // String is fine
    "alice@example.com",         // String is fine  
    time.Now().Format("2006-01-02 15:04:05"), // Format timestamp
}
```

## Performance Considerations

1. **Memory Usage**: Current implementation loads all rows into memory
   - Solution: Implement streaming with cursors

2. **Concurrent Access**: Uses simple RWMutex
   - Solution: Implement MVCC for better concurrency

3. **Large Tables**: Single partition limits parallelism
   - Solution: Implement range or hash partitioning

4. **No Query Cache**: Recomputes everything
   - Solution: Add result caching layer

## Testing

### Unit Tests
```go
func TestTableScan(t *testing.T) {
    storage := NewMemoryStorage()
    storage.CreateDatabase("test")
    storage.CreateTable("test", "users", schema)
    storage.InsertRow("test", "users", sql.Row{1, "Alice", "alice@example.com", "2023-01-01"})
    
    table := NewTable("users", schema, storage, "test")
    
    ctx := sql.NewEmptyContext()
    partitions, _ := table.Partitions(ctx)
    partition, _ := partitions.Next(ctx)
    
    rows, _ := table.PartitionRows(ctx, partition)
    row, _ := rows.Next(ctx)
    
    assert.Equal(t, int32(1), row[0])
    assert.Equal(t, "Alice", row[1])
}
```

### Integration Tests
```go
func TestSQLQueries(t *testing.T) {
    engine := createTestEngine()
    ctx := sql.NewEmptyContext()
    
    _, iter, _ := engine.Query(ctx, "SELECT * FROM users WHERE id = 1")
    rows, _ := sql.RowIterToRows(ctx, iter)
    
    assert.Len(t, rows, 1)
    assert.Equal(t, "Alice", rows[0][1])
}
```

## Conclusion

This implementation demonstrates:
1. How to implement go-mysql-server interfaces
2. How to build a storage abstraction
3. How SQL queries flow through the system
4. How to add debugging and instrumentation

The modular design makes it easy to:
- Swap storage backends
- Add new features
- Optimize performance
- Debug query execution

Start with this implementation and gradually add features based on your needs!