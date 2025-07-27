# How SQL Parsing, Execution, and Storage Work in go-mysql-server

This project demonstrates a complete MySQL-compatible server implementation using go-mysql-server. Here's how everything works together:

## 🔄 Complete SQL Execution Flow

### 1. **SQL Text → Abstract Syntax Tree (AST)**
```
"SELECT * FROM users WHERE name = 'Alice'"
    ↓ (Vitess SQL Parser)
&sqlparser.Select{
    SelectExprs: [&sqlparser.StarExpr{}],
    From: [&sqlparser.AliasedTableExpr{Expr: "users"}],
    Where: &sqlparser.ComparisonExpr{...}
}
```

### 2. **AST → go-mysql-server Plan Tree**
```
Project[*]
└── Filter[name = 'Alice']
    └── ResolvedTable[users]
```

### 3. **Optimization**
```
// Before: Filter after table scan
ResolvedTable[users] → Filter[name = 'Alice']

// After: Filter pushed down (if storage supports it)
FilteredTable[users, name = 'Alice']
```

### 4. **Execution → Storage Calls**
```go
// Your storage interface is called:
table.Partitions(ctx)                    // Get data partitions
table.PartitionRows(ctx, partition)      // Get row iterator
storage.GetRows("testdb", "users")       // Your custom storage
rowIter.Next(ctx)                        // Read each row
```

## 🏗️ Architecture Layers

### Layer 1: MySQL Protocol (go-mysql-server handles)
- **Client Connection**: MySQL wire protocol, authentication
- **SQL Parsing**: Converts SQL text to AST using Vitess parser
- **Session Management**: User sessions, variables, transactions

### Layer 2: Query Planning & Optimization (go-mysql-server handles)
- **Analysis**: Resolve table/column references, type checking
- **Optimization**: Predicate pushdown, join reordering, index selection
- **Plan Generation**: Create execution plan tree

### Layer 3: Execution Engine (go-mysql-server handles)
- **Plan Execution**: Walk the plan tree, coordinate operations
- **Join Processing**: Nested loop, hash join, merge join algorithms
- **Aggregation**: Hash-based grouping, streaming aggregation
- **Sorting & Limiting**: Order results, apply limits

### Layer 4: Storage Interface (YOU implement)
- **Database Provider**: Manage multiple databases
- **Table Interface**: Schema, partitioning, row iteration
- **Row Access**: Your custom storage backend

## 📊 How Different Operations Work

### Simple SELECT
```sql
SELECT name, email FROM users WHERE id = 1
```

**Execution Flow:**
1. Parse SQL → `Project[name, email] → Filter[id = 1] → Table[users]`
2. Call `table.PartitionRows()` → calls your `storage.GetRows()`
3. Apply filter: check each row where `id = 1`
4. Project: return only `name, email` columns
5. Send results to client

### JOIN Operations
```sql
SELECT u.name, p.name FROM users u JOIN products p ON u.id = p.user_id
```

**Execution Flow:**
1. Parse SQL → `Project[u.name, p.name] → Join[u.id = p.user_id] → [Table[users], Table[products]]`
2. **Left Table Scan**: Call `storage.GetRows("testdb", "users")`
3. **Right Table Scan**: For each left row, call `storage.GetRows("testdb", "products")`
4. **Join Logic**: Test condition `u.id = p.user_id` for each row pair
5. **Project**: Return only requested columns
6. Send results to client

**Join Algorithms Used:**
- **Nested Loop**: Default for small tables
- **Hash Join**: When one side is significantly smaller
- **Merge Join**: When both sides are pre-sorted

### GROUP BY / Aggregation
```sql
SELECT category, COUNT(*), AVG(price) FROM products GROUP BY category
```

**Execution Flow:**
1. Parse SQL → `Project[category, COUNT(*), AVG(price)] → GroupBy[category] → Table[products]`
2. **Scan Phase**: Call `storage.GetRows("testdb", "products")`
3. **Grouping Phase**: Hash table with category as key
4. **Aggregation Phase**: Update COUNT and AVG for each group
5. **Result Phase**: Emit one row per group
6. Send results to client

**Aggregation Implementation:**
- **Hash-based Grouping**: `map[groupKey]*AggregateState`
- **Streaming Aggregation**: For pre-sorted input
- **Parallel Aggregation**: For very large datasets

## 🔧 Storage Interface Requirements

### Minimum Implementation (Basic Functionality)
```go
type Storage interface {
    GetRows(database, tableName string) ([]sql.Row, error)
    CreateTable(database, tableName string, schema sql.Schema) error
    InsertRow(database, tableName string, row sql.Row) error
}
```

### Enhanced Implementation (Better Performance)
```go
type EnhancedStorage interface {
    Storage
    
    // Predicate pushdown
    GetRowsWithFilters(database, tableName string, filters []Filter) ([]sql.Row, error)
    
    // Column pruning  
    GetRowsWithProjection(database, tableName string, columns []string) ([]sql.Row, error)
    
    // Statistics for optimization
    GetTableStats(database, tableName string) (TableStats, error)
    
    // Index support
    CreateIndex(database, tableName, indexName string, columns []string) error
    LookupByIndex(database, tableName, indexName string, key interface{}) ([]sql.Row, error)
}
```

## 🚀 Performance Optimizations

### 1. **Predicate Pushdown**
```sql
-- Instead of filtering 1M rows in memory:
SELECT * FROM huge_table WHERE status = 'active'

-- Push filter to storage:
storage.GetRowsWithFilters("db", "huge_table", [Filter{Column: "status", Op: Equals, Value: "active"}])
```

### 2. **Column Pruning**
```sql
-- Instead of reading all 50 columns:
SELECT name, email FROM users

-- Only read needed columns:
storage.GetRowsWithProjection("db", "users", ["name", "email"])
```

### 3. **Index Usage**
```sql
-- Instead of full table scan:
SELECT * FROM users WHERE id = 123

-- Use index lookup:
storage.LookupByIndex("db", "users", "PRIMARY", 123)
```

### 4. **Partition Pruning**
```go
// Time-based partitioning
func (t *Table) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
    // Return only partitions that match query date range
    return &datePartitionIter{startDate: jan1, endDate: dec31}, nil
}
```

## 🎯 Key Benefits of This Architecture

### ✅ **What go-mysql-server Handles For You**
- **MySQL Protocol**: Wire protocol, authentication, SSL
- **SQL Parsing**: Complete SQL syntax support
- **Query Optimization**: Cost-based optimizer, join reordering
- **Execution Engine**: All join algorithms, aggregation, sorting
- **Standard Functions**: Math, string, date functions
- **Transactions**: ACID properties, isolation levels
- **Information Schema**: SHOW commands, metadata queries

### ✅ **What You Only Need to Implement**
- **Row Storage**: How to store/retrieve rows
- **Schema Management**: Table and column definitions
- **Basic CRUD**: Insert, Update, Delete operations

### ✅ **Optional Advanced Features**
- **Indexes**: For better query performance
- **Statistics**: For query optimization
- **Custom Functions**: Domain-specific functions
- **Partitioning**: For scalability

## 🔍 Debugging and Tracing

### Run the Debug Server
```bash
make run-trace
```

This starts a server with detailed execution tracing that shows:
- **Table Lookups**: When tables are accessed
- **Row Scanning**: Each row read from storage
- **Query Plans**: How queries are optimized
- **Execution Flow**: Step-by-step query execution

### Sample Debug Output
```
INFO[2024-01-01T10:00:00Z] 🔍 Looking up database               database=testdb
INFO[2024-01-01T10:00:01Z] ✅ Database found                   database=testdb
INFO[2024-01-01T10:00:01Z] 🔍 Looking up table                 table=users
INFO[2024-01-01T10:00:01Z] ✅ Table found                      columns=4 table=users
INFO[2024-01-01T10:00:01Z] 🔍 Getting partitions               table=users
INFO[2024-01-01T10:00:01Z] 📊 Starting table scan              table=users
DEBUG[2024-01-01T10:00:01Z] 📄 Reading row                      data="[1 Alice alice@example.com 2023-01-01 00:00:00]" row=1 table=users
DEBUG[2024-01-01T10:00:01Z] 📄 Reading row                      data="[2 Bob bob@example.com 2023-01-02 00:00:00]" row=2 table=users
INFO[2024-01-01T10:00:01Z] ✅ Finished scanning table          table=users total_rows=2
```

This shows exactly how your storage layer is being called and how the SQL engine processes your data!

## 🎪 Try It Yourself

1. **Start the server**: `make run-trace`
2. **Connect**: `mysql -h localhost -P 3306 -u root`
3. **Run queries** and watch the detailed execution logs
4. **Experiment** with different SQL constructs to see how they're processed

The debug output will show you exactly how go-mysql-server transforms your SQL into storage calls!