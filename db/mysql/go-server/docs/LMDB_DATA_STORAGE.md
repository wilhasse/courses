# LMDB Data Storage Architecture

This document explains how MySQL-like relational data is stored in the LMDB key-value database.

## 🎯 Overview

LMDB (Lightning Memory-Mapped Database) is fundamentally a **key-value store**, not a relational database. Our implementation simulates MySQL tables using a **structured key naming convention** combined with **JSON serialization**.

## 🗄️ Storage Model

### Key-Value Foundation
```
LMDB Core: Key (bytes) → Value (bytes)
Our Layer: Structured Keys → JSON Values
```

### Hierarchical Key Design
We use a hierarchical naming convention to organize relational data:

```
Pattern: db:{database}:table:{table}:row:{id}
Example: "db:testdb:table:users:row:1"
```

## 📋 Complete Key Structure

```
LMDB Storage Layout:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key                                    │ Value (JSON)
──────────────────────────────────────────────────────────────
databases                             │ ["testdb", "mydb"]
db:testdb:tables                       │ ["users", "products", "orders"]
db:testdb:table:users:schema           │ {"columns": [...]}
db:testdb:table:users:nextid           │ "66"
db:testdb:table:users:row:1            │ [1, "Alice", "alice@example.com", "2023-01-01T00:00:00Z"]
db:testdb:table:users:row:2            │ [2, "Bob", "bob@example.com", "2023-01-02T00:00:00Z"]
db:testdb:table:products:schema        │ {"columns": [...]}
db:testdb:table:products:row:1         │ [1, "Laptop", 999.99, "Electronics"]
db:mydb:tables                         │ ["customers"]
```

## 🔑 Key Categories

### 1. Database Registry
```
Key:   "databases"
Value: ["testdb", "mydb", "analytics"]
```
**Purpose**: Central registry of all databases in the system.

### 2. Table Registry
```
Key:   "db:testdb:tables"
Value: ["users", "products", "orders"]
```
**Purpose**: List of tables within each database.

### 3. Schema Storage
```
Key:   "db:testdb:table:users:schema"
Value: {
  "columns": [
    {"name": "id", "type_name": "INT", "nullable": false, "primary_key": true},
    {"name": "name", "type_name": "VARCHAR(100)", "nullable": false, "primary_key": false},
    {"name": "email", "type_name": "VARCHAR(255)", "nullable": false, "primary_key": false},
    {"name": "created_at", "type_name": "TIMESTAMP", "nullable": false, "primary_key": false}
  ]
}
```
**Purpose**: Table structure definition with column metadata.

### 4. Auto-Increment Counters
```
Key:   "db:testdb:table:users:nextid"
Value: "66"
```
**Purpose**: Track next available ID for new row insertion.

### 5. Row Data
```
Key:   "db:testdb:table:users:row:1"
Value: [1, "Alice", "alice@example.com", "2023-01-01T00:00:00Z"]
```
**Purpose**: Actual row data as JSON arrays.

## 💾 Row Storage Format

### Positional Arrays
Each row is stored as a **JSON array** where column values are stored by position:

```json
// Schema defines column order:
// [0]=id, [1]=name, [2]=email, [3]=created_at

// Row data:
[1, "Alice", "alice@example.com", "2023-01-01T00:00:00Z"]
[2, "Bob", "bob@example.com", "2023-01-02T00:00:00Z"]
```

### Data Type Handling
- **Integers**: JSON numbers (`1`, `42`)
- **Strings**: JSON strings (`"Alice"`, `"Electronics"`)
- **Decimals**: JSON numbers (`999.99`, `19.95`)
- **Timestamps**: ISO 8601 strings (`"2023-01-01T00:00:00Z"`)
- **Booleans**: JSON booleans (`true`, `false`)
- **NULL**: JSON null (`null`)

## 🔍 Data Access Patterns

### Insert Operation
```go
func (s *LMDBStorage) InsertRow(database, tableName string, row sql.Row) error {
    // 1. Get next ID from counter
    counterKey := fmt.Sprintf("db:%s:table:%s:nextid", database, tableName)
    nextID, _ := strconv.Atoi(string(counterData))
    
    // 2. Store row with generated key
    rowKey := fmt.Sprintf("db:%s:table:%s:row:%d", database, tableName, nextID)
    rowData, _ := json.Marshal(row)
    txn.Put(db, []byte(rowKey), rowData, 0)
    
    // 3. Increment counter
    newCounter := strconv.Itoa(nextID + 1)
    txn.Put(db, []byte(counterKey), []byte(newCounter), 0)
}
```

### Query Operation (Table Scan)
```go
func (s *LMDBStorage) GetRows(database, tableName string) ([]sql.Row, error) {
    // Use cursor to iterate through all matching keys
    prefix := fmt.Sprintf("db:%s:table:%s:row:", database, tableName)
    
    cursor.First()
    for key, data, err := cursor.Next(); err == nil; {
        if strings.HasPrefix(string(key), prefix) {
            var row sql.Row
            json.Unmarshal(data, &row)
            rows = append(rows, row)
        }
    }
}
```

### Schema Lookup
```go
func (s *LMDBStorage) GetTableSchema(database, tableName string) (sql.Schema, error) {
    schemaKey := fmt.Sprintf("db:%s:table:%s:schema", database, tableName)
    data, err := txn.Get(db, []byte(schemaKey))
    
    var serializableSchema SerializableSchema
    json.Unmarshal(data, &serializableSchema)
    return serializableToSchema(serializableSchema), nil
}
```

## ⚖️ Relational vs Key-Value Comparison

| **Relational Database** | **Our LMDB Implementation** |
|--------------------------|------------------------------|
| Tables with columns/rows | Hierarchical keys + JSON values |
| Primary key indexes | Auto-increment counters |
| SQL schema definitions | JSON schema metadata |
| Foreign key relationships | Application-level references |
| B-tree column indexes | LMDB's B+ tree on keys only |
| JOIN operations | Application-level joins |
| WHERE clause indexing | Full table scans |
| ACID transactions | LMDB ACID transactions ✅ |
| Concurrent access | LMDB MVCC ✅ |

## 🚀 Performance Characteristics

### Fast Operations ⚡
- **Primary key lookups**: O(log n) via LMDB B+ tree
- **Schema access**: O(log n) single key lookup
- **Transactions**: Full ACID with minimal overhead
- **Memory efficiency**: Memory-mapped files

### Slow Operations 🐌
- **WHERE clauses**: O(n) full table scan required
- **JOINs**: Must scan multiple tables
- **Column queries**: No secondary indexes
- **Complex aggregations**: Must process all rows

## 💡 Optimization Strategies

### Current Implementation
```sql
-- Fast (uses key lookup):
SELECT * FROM users WHERE id = 1;

-- Slow (requires full scan):
SELECT * FROM users WHERE name = 'Alice';
SELECT * FROM users WHERE email LIKE '%@gmail.com';
```

### Potential Improvements
1. **Secondary Indexes**: Additional keys for frequently queried columns
   ```
   db:testdb:table:users:index:email:alice@example.com → 1
   ```

2. **Composite Keys**: Multi-column lookups
   ```
   db:testdb:table:orders:index:user_id:product_id:1:5 → 42
   ```

3. **Cached Aggregations**: Pre-computed results
   ```
   db:testdb:table:users:count → 65
   ```

## 🔧 Implementation Benefits

### ✅ Advantages
- **High Performance**: Memory-mapped I/O and B+ tree efficiency
- **ACID Compliance**: Full transaction support
- **Single File**: Easy backup and deployment
- **Cross-Platform**: Works on all major operating systems
- **No Dependencies**: Embedded database, no separate server
- **Crash Safe**: Copy-on-write ensures data integrity

### ❌ Limitations
- **No Query Optimization**: Every WHERE clause scans entire table
- **Limited Joins**: Application must implement join logic
- **Schema Rigidity**: Column changes require data migration
- **Memory Usage**: Large datasets must fit in virtual memory
- **Concurrency**: Single writer (multiple readers allowed)

## 📊 Real-World Example

### Current Database State
```bash
$ make run-debug
...
📁 Databases: ["testdb"]
📋 Tables in 'testdb': ["users", "products", "orders"]
🏗️ Schema for 'users': [id, name, email, created_at]
📄 Sample rows: 65 total rows
Row 1: [1, "Alice", "alice@example.com", "2023-01-01T00:00:00Z"]
Row 2: [2, "Bob", "bob@example.com", "2023-01-02T00:00:00Z"]
```

### Actual LMDB Keys
```
databases                              → ["testdb"]
db:testdb:tables                       → ["users", "products", "orders"]
db:testdb:table:users:schema           → {"columns": [...]}
db:testdb:table:users:nextid           → "66"
db:testdb:table:users:row:1            → [1, "Alice", "alice@example.com", "2023-01-01T00:00:00Z"]
db:testdb:table:users:row:2            → [2, "Bob", "bob@example.com", "2023-01-02T00:00:00Z"]
... (63 more user rows)
db:testdb:table:products:schema        → {"columns": [...]}
db:testdb:table:products:nextid        → "9"
db:testdb:table:products:row:1         → [1, "Laptop", 999.99, "Electronics"]
... (7 more product rows)
```

## 🎯 Design Philosophy

This architecture provides a **practical middle ground**:

1. **SQL Compatibility**: Familiar MySQL interface for applications
2. **High Performance**: LMDB's proven speed and reliability
3. **Operational Simplicity**: Single file, no configuration, embedded
4. **Development Velocity**: Quick setup and iteration

While it sacrifices some advanced database features (complex indexing, query optimization), it delivers excellent performance for many common use cases, especially:

- **Prototyping and Development**: Quick setup with real persistence
- **Embedded Applications**: No external database dependencies
- **Read-Heavy Workloads**: Fast key lookups and table scans
- **Simple Data Models**: Straightforward relational structures

## 🔗 Related Documentation

- [JSON Schema Serialization](./JSON_SCHEMA_SERIALIZATION.md) - How schema persistence works
- [Debug Usage Guide](./DEBUG_USAGE.md) - Tools for inspecting stored data
- [LMDB Integration](./LMDB_INTEGRATION.md) - Setup and configuration details

This storage model transforms LMDB's simple key-value interface into a functional MySQL-compatible database! 🚀