# LMDB Integration Guide

This document explains how LMDB (Lightning Memory-Mapped Database) was integrated into the MySQL-compatible server project to provide persistent storage.

## Overview

LMDB is a high-performance, memory-mapped key-value database that provides ACID transactions and excellent read performance. This project integrates LMDB as a persistent storage backend to replace the previous in-memory storage, enabling data persistence across server restarts.

## Integration Architecture

### Storage Layer Design

The integration follows a clean layered architecture:

```
┌─────────────────────────────────────┐
│          MySQL Protocol Layer       │  (go-mysql-server)
├─────────────────────────────────────┤
│        Database Provider Layer      │  (pkg/provider/)
├─────────────────────────────────────┤
│         Storage Interface Layer     │  (pkg/storage/storage.go)
├─────────────────────────────────────┤
│           LMDB Storage Backend      │  (pkg/storage/lmdb.go)
├─────────────────────────────────────┤
│             LMDB Library            │  (lmdb-lib/)
└─────────────────────────────────────┘
```

### Key Components

#### 1. Storage Interface (`pkg/storage/storage.go`)
Defines a clean abstraction for storage backends:
- Database operations (Create, Drop, List)
- Table operations (Create, Drop, List, Schema)
- Row operations (Insert, Update, Delete, Select)
- Transaction support

#### 2. LMDB Storage Implementation (`pkg/storage/lmdb.go`)
Implements the storage interface using LMDB:
- **Database Management**: Each database is stored as a separate LMDB sub-database
- **Table Schema Storage**: Table schemas are serialized as JSON and stored with a `__schema__` prefix
- **Row Storage**: Rows are serialized as JSON with auto-generated keys
- **Concurrent Access**: Uses LMDB's built-in transaction system for thread safety

#### 3. CGO Bindings (`pkg/storage/lmdb_cgo.go`)
Provides the Go-to-C interface for LMDB operations using the `wellquite.org/golmdb` library.

#### 4. SQL Initialization System (`pkg/initializer/`)
- **SQL Runner**: Executes SQL scripts for database initialization
- **State Checking**: Verifies if database has been initialized
- **Script Processing**: Handles multi-statement SQL files with proper statement separation

## Data Storage Format

### Database Organization
```
LMDB Environment (./data/)
├── Database: "information_schema"
├── Database: "testdb"
│   ├── "__schema__users" → JSON schema for users table
│   ├── "__schema__products" → JSON schema for products table  
│   ├── "users:1" → JSON row data for user ID 1
│   ├── "users:2" → JSON row data for user ID 2
│   ├── "products:1" → JSON row data for product ID 1
│   └── ...
└── Database: "other_db"
    └── ...
```

### Schema Storage Format
Table schemas are stored as JSON with the key format `__schema__{table_name}`:
```json
{
  "columns": [
    {
      "name": "id",
      "type": "INT",
      "nullable": false,
      "primaryKey": true
    },
    {
      "name": "name", 
      "type": "VARCHAR(100)",
      "nullable": false,
      "primaryKey": false
    }
  ]
}
```

### Row Storage Format
Row data is stored as JSON arrays with keys like `{table_name}:{row_id}`:
```json
[1, "Alice", "alice@example.com", "2023-01-01 00:00:00"]
```

## Migration from Memory Storage

### Previous Implementation
- **In-Memory Storage**: Data stored in Go maps and slices
- **No Persistence**: Data lost on server restart
- **Hardcoded Sample Data**: Sample tables created programmatically

### New Implementation
- **LMDB Persistence**: Data stored in memory-mapped files
- **Full Persistence**: Data survives server restarts
- **SQL Initialization**: Sample data loaded from SQL scripts

### Key Changes Made

1. **Storage Backend Replacement**
   ```go
   // Before
   store := storage.NewMemoryStorage()
   
   // After  
   store, err := storage.NewLMDBStorage(dbPath, logger)
   ```

2. **Initialization System**
   ```go
   // Before
   database.CreateSampleTables()
   
   // After
   if !initializer.CheckInitialized(engine) {
       runner := initializer.NewSQLRunner(engine)
       err := runner.ExecuteScript("scripts/init.sql")
   }
   ```

3. **Database Provider Updates**
   - Added `CreateDatabase()` method to storage interface
   - Updated provider to handle database creation in storage layer
   - Removed hardcoded table creation methods

## Performance Characteristics

### LMDB Advantages
- **Memory-Mapped**: Zero-copy reads, excellent read performance
- **ACID Transactions**: Full transaction support with rollback
- **Copy-on-Write**: Multiple readers don't block writers
- **Crash Recovery**: Automatic recovery from system crashes
- **No Write-Ahead Log**: Simplified architecture

### Trade-offs
- **Write Performance**: Slower than in-memory for writes
- **Memory Usage**: Uses virtual memory for mapping
- **File Size**: Database files can grow larger than data size
- **Platform Dependency**: Requires LMDB library installation

## Error Handling

The integration includes comprehensive error handling:

1. **Environment Creation**: Handles LMDB environment setup failures
2. **Transaction Management**: Proper cleanup on transaction failures  
3. **Serialization Errors**: JSON encoding/decoding error handling
4. **Schema Validation**: Table schema consistency checks
5. **Graceful Shutdown**: Proper LMDB environment closure

## Concurrency Model

LMDB provides excellent concurrency characteristics:
- **Multiple Readers**: Unlimited concurrent read transactions
- **Single Writer**: One write transaction at a time (per environment)
- **Reader-Writer Isolation**: Readers see consistent snapshots
- **Deadlock Prevention**: LMDB's design prevents deadlocks

The Go implementation wraps LMDB transactions appropriately to maintain these guarantees while providing a clean interface to the SQL engine.

## Future Enhancements

Potential improvements for the LMDB integration:

1. **Indexing**: Add support for secondary indexes
2. **Compression**: Implement row-level compression
3. **Sharding**: Support for multiple LMDB environments
4. **Backup**: Add hot backup functionality
5. **Metrics**: Expose LMDB statistics and performance metrics
6. **Schema Evolution**: Handle table schema changes gracefully

## Troubleshooting

Common issues and solutions:

### Build Issues
- Ensure LMDB headers are in `lmdb-lib/include/`
- Verify LMDB library is in `lmdb-lib/lib/`
- Check CGO environment variables are set correctly

### Runtime Issues
- Verify data directory permissions
- Check available disk space for LMDB files
- Monitor LMDB environment size limits

### Performance Issues
- Adjust LMDB map size for large datasets
- Consider read-only transactions for queries
- Monitor transaction duration and cleanup