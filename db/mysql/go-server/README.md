# MySQL Server with Virtual Database Support

A MySQL-compatible server implementation using go-mysql-server with support for virtual databases that proxy to remote MySQL servers. This server provides a complete MySQL protocol implementation with persistent storage using LMDB and the ability to create virtual databases that transparently forward queries to remote MySQL instances.

## Quick Start

```bash
# One-command setup, build, and run
make

# Or use the setup script
./setup.sh

# Connect with MySQL client
mysql -h localhost -P 3306 -u root
```

See [**QUICKSTART.md**](QUICKSTART.md) for detailed getting started guide.

## Documentation

### Virtual Database Feature
- 🚀 [**Quick Start Guide**](QUICKSTART.md) - Get running in 5 minutes
- 📖 [**Virtual Database User Guide**](docs/VIRTUAL_DATABASE_USER_GUIDE.md) - Complete guide for virtual databases
- 🔧 [**Remote Database Working Example**](docs/REMOTE_DATABASE_WORKING_EXAMPLE.md) - Real-world configuration example
- 📚 [**Remote Database Guide**](docs/REMOTE_DATABASE_GUIDE.md) - Technical implementation details

### General Documentation
- [**Complete Guide**](docs/GO_MYSQL_SERVER_GUIDE.md) - Comprehensive guide to understanding and using go-mysql-server
- [**Implementation Walkthrough**](docs/IMPLEMENTATION_WALKTHROUGH.md) - Detailed explanation of this project's implementation
- [**Quick Reference**](docs/QUICK_REFERENCE.md) - Quick reference for common patterns and interfaces
- [**SQL Execution Flow**](docs/sql_execution_flow.md) - How SQL queries are processed
- [**Storage Abstraction**](docs/storage_abstraction.md) - Storage layer design patterns

## Architecture

```
go-server/
├── main.go                      # Server entry point with debug support
├── pkg/
│   ├── provider/               # go-mysql-server integration layer
│   │   ├── database_provider.go   # Database management
│   │   ├── database.go            # Database implementation
│   │   ├── table.go               # Table operations
│   │   ├── remote_database.go     # Virtual database proxy
│   │   ├── remote_database_handler.go # Remote connection handling
│   │   └── session.go             # Session management
│   ├── storage/                # Storage backends
│   │   ├── storage.go             # Storage interface
│   │   ├── lmdb.go               # LMDB persistent storage
│   │   ├── lmdb_cgo.go           # CGO bindings for LMDB
│   │   └── memory.go              # In-memory storage
│   └── initializer/            # Database initialization
│       └── sql_runner.go          # SQL script execution
├── scripts/
│   ├── init.sql                # Default database schema
│   └── install-lmdb.sh         # LMDB library installer
└── lmdb-lib/                   # Local LMDB installation
    ├── include/                # C headers
    └── lib/                    # Compiled libraries
```

## Installation & Running

### Prerequisites
- Go 1.24+ (automatic toolchain download)
- C compiler (gcc/clang) for CGO support
- MySQL client for testing connections

### Build and Run

```bash
# Complete setup, build, and run
make

# Or step by step:
make deps          # Install dependencies
make build         # Build server binary
make run           # Run the server

# Run with debug mode
make run-debug     # Port 3306 with debug logging
make run-debug-port # Port 3311 to avoid conflicts
```

### Docker Support

```bash
# Production mode
docker-compose up mysql-server

# Development mode with debug tools
docker-compose up mysql-dev
```

## Key Components

### 1. Database Provider (`pkg/provider/database_provider.go`)
- Implements `sql.DatabaseProvider` and `sql.MutableDatabaseProvider`
- Manages multiple databases
- Handles CREATE/DROP DATABASE operations

### 2. Database (`pkg/provider/database.go`)
- Implements `sql.Database`, `sql.TableCreator`, `sql.TableDropper`
- Manages tables within a database
- Creates sample tables with data for testing

### 3. Table (`pkg/provider/table.go`)
- Implements `sql.Table` and `sql.UpdatableTable`
- Handles SELECT, INSERT, UPDATE, DELETE operations
- Uses partitioning system required by go-mysql-server

### 4. Storage Backend (`pkg/storage/`)
- Defines a clean interface for storage operations
- Includes in-memory implementation for demonstration
- Can be replaced with any storage system (files, databases, etc.)

## Features

### Core Features
- ✅ **MySQL Protocol Compatibility** - Works with any MySQL client
- ✅ **Persistent Storage** - LMDB backend for ACID transactions
- ✅ **Multiple Databases** - Create and manage multiple databases
- ✅ **Full SQL Support** - CREATE/DROP DATABASE/TABLE, INSERT/UPDATE/DELETE/SELECT
- ✅ **Schema Validation** - Type checking and constraint enforcement
- ✅ **Virtual Databases** - Proxy queries to remote MySQL servers
- ✅ **Auto-initialization** - SQL-based initialization with sample data
- ✅ **Cross-platform** - Linux, macOS, Windows support

### Virtual Database Features
- 🔗 **Remote Proxy** - Forward queries to remote MySQL instances
- 🏢 **Federation** - Query multiple remote databases from one interface
- 🔒 **Read-only Access** - Safe production database access
- 🚀 **Connection Pooling** - Efficient remote connection management
- 💾 **Schema Caching** - Performance optimization for remote tables

## Example Usage

Once connected, you can use standard MySQL commands:

```sql
-- Show databases
SHOW DATABASES;

-- Use the test database
USE testdb;

-- Show tables
SHOW TABLES;

-- Query sample data
SELECT * FROM users;
SELECT * FROM products WHERE price > 20;

-- Insert new data
INSERT INTO users (name, email, created_at) VALUES ('Charlie', 'charlie@example.com', NOW());

-- Update data
UPDATE products SET price = 25.99 WHERE name = 'Book';

-- Create new tables
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    product_id INT,
    quantity INT,
    total DECIMAL(10,2)
);

-- Create new database
CREATE DATABASE myapp;
USE myapp;

-- Drop tables/databases
DROP TABLE orders;
DROP DATABASE myapp;

-- Virtual Database Example
-- Create a proxy to remote MySQL server
CREATE DATABASE remote_prod__remote__192_168_1_100__3306__production__readonly__[PASSWORD];

-- Use virtual database (queries forwarded to remote)
USE remote_prod;
SELECT * FROM customers LIMIT 10;
```

## Extending the Example

### Adding More Storage Backends

Create new implementations of the `Storage` interface:

```go
// pkg/storage/postgres.go
type PostgresStorage struct {
    db *sql.DB
}

func (p *PostgresStorage) CreateTable(database, tableName string, schema sql.Schema) error {
    // Implement PostgreSQL table creation
}

// pkg/storage/s3.go
type S3Storage struct {
    bucket string
    client *s3.Client
}

func (s *S3Storage) InsertRow(database, tableName string, row sql.Row) error {
    // Implement S3-based row storage
}
```

### Adding Indexes

Implement `sql.IndexedTable` in your table:

```go
func (t *Table) GetIndexes(ctx *sql.Context) ([]sql.Index, error) {
    // Return available indexes
}

func (t *Table) CreateIndex(ctx *sql.Context, indexName string, using sql.IndexUsing, constraint sql.IndexConstraint, columns []sql.IndexColumn, comment string) error {
    // Create new index
}
```

### Adding Custom Functions

Add functions to your database provider:

```go
// Implement sql.FunctionProvider
func (p *DatabaseProvider) Function(ctx *sql.Context, name string) (sql.Function, bool) {
    switch strings.ToLower(name) {
    case "my_custom_function":
        return &MyCustomFunction{}, true
    }
    return nil, false
}
```

### Adding Authentication

Modify the session builder to add authentication:

```go
func NewSessionFactory() func(ctx context.Context, conn *mysql.Conn, addr string) (sql.Session, error) {
    return func(ctx context.Context, conn *mysql.Conn, addr string) (sql.Session, error) {
        // Validate user credentials
        if !validateUser(conn.User, conn.Password) {
            return nil, sql.ErrDatabaseAccessDeniedForUser.New(conn.User, addr)
        }
        
        session := sql.NewBaseSession()
        session.SetClient(sql.Client{
            User:         conn.User,
            Address:      addr,
            Capabilities: 0,
        })
        
        return session, nil
    }
}
```

## Performance Considerations

1. **Indexing**: Implement indexes for better query performance
2. **Caching**: Add caching layers for frequently accessed data
3. **Connection Pooling**: Configure appropriate connection limits
4. **Partitioning**: Use table partitioning for large datasets
5. **Batch Operations**: Optimize bulk inserts/updates

## Production Deployment

For production use, consider:

1. **Security**: Add proper authentication and authorization
2. **Monitoring**: Add metrics and health checks
3. **Configuration**: Make settings configurable via files/env vars
4. **Logging**: Add structured logging with different levels
5. **Graceful Shutdown**: Handle shutdown signals properly
6. **TLS**: Enable encrypted connections
7. **Backup/Recovery**: Implement data backup strategies

## Configuration

### Environment Variables
```bash
# Server configuration
export PORT=3306              # Server port (default: 3306)
export BIND_ADDR=0.0.0.0     # Bind address (default: localhost)
export DEBUG=true             # Enable debug logging
export VERBOSE=true           # Enable verbose logging

# Required for LMDB
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH
```

### Command Line Flags
```bash
./bin/mysql-server --port 3306 --bind 0.0.0.0 --debug
```

## Development

### Testing
```bash
# Run tests
make test

# Test connection
make test-connection

# Clean build artifacts
make clean
```

### Debug Mode
The server includes comprehensive debug logging:
```bash
# Run with debug tracing
make run-debug

# View execution flow
tail -f server.log
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [go-mysql-server](https://github.com/dolthub/go-mysql-server) - MySQL protocol implementation
- [LMDB](https://symas.com/lmdb/) - Lightning Memory-Mapped Database
- [Vitess](https://vitess.io/) - MySQL protocol parsing
