# MySQL Server Example using go-mysql-server

This is a complete example showing how to build a MySQL-compatible server using go-mysql-server with a custom storage backend.

## Architecture

```
mysql-server-example/
├── main.go                      # Server entry point
├── pkg/
│   ├── provider/               # go-mysql-server integration layer
│   │   ├── database_provider.go   # Implements sql.DatabaseProvider
│   │   ├── database.go            # Implements sql.Database
│   │   ├── table.go               # Implements sql.Table
│   │   └── session.go             # Session management
│   └── storage/                # Custom storage backend
│       ├── storage.go             # Storage interface
│       └── memory.go              # In-memory implementation
└── go.mod                      # Go module definition
```

## Running the Server

1. **Initialize the module:**
   ```bash
   cd mysql-server-example
   # go mod tidy
   make deps
   ```

2. **Start the server:**
   ```bash
   # go run main.go
   make run-trace
   ```

3. **Connect with MySQL client:**
   ```bash
   mysql -h localhost -P 3306 -u root
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

## Features Implemented

- ✅ MySQL protocol compatibility
- ✅ Multiple databases
- ✅ CREATE/DROP DATABASE
- ✅ CREATE/DROP TABLE
- ✅ INSERT/UPDATE/DELETE/SELECT
- ✅ Schema definition and validation
- ✅ Sample data for testing

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

This example provides a solid foundation for building MySQL-compatible servers with go-mysql-server and can be extended to support any storage backend or use case.
