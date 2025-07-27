# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MySQL-compatible server implementation using go-mysql-server library with a custom storage backend. The server provides full MySQL protocol compatibility and can be connected to using standard MySQL clients.

## Build and Development Commands

**Quick Start:**
```bash
make run-trace    # Run debug server with detailed execution tracing
make run          # Run server directly with go run
make run-verbose  # Run with verbose logging (LOGLEVEL=debug)
```

**Build Commands:**
```bash
make build        # Build main server binary to bin/mysql-server
make build-debug  # Build debug server binary to bin/mysql-debug-server
make start        # Build and run the binary
```

**Development:**
```bash
make deps         # Download dependencies (go mod tidy + download)
make test         # Run tests with go test ./...
make clean        # Clean build artifacts
make dev-setup    # Set up development environment
```

**Testing Connection:**
```bash
mysql -h localhost -P 3306 -u root
```

## Architecture

The codebase follows a layered architecture:

1. **Server Layer** (`main.go`): MySQL protocol server using go-mysql-server
2. **Provider Layer** (`pkg/provider/`): go-mysql-server integration interfaces
3. **Storage Layer** (`pkg/storage/`): Pluggable storage backends

### Key Components

**DatabaseProvider** (`pkg/provider/database_provider.go`):
- Implements `sql.DatabaseProvider` and `sql.MutableDatabaseProvider`
- Manages multiple databases with CREATE/DROP operations
- Thread-safe with mutex protection
- Creates default "testdb" with sample data

**Storage Interface** (`pkg/storage/storage.go`):
- Clean abstraction for storage backends
- Supports database, table, and row operations
- Currently has in-memory implementation (`memory.go`)
- Designed for easy extension to other backends (PostgreSQL, S3, etc.)

**Table Implementation** (`pkg/provider/table.go`):
- Implements `sql.Table` and `sql.UpdatableTable`
- Handles SELECT, INSERT, UPDATE, DELETE operations
- Uses partitioning system required by go-mysql-server

## Key Features

- MySQL protocol compatibility (connects with mysql client, HeidiSQL, etc.)
- Multiple database support with CREATE/DROP DATABASE
- Full table operations: CREATE/DROP TABLE, INSERT/UPDATE/DELETE/SELECT
- Schema definition and validation
- Sample data for testing (users, products tables in testdb)
- Graceful shutdown handling
- Configurable logging levels

## Development Notes

- Server runs on `localhost:3306` by default
- Uses logrus for structured logging
- No authentication required (connects as root)
- Debug server (`cmd/debug-server/main.go`) provides detailed execution tracing
- Storage backend is pluggable - current implementation is in-memory only

## Testing

The project includes sample data in the default "testdb" database:
- `users` table with sample user records
- `products` table with sample product data

Test queries:
```sql
SHOW DATABASES;
USE testdb;
SHOW TABLES;
SELECT * FROM users;
SELECT * FROM products WHERE price > 20;
```