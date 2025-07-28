# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MySQL-compatible server implementation using go-mysql-server library with a custom storage backend. The server provides full MySQL protocol compatibility and can be connected to using standard MySQL clients.

## Build and Development Commands

**🚀 One-Command Quick Start:**
```bash
make              # Complete setup, build, and run (recommended)
./setup.sh        # Universal setup script (all platforms)
```

**🐳 Docker (Zero Dependencies):**
```bash
docker-compose up mysql-server    # Production mode
docker-compose up mysql-dev       # Development mode with debug tools
```

**🔨 Build Commands:**
```bash
make build        # Build main server binary to bin/mysql-server
make build-all    # Build all binaries (currently just main server)
```

**🚀 Run Commands:**
```bash
make run          # Run server directly with go run
make run-debug    # Run with integrated debug mode (recommended for development)
make run-debug-port # Run debug mode on port 3311 (avoid port conflicts)
make run-trace    # Run debug server with detailed execution tracing
make run-verbose  # Run with verbose logging
make start        # Build and run the binary
```

**🔍 Debug Modes:**
- **Integrated Debug Mode**: `make run-debug` - Main server with debug features built-in
- **Legacy Trace Alias**: `make run-trace` - Alias for run-debug (backward compatibility)
- **Environment Variables**: `DEBUG=true ./bin/mysql-server` or `VERBOSE=true ./bin/mysql-server`
- **Command Line Flags**: `./bin/mysql-server --debug --port 3311`

**🛠️ Development:**
```bash
make dev-setup    # Complete development environment setup
make test         # Run tests with go test ./...
make clean        # Clean build artifacts
make help         # Show all available commands
```

**🔌 Testing Connection:**
```bash
mysql -h localhost -P 3306 -u root
make test-connection    # Automated connection test
```

**✨ New Features**: Automatic LMDB installation, embedded CGO configuration, cross-platform support. No manual environment setup required!

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
- **LMDB persistent storage** - data survives server restarts
- SQL-based initialization system with sample data
- Graceful shutdown handling
- Configurable logging levels

## Development Notes

- Server runs on `localhost:3306` by default
- Uses logrus for structured logging
- No authentication required (connects as root)
- Debug server (`cmd/debug-server/main.go`) provides detailed execution tracing
- **LMDB storage backend** provides ACID transactions and persistence
- **CGO dependency** - requires LMDB C library integration

## Documentation

- **[Build Improvements Guide](docs/BUILD_IMPROVEMENTS.md)** - ⭐ **NEW**: Automated build process overview
- **[Build and Run Guide](docs/BUILD_AND_RUN.md)** - Complete build instructions and deployment
- **[LMDB Integration Guide](docs/LMDB_INTEGRATION.md)** - Detailed explanation of persistent storage implementation
- **[CGO Setup Guide](docs/CGO_SETUP.md)** - Environment configuration (now automated!)

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