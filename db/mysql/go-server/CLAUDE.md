# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MySQL-compatible server implementation using go-mysql-server library with LMDB persistent storage and virtual database support. The server provides full MySQL protocol compatibility, persistent data storage, and the ability to create virtual databases that proxy queries to remote MySQL servers.

### Key Technologies
- **go-mysql-server**: MySQL protocol implementation and SQL engine
- **LMDB**: Lightning Memory-Mapped Database for persistent storage
- **chDB**: Embedded ClickHouse for analytical queries
- **CGO**: C bindings for LMDB integration
- **Virtual Databases**: Proxy functionality for remote MySQL servers
- **MySQL Passthrough**: Zero-overhead baseline for benchmarking

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

**📊 Benchmarking:**
```bash
./benchmark.sh         # Run performance benchmarks
./analyze_benchmark.py # Analyze benchmark results
```

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

**🌐 External Access:**
```bash
# Allow connections from all network interfaces
./bin/mysql-server --bind 0.0.0.0

# Or use environment variable
export BIND_ADDR=0.0.0.0
./bin/mysql-server

# Connect from another machine
mysql -h <server-ip> -P 3306 -u root
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

### Core Functionality
- MySQL protocol compatibility (connects with mysql client, HeidiSQL, DBeaver, etc.)
- Multiple database support with CREATE/DROP DATABASE
- Full table operations: CREATE/DROP TABLE, INSERT/UPDATE/DELETE/SELECT
- Schema definition and validation with type checking
- **Hybrid Storage System**:
  - **LMDB** for hot data (< 1M rows) - ACID transactions, fast point queries
  - **chDB** for analytical data (> 10M rows) - Columnar storage, 100x faster analytics
  - **Intelligent routing** based on table size and access patterns
- SQL-based initialization system with sample data
- Graceful shutdown handling with proper cleanup
- Configurable logging levels (debug, verbose, info)

### Virtual Database Features
- Create database proxies to remote MySQL servers
- Transparent query forwarding to remote instances
- Connection pooling for efficient resource usage
- Schema caching for improved performance
- Support for multiple simultaneous virtual databases
- Read-only access patterns for production safety

### Analytical Features (with chDB)
- 100-1000x faster aggregation queries
- Columnar data compression (10-50x)
- Parallel query execution
- Native support for window functions
- Efficient JOIN operations on large datasets

## Development Notes

### Server Configuration
- Default port: `3306` (configurable via `--port` flag or `PORT` env var)
- Default bind: `localhost` (use `--bind 0.0.0.0` for external access)
- No authentication required (connects as root with no password)
- Supports graceful shutdown on SIGINT/SIGTERM

### Logging and Debugging
- Uses logrus for structured logging
- Debug mode: `--debug` flag or `DEBUG=true` env var
- Verbose mode: `--verbose` flag or `VERBOSE=true` env var
- Integrated debug server with execution tracing
- Log files: `server.log`, `debug.log`

### Storage Implementation
- **Multiple storage backends** with different performance characteristics:
  - **MySQL Passthrough** - Direct forwarding to remote MySQL (baseline)
  - **LMDB** (default `./data/`) - For transactional data, dimension tables
  - **chDB** (default `./chdb_data/`) - For analytical data, fact tables
  - **Hybrid** - Intelligent routing based on table characteristics
- Configuration via `config.yaml` or command-line flags
- Automatic database initialization from `scripts/init.sql`
- **CGO dependency** - requires LMDB C library (auto-installed by setup)
- **chDB integration** - Optional, installed via `./install-chdb.sh`

## CGO Compilation Instructions

To compile this project with LMDB support, you must set the CGO environment variables:

```bash
# Set CGO flags to find LMDB headers and libraries
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib -llmdb"

# Then build normally
go build -o bin/mysql-server .

# Or use make (which sets these automatically)
make build
```

**Important**: The LMDB library is installed locally in `./lmdb-lib/` directory. The CGO flags must point to the absolute paths of:
- Headers: `./lmdb-lib/include/lmdb.h`
- Library: `./lmdb-lib/liblmdb.a`

## Documentation

- **[Build Improvements Guide](docs/BUILD_IMPROVEMENTS.md)** - ⭐ **NEW**: Automated build process overview
- **[Build and Run Guide](docs/BUILD_AND_RUN.md)** - Complete build instructions and deployment
- **[LMDB Integration Guide](docs/LMDB_INTEGRATION.md)** - Detailed explanation of persistent storage implementation
- **[chDB Integration Guide](docs/CHDB_INTEGRATION.md)** - ⭐ **NEW**: Analytical storage with embedded ClickHouse
- **[CGO Setup Guide](docs/CGO_SETUP.md)** - Environment configuration (now automated!)
- **[Benchmarking Guide](docs/BENCHMARKING_GUIDE.md)** - ⭐ **NEW**: Performance measurement and comparison

## Testing

### Sample Data
The project includes sample data in the default "testdb" database:
- `users` table with sample user records
- `products` table with sample product data

### Test Queries
```sql
-- Basic operations
SHOW DATABASES;
USE testdb;
SHOW TABLES;
SELECT * FROM users;
SELECT * FROM products WHERE price > 20;

-- Analytical queries (100x faster with chDB)
SELECT 
    category,
    COUNT(*) as product_count,
    AVG(price) as avg_price,
    MAX(price) as max_price
FROM products
GROUP BY category;

-- Virtual database testing
CREATE DATABASE test_remote__remote__localhost__3306__mysql__root__[PASSWORD];
USE test_remote;
SHOW TABLES;
```

### Running Tests
```bash
make test              # Run unit tests
make test-connection   # Test MySQL connection
```

## Common Tasks

### Adding New Features
1. Check existing patterns in `pkg/provider/` and `pkg/storage/`
2. Follow the layered architecture (provider -> storage)
3. Add appropriate debug logging for new functionality
4. Update tests and documentation

### Debugging Issues
1. Enable debug mode: `make run-debug`
2. Check log files: `tail -f server.log debug.log`
3. Use MySQL client for testing: `mysql -h localhost -P 3306 -u root`
4. For CGO issues, check: `ldd bin/mysql-server`

### Working with Virtual Databases
1. Format: `name__remote__host__port__database__user__password`
2. Replace dots with underscores in IP addresses
3. Replace @ with AT in passwords
4. Test connection first: `mysql -h <remote> -u <user> -p`

## Security Considerations

⚠️ **Important Security Notes:**
- Never commit real passwords to the repository
- Use environment variables or secure vaults for credentials
- Virtual database names contain passwords - limit access to database listing
- Consider using SSH tunneling for remote connections
- Always use read-only credentials for production access