# Build and Run Guide

This guide explains how to build and run the MySQL-compatible server with LMDB persistent storage.

## Prerequisites

### System Requirements
- Go 1.24.0 or higher
- Linux/macOS/Windows (LMDB is cross-platform)
- Make (optional, for convenience commands)
- MySQL client (for testing connections)

### Dependencies
The project includes all necessary dependencies:
- **LMDB Library**: Pre-compiled libraries in `lmdb-lib/`
- **chDB Library**: Embedded ClickHouse for analytics (optional)
- **Go Dependencies**: Managed via `go.mod`
- **CGO**: Required for LMDB bindings

## Directory Structure

```
go-server/
├── lmdb-lib/                 # LMDB library files
│   ├── include/
│   │   └── lmdb.h           # LMDB header file
│   └── lib/
│       ├── liblmdb.a        # Static library
│       └── liblmdb.so       # Shared library
├── pkg/
│   ├── storage/
│   │   ├── lmdb.go          # LMDB storage implementation
│   │   ├── lmdb_cgo.go      # CGO bindings
│   │   ├── chdb_storage.go  # chDB storage implementation
│   │   ├── hybrid_storage.go # Intelligent storage routing
│   │   └── table_metadata.go # Table statistics tracking
│   ├── provider/            # Database provider layer
│   ├── config/              # Configuration management
│   └── initializer/         # SQL initialization system
├── scripts/
│   └── init.sql            # Database initialization script
├── cmd/
│   └── debug-server/       # Debug server with detailed logging
├── main.go                 # Main server entry point
├── Makefile               # Build automation
└── data/                  # LMDB data files (created at runtime)
```

## Building the Project

### Environment Setup

The project requires specific CGO environment variables to locate LMDB libraries:

```bash
# Set LMDB include path
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"

# Set LMDB library path
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"

# Set runtime library path
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"
```

### Build Methods

#### Method 1: Using Make (Recommended)
```bash
# Set up dependencies
make deps

# Build main server
make build

# Build debug server
make build-debug

# Build and run
make start
```

#### Method 2: Direct Go Commands
```bash
# Download dependencies
go mod tidy
go mod download

# Build with CGO environment
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"
go build -o bin/mysql-server main.go

# Or run directly
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"
go run main.go
```

#### Method 3: Using Build Script
```bash
# Make the build script executable
chmod +x build.sh

# Run the build script (sets environment automatically)
./build.sh
```

### Build Troubleshooting

#### Common Build Errors

**Error: `lmdb.h: No such file or directory`**
```bash
# Solution: Verify LMDB header path
ls -la lmdb-lib/include/lmdb.h
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
```

**Error: `cannot find -llmdb`**
```bash
# Solution: Verify LMDB library path
ls -la lmdb-lib/lib/liblmdb.*
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"
```

**Error: `error while loading shared libraries: liblmdb.so`**
```bash
# Solution: Set runtime library path
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"
```

## Installing chDB (Optional)

For analytical query support with chDB:

```bash
# Install chDB library
./install-chdb.sh

# The script automatically detects CPU capabilities
# and installs the appropriate version
```

## Running the Server

### Quick Start
```bash
# Start server with automatic environment setup
make run

# Or start with debug logging
make run-debug

# Or start with verbose logging
make run-verbose
```

### Storage Backend Options
```bash
# Run with hybrid storage (default - LMDB + chDB)
./bin/mysql-server --storage hybrid

# Run with LMDB only (transactional)
./bin/mysql-server --storage lmdb

# Run with chDB only (analytical)
./bin/mysql-server --storage chdb

# Configure storage thresholds
./bin/mysql-server \
  --hot-data-threshold 500000 \
  --analytical-threshold 5000000
```

### Manual Start
```bash
# Set environment variables
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"

# Run the server with options
go run main.go --storage hybrid --debug
```

### Server Output
```
time="2025-01-01T10:00:00Z" level=info msg="Database not initialized. Running initialization script..."
time="2025-01-01T10:00:00Z" level=info msg="Database initialization completed successfully"
time="2025-01-01T10:00:00Z" level=info msg="Starting MySQL server on 127.0.0.1:3306"
time="2025-01-01T10:00:00Z" level=info msg="Connect with: mysql -h 127.0.0.1 -P 3306 -u root"
time="2025-01-01T10:00:00Z" level=info msg="Server ready. Accepting connections."
```

## Connecting to the Server

### Using MySQL Client
```bash
# Connect to the server
mysql -h 127.0.0.1 -P 3306 -u root

# Or with explicit host/port
mysql -h localhost -P 3306 -u root
```

### Basic Commands
```sql
-- Show available databases
SHOW DATABASES;

-- Use the test database
USE testdb;

-- Show tables
SHOW TABLES;

-- Query sample data
SELECT * FROM users;
SELECT * FROM products WHERE price > 100;

-- Test joins
SELECT u.name, p.name, o.quantity 
FROM users u 
JOIN orders o ON u.id = o.user_id 
JOIN products p ON o.product_id = p.id;
```

## Development Workflow

### Running in Development Mode
```bash
# Terminal 1: Start server with auto-restart on changes
make run-verbose

# Terminal 2: Connect and test
mysql -h 127.0.0.1 -P 3306 -u root
```

### Debug Mode
```bash
# Start debug server with detailed query analysis
make run-trace

# Or run debug server directly
go run cmd/debug-server/main.go
```

### Testing Changes
```bash
# Run tests
make test

# Clean build artifacts
make clean

# Rebuild everything
make clean && make build
```

## Production Deployment

### Building for Production
```bash
# Build optimized binary
CGO_ENABLED=1 go build -ldflags="-s -w" -o mysql-server main.go

# Set library path for production
export LD_LIBRARY_PATH="/opt/mysql-server/lib:$LD_LIBRARY_PATH"
```

### Configuration Options

#### Environment Variables
- `LOGLEVEL`: Set to `debug`, `info`, `warn`, `error`
- `DATA_DIR`: LMDB data directory (default: `./data`)
- `MYSQL_PORT`: Server port (default: `3306`)
- `MYSQL_HOST`: Server host (default: `127.0.0.1`)

#### Example Production Start
```bash
export LOGLEVEL=info
export DATA_DIR=/var/lib/mysql-server/data
export LD_LIBRARY_PATH=/opt/mysql-server/lib:$LD_LIBRARY_PATH
./mysql-server
```

## Data Management

### Database Files
```bash
# LMDB creates these files in the data directory:
data/
├── data.mdb    # Main database file
└── lock.mdb    # Lock file for concurrent access
```

### Backup and Restore
```bash
# Backup (server must be stopped)
cp -r data/ backup-$(date +%Y%m%d)/

# Restore (server must be stopped)
rm -rf data/
cp -r backup-20250101/ data/
```

### Cleaning Data
```bash
# Stop server and remove all data
pkill -f "go run main.go"
rm -rf data/

# Restart server (will reinitialize with sample data)
make run
```

## Troubleshooting

### Common Runtime Issues

**Server won't start - Port in use**
```bash
# Check what's using port 3306
lsof -i :3306

# Kill existing MySQL processes
sudo pkill mysqld
```

**Permission denied on data directory**
```bash
# Fix data directory permissions
chmod 755 data/
```

**LMDB errors**
```bash
# Check disk space
df -h .

# Check LMDB file integrity
file data/data.mdb
```

### Log Analysis
```bash
# View server logs with timestamps
make run-verbose | tee server.log

# Monitor specific issues
grep -i error server.log
grep -i "failed" server.log
```

### Performance Monitoring
```bash
# Monitor server resources
top -p $(pgrep -f "go run main.go")

# Monitor database file sizes
watch -n 5 'ls -lh data/'

# Monitor connections
netstat -an | grep :3306
```