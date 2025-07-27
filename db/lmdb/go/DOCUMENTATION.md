# LMDB Go Project Documentation

## Overview

This project demonstrates how to use LMDB (Lightning Memory-Mapped Database) in Go by:
- Downloading and building the LMDB C library from GitHub
- Using the `wellquite.org/golmdb` Go binding to interface with the C library
- Providing a complete example of database operations

## Architecture

```
┌─────────────────┐
│   Go Application│
│    (main.go)    │
└────────┬────────┘
         │
         │ Go Binding
         ▼
┌─────────────────┐
│     golmdb      │
│  (Go package)   │
└────────┬────────┘
         │
         │ CGO/FFI
         ▼
┌─────────────────┐
│   LMDB C Library│
│   (liblmdb.so)  │
└─────────────────┘
```

## Setup Process

### Prerequisites
- Go 1.24.5 or later
- GCC (for building LMDB C library)
- Make (optional, for using Makefile)

### Initial Setup

1. **LMDB C Library Download and Build**
   - The `setup-lmdb.sh` script clones LMDB from https://github.com/LMDB/lmdb
   - Builds the C library using the included Makefile
   - Creates local directories with headers and libraries:
     - `lmdb-lib/include/` - LMDB header files
     - `lmdb-lib/lib/` - Compiled libraries (liblmdb.a, liblmdb.so)

2. **Go Module and Dependencies**
   - Module: `github.com/cslog/lmdb-go-example`
   - Main dependency: `wellquite.org/golmdb` - Go bindings for LMDB
   - Additional dependency: `github.com/rs/zerolog` - Logging library required by golmdb

## Build Methods

### Method 1: Using Makefile (Recommended)
```bash
# Complete setup and build
make

# Run the example
make run

# Other targets
make clean      # Remove binary and test database
make distclean  # Also remove LMDB library
make test       # Run the example
```

### Method 2: Using Shell Scripts
```bash
# Setup LMDB C library
./setup-lmdb.sh

# Build with proper CGO flags
./build.sh

# Run with library path
./run.sh
```

### Method 3: Standard Go Commands
```bash
# Option A: Source environment variables
source .envrc
go build
./lmdb-example

# Option B: Use make wrappers
make go-build
make go-run

# Option C: Set flags manually
CGO_CFLAGS="-I$(pwd)/lmdb-lib/include" \
CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb" \
go build
```

## CGO Configuration

The project requires CGO to link with the LMDB C library. Three environment variables are crucial:

1. **CGO_CFLAGS**: Points to LMDB header files
   ```bash
   export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
   ```

2. **CGO_LDFLAGS**: Points to LMDB library files
   ```bash
   export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"
   ```

3. **LD_LIBRARY_PATH**: Runtime library path for dynamic linking
   ```bash
   export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"
   ```

## Example Program Features

The `main.go` example demonstrates:

1. **Database Initialization**
   - Creates LMDB environment with 1GB map size
   - Configures max readers and databases
   - Uses golmdb's actor-based transaction batching

2. **Write Operations**
   - Uses `client.Update()` for write transactions
   - Creates named database with `txn.DBRef("mydb", golmdb.Create)`
   - Stores key-value pairs with `txn.Put()`

3. **Read Operations**
   - Uses `client.View()` for read-only transactions
   - Retrieves values with `txn.Get()`
   - Iterates through all entries using cursors

4. **Cursor Operations**
   - Creates cursor with `txn.NewCursor()`
   - Positions at first entry with `cursor.First()`
   - Iterates with `cursor.Next()`

## Project Structure

```
lmdb/go/
├── main.go          # Example Go program
├── go.mod           # Go module definition
├── go.sum           # Go dependencies lock file
├── setup-lmdb.sh    # Script to download and build LMDB
├── build.sh         # Build script with CGO flags
├── run.sh           # Run script with library path
├── Makefile         # Build automation
├── .envrc           # Environment variables
├── .gitignore       # Git ignore rules
├── README.md        # Quick start guide
├── DOCUMENTATION.md # This file
└── lmdb-lib/        # LMDB C library (git-ignored)
    ├── include/     # Header files
    └── lib/         # Compiled libraries
```

## golmdb Binding Details

The golmdb binding provides:

1. **Actor-based Architecture**
   - Write transactions are batched for performance
   - Configurable batch size (default: 100)
   - Automatic transaction management

2. **Type-safe API**
   - `LMDBClient` - Main client interface
   - `ReadWriteTxn` - Write transaction handle
   - `ReadOnlyTxn` - Read-only transaction handle
   - `DBRef` - Database reference within environment

3. **Database Flags**
   - `golmdb.Create` - Create database if it doesn't exist
   - `golmdb.DupSort` - Allow duplicate keys
   - `golmdb.IntegerKey` - Use integer keys
   - Various sorting options

## Troubleshooting

### Build Errors

1. **"lmdb.h: No such file or directory"**
   - Ensure `setup-lmdb.sh` has been run
   - Check CGO_CFLAGS is set correctly
   - Verify lmdb-lib/include/lmdb.h exists

2. **"cannot open shared object file"**
   - Set LD_LIBRARY_PATH before running
   - Use `./run.sh` instead of direct execution
   - Consider static linking with liblmdb.a

3. **"invalid argument" when creating LMDB client**
   - Check parameter order matches golmdb.NewLMDB signature
   - Verify batch size parameter is included

### Runtime Issues

1. **Database locked errors**
   - Ensure previous instances are terminated
   - Check file permissions on database directory
   - Remove lock files if process crashed

2. **Performance considerations**
   - Adjust batch size for write-heavy workloads
   - Use NoReadAhead flag for large datasets
   - Consider memory mapping size vs available RAM

## Best Practices

1. **Always use defer for cleanup**
   ```go
   defer client.TerminateSync()
   defer cursor.Close()
   ```

2. **Handle errors properly in transactions**
   ```go
   err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
       // Return errors to trigger rollback
       return err
   })
   ```

3. **Use appropriate transaction types**
   - `View()` for read-only operations
   - `Update()` for write operations
   - Never write in View transactions

4. **Database naming**
   - Use meaningful names for databases
   - Remember LMDB supports multiple named databases
   - Default database uses empty string ""

## References

- LMDB Documentation: http://www.lmdb.tech/doc/
- golmdb Package: https://pkg.go.dev/wellquite.org/golmdb
- LMDB GitHub: https://github.com/LMDB/lmdb
- CGO Documentation: https://golang.org/cmd/cgo/