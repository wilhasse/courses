# LMDB Go Example

This project demonstrates using LMDB (Lightning Memory-Mapped Database) with Go through the golmdb binding.

## Setup

### Quick Start (using Makefile)

```bash
# Setup LMDB and build the example
make

# Run the example
make run
```

### Manual Setup

1. First, download and build the LMDB C library:
   ```bash
   ./setup-lmdb.sh
   ```

2. Build the Go example:
   ```bash
   ./build.sh
   ```

3. Run the example:
   ```bash
   ./run.sh
   ```

### Using standard Go commands

To use standard `go build` or `go run` commands, you need to set CGO environment variables:

```bash
# Option 1: Source the .envrc file
source .envrc
go build
./lmdb-example

# Option 2: Use make targets
make go-build
make go-run

# Option 3: Set variables inline
CGO_CFLAGS="-I$(pwd)/lmdb-lib/include" \
CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb" \
go build
```

## Project Structure

- `setup-lmdb.sh` - Downloads LMDB from GitHub and builds the C library
- `build.sh` - Builds the Go program with proper CGO flags
- `main.go` - Example Go program demonstrating LMDB operations
- `lmdb-lib/` - Contains the downloaded LMDB C library (ignored by git)
- `testdb/` - LMDB database files created by the example (ignored by git)

## Dependencies

- Go 1.24.5 or later
- GCC (for building LMDB C library)
- golmdb binding: `wellquite.org/golmdb@latest`

## Example Operations

The example program demonstrates:
- Creating an LMDB environment
- Writing key-value pairs
- Reading values by key
- Iterating through all entries with a cursor
- Displaying database statistics