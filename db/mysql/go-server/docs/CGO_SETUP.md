# CGO Environment Setup Guide

This guide explains the CGO (C-Go) environment setup required for building the MySQL server with LMDB integration.

## What is CGO?

CGO is a feature of Go that allows Go programs to call C libraries. Our project uses CGO to interface with the LMDB (Lightning Memory-Mapped Database) C library through the `wellquite.org/golmdb` Go wrapper.

## Why CGO is Required

### LMDB Library Structure
LMDB is a C library that provides:
- High-performance key-value storage
- ACID transactions
- Memory-mapped file access
- Cross-platform compatibility

### Go Integration Challenge
- Go cannot directly call C functions
- C libraries need to be linked at compile time
- Header files must be accessible during compilation
- Runtime library loading requires proper paths

## CGO Environment Variables

### Required Variables

#### `CGO_CFLAGS`
- **Purpose**: Tells the C compiler where to find header files
- **Value**: `-I$(pwd)/lmdb-lib/include`
- **What it does**: Points to the `lmdb.h` header file

#### `CGO_LDFLAGS` 
- **Purpose**: Tells the linker where to find libraries and which to link
- **Value**: `-L$(pwd)/lmdb-lib/lib -llmdb`
- **What it does**: 
  - `-L$(pwd)/lmdb-lib/lib`: Library search path
  - `-llmdb`: Link against `liblmdb.so` or `liblmdb.a`

#### `LD_LIBRARY_PATH` (Linux/macOS)
- **Purpose**: Tells the runtime where to find shared libraries
- **Value**: `$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH`
- **What it does**: Allows the program to find `liblmdb.so` at runtime

## Library Files Explained

### Header Files (`lmdb-lib/include/`)
```
lmdb-lib/include/
└── lmdb.h              # LMDB C API declarations
```

**Purpose**: Contains C function declarations and data structures that Go needs to understand the LMDB API.

**Example content**:
```c
// LMDB function declarations
int mdb_env_create(MDB_env **env);
int mdb_env_open(MDB_env *env, const char *path, unsigned int flags, mdb_mode_t mode);
// ... more functions
```

### Library Files (`lmdb-lib/lib/`)
```
lmdb-lib/lib/
├── liblmdb.a           # Static library (compiled LMDB code)
└── liblmdb.so          # Shared library (dynamic LMDB code)
```

**Static Library (`liblmdb.a`)**:
- Compiled LMDB code embedded into your binary
- No runtime dependencies
- Larger binary size
- Used when statically linking

**Shared Library (`liblmdb.so`)**:
- Separate file loaded at runtime
- Smaller binary size
- Requires library to be present on target system
- Used when dynamically linking

## Platform-Specific Setup

### Linux
```bash
# Set compilation flags
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"

# Set runtime library path
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"

# Build
go build main.go
```

### macOS
```bash
# Set compilation flags (same as Linux)
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"

# Set runtime library path (macOS uses DYLD_LIBRARY_PATH)
export DYLD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$DYLD_LIBRARY_PATH"

# Build
go build main.go
```

### Windows
```cmd
REM Set compilation flags
set CGO_CFLAGS=-I%CD%\lmdb-lib\include
set CGO_LDFLAGS=-L%CD%\lmdb-lib\lib -llmdb

REM Windows finds DLLs in PATH or current directory
set PATH=%CD%\lmdb-lib\lib;%PATH%

REM Build
go build main.go
```

## Automated Setup Scripts

### Shell Script (setup-env.sh)
```bash
#!/bin/bash
# Setup CGO environment for LMDB

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CGO_CFLAGS="-I${SCRIPT_DIR}/lmdb-lib/include"
export CGO_LDFLAGS="-L${SCRIPT_DIR}/lmdb-lib/lib -llmdb"

# Platform-specific library path
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${SCRIPT_DIR}/lmdb-lib/lib:$DYLD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="${SCRIPT_DIR}/lmdb-lib/lib:$LD_LIBRARY_PATH"
fi

echo "CGO environment configured for LMDB"
echo "CGO_CFLAGS: $CGO_CFLAGS"
echo "CGO_LDFLAGS: $CGO_LDFLAGS"
```

Usage:
```bash
source setup-env.sh
go run main.go
```

### Makefile Integration
```makefile
# Set CGO environment in Makefile
export CGO_CFLAGS := -I$(shell pwd)/lmdb-lib/include
export CGO_LDFLAGS := -L$(shell pwd)/lmdb-lib/lib -llmdb
export LD_LIBRARY_PATH := $(shell pwd)/lmdb-lib/lib:$(LD_LIBRARY_PATH)

build:
	go build -o bin/mysql-server main.go

run:
	go run main.go
```

## Troubleshooting CGO Issues

### Build-Time Issues

#### Error: `lmdb.h: No such file or directory`
**Cause**: CGO can't find the LMDB header file

**Solutions**:
```bash
# Check if header exists
ls -la lmdb-lib/include/lmdb.h

# Verify CGO_CFLAGS
echo $CGO_CFLAGS

# Fix path
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
```

#### Error: `cannot find -llmdb`
**Cause**: Linker can't find LMDB library

**Solutions**:
```bash
# Check if library exists
ls -la lmdb-lib/lib/liblmdb.*

# Verify CGO_LDFLAGS
echo $CGO_LDFLAGS

# Fix path
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"
```

#### Error: `undefined reference to 'mdb_*'`
**Cause**: Library not properly linked

**Solutions**:
```bash
# Ensure library is not corrupted
file lmdb-lib/lib/liblmdb.a

# Try static linking
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb -static"

# Or try shared linking explicitly
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb -shared"
```

### Runtime Issues

#### Error: `error while loading shared libraries: liblmdb.so`
**Cause**: Runtime can't find shared library

**Solutions**:
```bash
# Check if library exists
ls -la lmdb-lib/lib/liblmdb.so

# Set runtime path
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"

# Verify library path
ldd ./mysql-server | grep lmdb

# Alternative: Use static linking
go build -ldflags '-extldflags "-static"' main.go
```

### Debugging CGO

#### Verbose CGO Output
```bash
# Enable CGO debugging
export CGO_CFLAGS_ALLOW=".*"
export CGO_LDFLAGS_ALLOW=".*"
go build -x main.go 2>&1 | grep -i cgo
```

#### Check Linked Libraries
```bash
# Linux/macOS: Check shared library dependencies
ldd ./mysql-server

# macOS specific
otool -L ./mysql-server

# Check symbols
nm ./mysql-server | grep mdb_
```

## Best Practices

### Development Environment
1. **Use relative paths**: `$(pwd)` ensures portability
2. **Version control**: Don't commit environment variables
3. **Documentation**: Always document CGO requirements
4. **Testing**: Test on target platforms

### Production Deployment
1. **Static linking**: Consider static builds for deployment
2. **Library placement**: Place libraries in standard locations
3. **Container builds**: Use multi-stage Docker builds
4. **Dependency management**: Document exact library versions

### Cross-Compilation
```bash
# Cross-compile for different platforms
GOOS=linux GOARCH=amd64 CGO_ENABLED=1 go build main.go
GOOS=darwin GOARCH=amd64 CGO_ENABLED=1 go build main.go
GOOS=windows GOARCH=amd64 CGO_ENABLED=1 go build main.go
```

**Note**: Cross-compilation with CGO requires platform-specific libraries and toolchains.

## Alternative Approaches

### Using System LMDB
If LMDB is installed system-wide:
```bash
# Ubuntu/Debian
sudo apt-get install liblmdb-dev

# macOS with Homebrew  
brew install lmdb

# Use system libraries
export CGO_LDFLAGS="-llmdb"
# No need for CGO_CFLAGS if headers are in standard locations
```

### Using Go Modules with CGO
```go
// go.mod with build constraints
//go:build cgo
// +build cgo

package main

/*
#cgo CFLAGS: -I./lmdb-lib/include
#cgo LDFLAGS: -L./lmdb-lib/lib -llmdb
#include <lmdb.h>
*/
import "C"
```

This approach embeds CGO flags directly in the Go source code, reducing dependency on environment variables.