# Documentation Overview

This directory contains comprehensive documentation for the MySQL-compatible server with LMDB persistent storage.

## Quick Start

**For Developers:**
1. Read [Build and Run Guide](BUILD_AND_RUN.md) for immediate setup
2. Use the automated build script: `./build.sh`
3. Connect with: `mysql -h 127.0.0.1 -P 3306 -u root`

**For System Administrators:**
1. Review [CGO Setup Guide](CGO_SETUP.md) for environment requirements
2. Follow [Build and Run Guide](BUILD_AND_RUN.md) deployment section
3. Read [LMDB Integration Guide](LMDB_INTEGRATION.md) for storage architecture

## Documentation Structure

### [Build and Run Guide](BUILD_AND_RUN.md) 📋
**Essential for getting started**
- Prerequisites and dependencies
- Build instructions (Make, Go, automated script)
- Environment setup and troubleshooting
- Development workflow and production deployment
- Data management and backup procedures

### [LMDB Integration Guide](LMDB_INTEGRATION.md) 🔧
**Technical deep-dive for developers**
- Storage architecture and design decisions
- Data format and serialization
- Migration from in-memory storage
- Performance characteristics and trade-offs
- Concurrency model and error handling

### [CGO Setup Guide](CGO_SETUP.md) ⚙️
**Required reading for build issues**
- What is CGO and why it's needed
- Environment variables explanation
- Platform-specific setup (Linux/macOS/Windows)
- Troubleshooting compilation and runtime issues
- Best practices for production deployment

## Common Workflows

### First-Time Setup
```bash
# 1. Clone repository and navigate to project
cd go-server/

# 2. Run automated build (handles CGO setup)
./build.sh

# 3. Start server
make run

# 4. Connect and test
mysql -h 127.0.0.1 -P 3306 -u root -e "SHOW DATABASES;"
```

### Development Cycle
```bash
# 1. Make code changes
vim pkg/storage/lmdb.go

# 2. Test with debug server
make run-trace

# 3. Run tests
make test

# 4. Build for production
make build
```

### Troubleshooting Workflow
1. **Build Issues**: Check [CGO Setup Guide](CGO_SETUP.md)
2. **Runtime Issues**: Check [Build and Run Guide](BUILD_AND_RUN.md) troubleshooting section
3. **Storage Issues**: Check [LMDB Integration Guide](LMDB_INTEGRATION.md) error handling section

## Key Concepts

### Storage Architecture
The project uses a layered storage architecture:
- **MySQL Protocol Layer**: Standard MySQL wire protocol
- **SQL Engine**: go-mysql-server provides SQL parsing and execution
- **Provider Layer**: Bridges SQL engine to storage backend
- **Storage Interface**: Clean abstraction for different backends
- **LMDB Backend**: Persistent key-value storage with ACID transactions

### CGO Integration
LMDB is a C library requiring CGO (C-Go) integration:
- **Headers**: C function declarations (`lmdb.h`)
- **Libraries**: Compiled C code (`liblmdb.a`, `liblmdb.so`)
- **Environment**: CGO flags tell Go where to find C dependencies
- **Runtime**: Shared libraries must be findable at execution time

### Data Persistence
- **Schema Storage**: Table schemas stored as JSON
- **Row Storage**: Row data serialized as JSON arrays
- **Key Format**: `{table}:{id}` for rows, `__schema__{table}` for schemas
- **Transactions**: LMDB provides ACID guarantees
- **Recovery**: Automatic crash recovery and data integrity

## Project Structure

```
go-server/
├── docs/                    # This documentation directory
│   ├── README.md           # This overview
│   ├── BUILD_AND_RUN.md    # Build and deployment guide
│   ├── LMDB_INTEGRATION.md # Storage architecture guide
│   └── CGO_SETUP.md        # CGO environment guide
├── lmdb-lib/               # LMDB C library files
│   ├── include/lmdb.h      # C headers
│   └── lib/lib*.{a,so}     # Compiled libraries
├── pkg/
│   ├── storage/            # Storage backends
│   ├── provider/           # Database provider layer
│   └── initializer/        # SQL initialization system
├── scripts/init.sql        # Database initialization SQL
├── build.sh               # Automated build script
├── Makefile              # Build automation
└── CLAUDE.md             # Project overview and commands
```

## Getting Help

### Documentation Priority
1. **Build Issues**: Start with [CGO Setup Guide](CGO_SETUP.md)
2. **Usage Questions**: Check [Build and Run Guide](BUILD_AND_RUN.md)  
3. **Architecture Questions**: Read [LMDB Integration Guide](LMDB_INTEGRATION.md)

### Self-Service Debugging
1. **Check Prerequisites**: Go version, LMDB library files
2. **Verify Environment**: CGO flags, library paths
3. **Test Components**: Build, run, connect separately
4. **Read Logs**: Server logs contain detailed error information

### Common Issues
- **`lmdb.h not found`**: CGO_CFLAGS not set correctly
- **`cannot find -llmdb`**: CGO_LDFLAGS not set correctly  
- **`liblmdb.so not found`**: LD_LIBRARY_PATH not set correctly
- **`permission denied`**: Data directory or library file permissions
- **`port in use`**: Another service using port 3306

## Contributing

When contributing to this project:
1. **Update Documentation**: Keep docs in sync with code changes
2. **Test Build Process**: Verify `build.sh` works on your platform
3. **Document New Features**: Add to appropriate guide
4. **Include Examples**: Provide working code samples

## Version History

- **v1.0**: Initial in-memory storage implementation
- **v2.0**: LMDB persistent storage integration
- **v2.1**: SQL initialization system
- **v2.2**: Comprehensive documentation and build automation