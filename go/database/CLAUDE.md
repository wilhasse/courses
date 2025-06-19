# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a "Build Your Own Database" course codebase implementing a database from scratch in Go. The project is structured as incremental modules (01/ through 14/) representing different stages of database implementation, from basic file operations to a complete SQL-queryable database with B-tree storage.

## Architecture

### Module Progression
- **01/**: Basic file I/O operations and atomic writes
- **04/**: B-tree data structure implementation  
- **06-07/**: Key-value store with B-tree backend and free list management
- **08-10/**: Table structure and B-tree iterators
- **11-12/**: Transaction support with MVCC (Multi-Version Concurrency Control)
- **13/**: SQL parsing and query execution (CREATE TABLE, SELECT, INSERT, UPDATE, DELETE)
- **14/**: Final implementation

### Core Components

**B-tree Implementation (`btree.go`)**:
- Page-based storage with 256-byte pages
- Internal nodes store keys and child pointers
- Leaf nodes store key-value pairs
- Split/merge operations for tree balancing

**Key-Value Store (`kv.go`)**:
- Memory-mapped file I/O for performance
- Free list management for deleted pages
- MVCC for concurrent read/write access
- Transaction isolation with reader/writer locks

**Table Layer (`table.go`)**:
- Schema definition with typed columns (TYPE_BYTES, TYPE_INT64)
- Row encoding/decoding for storage
- Secondary indexes and constraints
- Query execution engine

**SQL Layer (`ql_*.go`)**:
- Recursive descent parser for SQL statements
- Abstract syntax tree (AST) representation
- Query planning and execution

## Development Commands

### Running Tests
```bash
# Test a specific module
cd [module-number]/ && go mod tidy && go test

# Working modules (tests pass):
cd 12/ && go mod tidy && go test

# Module 13 has known failing tests (index out of range in btree_iter.go:49)
```

### Dependencies
All modules use:
- Go 1.15+
- `github.com/stretchr/testify` for test assertions

### Building
Each module is independent with its own `go.mod`. Build using standard Go commands:
```bash
cd [module-number]/
go build
```

## Key Implementation Details

### Page Management
- Fixed 256-byte page size (`BTREE_PAGE_SIZE`)
- Memory-mapped files for efficient I/O
- Free list tracking for page recycling

### Concurrency Model
- Single writer, multiple readers
- Version-based MVCC
- Heap-based reader tracking for garbage collection

### Storage Format
Node format: `| type | nkeys | pointers | offsets | key-values |`
- Internal nodes: `| klen | key |`
- Leaf nodes: `| klen | vlen | key | val |`

### Testing Patterns
Tests use testify/require for assertions and create mock B-tree implementations with in-memory page storage for isolated testing.