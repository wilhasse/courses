# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is module 01 of a "Build Your Own Database" course, focusing on basic file I/O operations and atomic writes. This module implements fundamental file handling patterns that will be used throughout the database implementation.

## Architecture

### Core Components

**File Operations (`files.go`)**:
- `SaveData1()`: Basic file writing with fsync for durability
- `SaveData2()`: Atomic file writing using temporary files and rename operations
- Demonstrates the progression from simple writes to crash-safe atomic updates

**Logging Operations (`logs.go`)**:
- `LogCreate()`: Creates append-only log files
- `LogAppend()`: Appends entries with fsync for durability
- `LogRead()`: Reads all log entries into memory
- Foundation for write-ahead logging used in later modules

## Development Commands

### Running Tests
```bash
go test
```

### Building
```bash
go build
```

### Dependencies
```bash
go mod tidy
```

## Key Implementation Details

### Atomic Writes Pattern
The module demonstrates two approaches to file writing:
1. Direct writing (`SaveData1`) - simple but not crash-safe
2. Atomic writing (`SaveData2`) - uses temp file + rename for atomicity

### Durability Guarantees
All write operations use `fsync()` to ensure data reaches persistent storage before returning success.

### Module Context
This is the first module in a progressive database implementation course. The file I/O patterns established here (especially atomic writes) are fundamental to the crash-safety guarantees required in later database modules.