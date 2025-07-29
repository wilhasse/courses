# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ClickHouse Go client example that demonstrates batch data ingestion using the native protocol. The application connects to a ClickHouse server, creates a test database and table, performs batch insertions, and queries the data.

## Commands

### Build and Run
```bash
# Download dependencies
go mod download

# Run the application
go run main.go

# Build executable
go build -o clickhouse-client main.go
```

### Testing
No test files currently exist in this project.

### Linting
No linting configuration present. To run standard Go linting:
```bash
gofmt -s -w .
go vet ./...
```

## Architecture

The application consists of:

### Main Files
- **main.go**: Core application with connection handling and database operations
- **config.yaml**: YAML configuration file for all settings

### Key Functions
- **loadConfig()**: Reads and parses YAML configuration
- **main()**: Orchestrates the connection and calls other functions in sequence  
- **createDatabase()**: Creates database and switches to it (names from config)
- **createTable()**: Creates table with MergeTree engine (names from config)
- **batchInsert()**: Inserts sample rows using batch operations (size from config)
- **queryData()**: Performs aggregation queries and displays results

### Configuration Structure
- **Config struct**: Nested struct matching YAML structure with proper tags
- **Sections**: clickhouse (connection), database (names), batch (settings)

## Connection Details

All connection details are read from `config.yaml`:
- Default: `192.168.20.16:9000` with credentials `root/test123`
- Configurable database names, connection pooling, timeouts, compression

## Key Dependencies

- `github.com/ClickHouse/clickhouse-go/v2`: Official ClickHouse Go driver for native protocol communication
- `gopkg.in/yaml.v3`: YAML configuration parsing

## Development Notes

- All database operations use fully qualified table names (`test_db.example_batch`)
- The application uses LZ4 compression for data transfer
- Connection pooling is configured with max 5 connections
- Error handling uses wrapped errors with `fmt.Errorf`