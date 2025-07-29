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

The application consists of a single `main.go` file with the following key functions:

- **main()**: Orchestrates the connection and calls other functions in sequence
- **createDatabase()**: Creates `test_db` database and switches to it
- **createTable()**: Creates `example_batch` table in test_db with MergeTree engine
- **batchInsert()**: Inserts 1000 sample rows using batch operations
- **queryData()**: Performs aggregation queries and displays results

## Connection Details

The application connects to ClickHouse using:
- Host: `192.168.20.16`
- Port: `9000` (native protocol)
- Username: `root`
- Password: `test123`
- Database: Creates and uses `test_db`

## Key Dependencies

- `github.com/ClickHouse/clickhouse-go/v2`: Official ClickHouse Go driver for native protocol communication

## Development Notes

- All database operations use fully qualified table names (`test_db.example_batch`)
- The application uses LZ4 compression for data transfer
- Connection pooling is configured with max 5 connections
- Error handling uses wrapped errors with `fmt.Errorf`