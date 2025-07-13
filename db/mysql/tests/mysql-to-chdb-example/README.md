# MySQL to ClickHouse Data Transfer with Persistence

This project demonstrates how to extract data from MySQL and load it into ClickHouse using chdb, with data persistence between program executions.

## Architecture

The project is split into two separate executables:
- **feed_data**: Connects to MySQL, extracts data, and loads it into ClickHouse
- **query_data**: Reads the persisted ClickHouse data and runs analytical queries

## Files

- `common.h` - Shared structures and constants
- `feed_data.cpp` - MySQL data extraction and ClickHouse loading
- `query_data.cpp` - ClickHouse analytical queries
- `chdb_persist.h` - Enhanced chdb header with persistence support (demo)
- `setup_mysql.sql` - SQL script to create sample MySQL database
- `Makefile` - Build configuration

## Prerequisites

- MySQL server (user: root, password: teste)
- MySQL client libraries
- C++ compiler with C++17 support
- Sample database created using setup_mysql.sql

## Building

```bash
# Build both executables
make

# Or build individually
make feed_data
make query_data
```

## Running

### Step 1: Set up MySQL database
```bash
mysql -u root -pteste < setup_mysql.sql
```

### Step 2: Feed data from MySQL to ClickHouse
```bash
make run-feed
# Or: ./feed_data
```

This will:
- Connect to MySQL
- Extract customers and orders data
- Create ClickHouse database and tables
- Load data into ClickHouse
- Persist data to `./clickhouse_data`

### Step 3: Query the persisted data
```bash
make run-query
# Or: ./query_data
```

This will:
- Connect to the persisted ClickHouse data
- Verify data exists
- Run various analytical queries

### Run both steps
```bash
make run-all
```

## Data Persistence

Data is persisted to the `./clickhouse_data` directory. This allows:
- Running feed_data once to load data
- Running query_data multiple times without reloading
- Data survives program restarts

## Cleaning

```bash
# Clean build artifacts
make clean

# Clean persisted data
make clean-data

# Clean everything
make clean-all
```

## Sample Queries

The query_data program demonstrates:
1. Customer count by city
2. Top customers by revenue
3. Monthly order statistics
4. Top selling products
5. Customer age distribution
6. Recent orders
7. Customer lifetime value

## Notes

- This example uses a simplified chdb_persist.h for demonstration
- In production, use the actual chdb library with proper persistence
- The data path is configurable in common.h