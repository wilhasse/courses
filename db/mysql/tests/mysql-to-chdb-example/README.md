# MySQL to ClickHouse Data Transfer with chDB

This project demonstrates how to extract data from MySQL and load it into ClickHouse using the real chDB library, with data persistence between program executions.

## Architecture

The project provides two API implementations:

### V2 API (Deprecated but Stable) - RECOMMENDED
- **feed_data_v2**: Uses `query_stable_v2` API to load data from MySQL to ClickHouse
- **query_data_v2**: Uses `query_stable_v2` API to query persisted data

### Modern API (Currently has connection issues)
- **feed_data**: Uses modern `chdb_connect` API (connection issues being investigated)
- **query_data**: Uses modern `chdb_query` API

## Files

- `common.h` - Shared structures and constants
- `feed_data_v2.cpp` - MySQL data extraction using v2 API (stable)
- `query_data_v2.cpp` - ClickHouse queries using v2 API (stable)
- `feed_data.cpp` - MySQL data extraction using modern API
- `query_data.cpp` - ClickHouse queries using modern API
- `setup_mysql.sql` - SQL script to create sample MySQL database
- `Makefile` - Build configuration

## Prerequisites

1. **MySQL server** (user: root, password: teste)
2. **MySQL client libraries**
3. **C++ compiler with C++17 support**
4. **chDB library** - Must be built first:
   ```bash
   cd /home/cslog/chdb
   make build
   ```
   This will create `libchdb.so` in the chdb directory.

## Building

```bash
# Build all versions
make

# Or build individually
make feed_data_v2 query_data_v2  # V2 API (recommended)
make feed_data query_data        # Modern API
```

## Running

### Using V2 API (Recommended)

#### Step 1: Set up MySQL database
```bash
mysql -u root -pteste < setup_mysql.sql
```

#### Step 2: Feed data from MySQL to ClickHouse
```bash
make run-feed-v2
# Or: ./feed_data_v2
```

#### Step 3: Query the persisted data
```bash
make run-query-v2
# Or: ./query_data_v2
```

#### Run both steps
```bash
make run-all-v2
```

### Using Modern API (Experimental)

```bash
make run-all  # This currently has connection issues
```

## Data Persistence

Data is persisted to the `./clickhouse_data` directory using chDB's persistence features. This allows:
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

Both query programs demonstrate:
1. Customer count by city
2. Top customers by revenue
3. Monthly order statistics
4. Customer demographics
5. Pretty formatted output

## Technical Details

### V2 API (query_stable_v2)
- Uses command-line style arguments
- More mature and stable
- Returns `local_result_v2` structure with error handling
- Suitable for production use

### Modern API (chdb_connect/chdb_query)
- Cleaner API design
- Currently investigating connection initialization issues
- Will be the preferred API once issues are resolved

## Troubleshooting

If you get "Failed to load libchdb.so":
1. Ensure chDB is built: `cd /home/cslog/chdb && make build`
2. Check that libchdb.so exists in the chdb directory
3. The programs will search for libchdb.so in multiple locations

## API Comparison

| Feature | V2 API | Modern API |
|---------|---------|------------|
| Function | `query_stable_v2()` | `chdb_connect()`, `chdb_query()` |
| Stability | Stable | Under development |
| Connection | Per-query | Persistent |
| Error Handling | `error_message` field | `chdb_result_error()` |
| Recommended | Yes (for now) | Future