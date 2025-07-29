# ClickHouse Go Client Example

A simple Go application demonstrating batch data ingestion into ClickHouse using the native protocol.

## Prerequisites

- Go 1.19 or higher
- Access to a ClickHouse server
- ClickHouse server running on `192.168.20.16:9000` (or update the connection details in `main.go`)

## Quick Start

1. Clone or download this example:
```bash
cd /path/to/your/project
```

2. Install dependencies:
```bash
go mod download
```

3. Run the example:
```bash
go run main.go
```

## What It Does

The application will:
1. Connect to ClickHouse server at `192.168.20.16:9000`
2. Create a new database called `test_db`
3. Create a table `example_batch` with columns: id, name, value, created_at
4. Insert 1000 sample rows in a single batch operation
5. Query and display aggregate statistics (count, min, max, avg)
6. Show the first 5 rows as a sample

## Expected Output

```
Connected to ClickHouse successfully!
Database 'test_db' created successfully!
Switched to database 'test_db'
Table created successfully in test_db!
Inserted 1000 rows successfully!

Query Results:
Total Rows: 1000
Min Value: 0.00
Max Value: 1228.77
Average Value: 614.38

First 5 rows:
ID	Name		Value		Created At
--	----		-----		----------
0	item_0	0.00		2025-07-29 15:55:20
1	item_1	1.23		2025-07-29 15:55:21
2	item_2	2.46		2025-07-29 15:55:22
3	item_3	3.69		2025-07-29 15:55:23
4	item_4	4.92		2025-07-29 15:55:24
```

## Configuration

To use with your own ClickHouse server, update the connection options in `main.go`:

```go
conn, err := clickhouse.Open(&clickhouse.Options{
    Addr: []string{"your-server:9000"},
    Auth: clickhouse.Auth{
        Database: "default",
        Username: "your-username",
        Password: "your-password",
    },
    // ... other options
})
```

## Features Demonstrated

- Native protocol connection with LZ4 compression
- Database and table creation
- Batch insert operations for efficient data loading
- Aggregate queries (COUNT, MIN, MAX, AVG)
- Row-by-row data retrieval
- Proper error handling and resource cleanup