# chDB API UDF Functions Guide

## Overview

This guide describes how to use MySQL UDF functions that connect to the chDB API server. This approach avoids loading the 722MB libchdb.so directly into MySQL, preventing crashes and improving performance.

## Architecture

```
┌─────────────────┐         Socket          ┌──────────────────┐
│   MySQL Server  │ ◄─────────────────────► │ chDB API Server  │
│                 │    (127.0.0.1:8125)     │                  │
│  ┌───────────┐  │                         │ ┌──────────────┐ │
│  │ UDF Funcs │  │                         │ │ libchdb.so   │ │
│  │           │  │                         │ │   (722MB)    │ │
│  └───────────┘  │                         │ └──────────────┘ │
└─────────────────┘                         └──────────────────┘
```

## Setup Instructions

### 1. Start the chDB API Server

First, ensure your ClickHouse data is loaded:
```bash
cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example
./feed_data_v2  # If not already done
```

Start the simple API server (no protobuf required):
```bash
cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example
make chdb_api_server_simple
./chdb_api_server_simple
```

The server will:
- Load libchdb.so once (722MB)
- Listen on port 8125
- Serve queries using simple binary protocol

### 2. Build the MySQL UDF Functions

```bash
cd /home/cslog/courses/db/mysql/tests/mysql-chdb-plugin
./scripts/build_api_udf.sh
```

### 3. Install the UDF Functions

```bash
# Copy the library to MySQL plugin directory
sudo cp build/chdb_api_functions.so /usr/lib/mysql/plugin/

# Register the functions in MySQL
mysql -u root  < scripts/install_api_udf.sql
```

## Available Functions

### chdb_query(query)
Execute any ClickHouse SQL query and return the result as a string.

```sql
-- Simple query
SELECT chdb_query('SELECT COUNT(*) FROM mysql_import.customers');

-- Complex query with formatting
SELECT chdb_query('
    SELECT city, COUNT(*) as cnt 
    FROM mysql_import.customers 
    GROUP BY city 
    ORDER BY cnt DESC
');
```

### chdb_count(table)
Get the row count of a ClickHouse table (convenience function).

```sql
SELECT chdb_count('mysql_import.customers');
-- Returns: 10
```

### chdb_sum(table, column)
Calculate the sum of a numeric column (convenience function).

```sql
SELECT chdb_sum('mysql_import.orders', 'price');
-- Returns: 12500.50
```

## Usage Examples

### Basic Queries

```sql
-- Get ClickHouse version
SELECT chdb_query('SELECT version()');

-- Count rows
SELECT chdb_count('mysql_import.customers');

-- Filter data
SELECT chdb_query('SELECT * FROM mysql_import.customers WHERE age > 30');
```

### Join with MySQL Tables

```sql
-- Create a MySQL table
CREATE TEMPORARY TABLE mysql_data (
    city VARCHAR(100),
    region VARCHAR(100)
);

INSERT INTO mysql_data VALUES 
    ('New York', 'East'),
    ('Los Angeles', 'West');

-- Join with ClickHouse data
SELECT 
    m.city,
    m.region,
    CAST(chdb_query(
        CONCAT('SELECT COUNT(*) FROM mysql_import.customers WHERE city = ''', 
               m.city, '''')
    ) AS UNSIGNED) AS customer_count
FROM mysql_data m;
```

### Analytical Queries

```sql
-- City statistics
SELECT chdb_query('
    SELECT 
        city,
        COUNT(*) as total_customers,
        AVG(age) as average_age,
        MIN(age) as youngest,
        MAX(age) as oldest
    FROM mysql_import.customers
    GROUP BY city
    ORDER BY total_customers DESC
');

-- Time-series analysis
SELECT chdb_query('
    SELECT 
        toDate(order_date) as date,
        COUNT(*) as orders,
        SUM(price * quantity) as revenue
    FROM mysql_import.orders
    GROUP BY date
    ORDER BY date
');
```

## Performance Considerations

1. **Connection Overhead**: Each UDF call creates a new socket connection. For bulk operations, consider batching queries.

2. **Result Size**: Results are limited by MySQL's `max_allowed_packet` setting.

3. **Timeout**: The UDF has a 30-second timeout for queries.

4. **Concurrency**: The API server handles multiple concurrent connections.

## Troubleshooting

### "Cannot connect to chDB API server"
- Ensure the API server is running: `ps aux | grep chdb_api_server`
- Check if port 8125 is available: `netstat -tlnp | grep 8125`
- Verify firewall settings

### Empty or NULL results
- Check the API server logs for errors
- Verify the table exists: `SELECT chdb_query('SHOW TABLES FROM mysql_import')`
- Check for SQL syntax errors

### Performance issues
- Monitor API server CPU/memory usage
- Consider adding result caching
- Use specific queries instead of `SELECT *`

## Advantages

1. **No MySQL crashes**: 722MB library stays in separate process
2. **Better performance**: Library loaded once, not per query
3. **Scalability**: API server can run on different machine
4. **Flexibility**: Easy to add caching, load balancing, etc.

## Limitations

1. **Network overhead**: Each query has socket connection overhead
2. **Text format**: Results returned as tab-separated strings
3. **No streaming**: Entire result must fit in memory

## Uninstalling

To remove the UDF functions:
```bash
mysql -u root  < scripts/uninstall_api_udf.sql
sudo rm /usr/lib/mysql/plugin/chdb_api_functions.so
```

## Future Enhancements

1. **Connection pooling**: Reuse socket connections
2. **Binary protocol**: More efficient data transfer
3. **Async queries**: Non-blocking query execution
4. **Result caching**: Cache frequent query results