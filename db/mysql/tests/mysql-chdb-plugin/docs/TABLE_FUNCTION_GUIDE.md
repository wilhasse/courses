# ClickHouse Table Functions for MySQL

## Overview

Since MySQL doesn't support true table-valued functions, this guide shows how to simulate them using UDF functions and recursive CTEs. This allows you to join ClickHouse tables with MySQL tables as if they were native MySQL tables.

## Architecture

```
MySQL Query with JOIN
        ↓
Recursive CTE generates row numbers
        ↓
UDF functions fetch each row from ClickHouse
        ↓
Results joined with MySQL tables
```

## Available Functions

### Generic Table Functions

1. **chdb_table_row_count(table_name)**
   - Returns the number of rows in a ClickHouse table
   - Example: `SELECT chdb_table_row_count('mysql_import.customers')`

2. **chdb_table_get_field(table, field, row_num)**
   - Gets a specific field value from a specific row
   - Example: `SELECT chdb_table_get_field('mysql_import.customers', 'name', 1)`

3. **chdb_table_get_row(table, row_num, [format])**
   - Gets an entire row as a formatted string
   - Example: `SELECT chdb_table_get_row('mysql_import.customers', 1)`

### Specialized Customer Table Functions

For better performance with the customers table:

1. **chdb_customers_get_id(row_num)** - Get customer ID
2. **chdb_customers_get_name(row_num)** - Get customer name  
3. **chdb_customers_get_city(row_num)** - Get customer city

## Installation

```bash
# Build the functions
cd mysql-chdb-plugin
./scripts/build_table_functions.sh

# Install to MySQL
sudo cp build/chdb_table_functions.so /usr/lib/mysql/plugin/
mysql -u root -pteste < scripts/install_table_functions.sql
```

## Usage Examples

### Example 1: Basic Table Simulation

```sql
-- Generate a virtual table from ClickHouse data
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
),
clickhouse_customers AS (
    SELECT 
        chdb_customers_get_id(n) AS id,
        chdb_customers_get_name(n) AS name,
        chdb_customers_get_city(n) AS city
    FROM row_numbers
)
SELECT * FROM clickhouse_customers;
```

### Example 2: Join with MySQL Table

```sql
-- Create MySQL table with additional data
CREATE TABLE mysql_customer_categories (
    city VARCHAR(100),
    category VARCHAR(50),
    discount_rate DECIMAL(3,2)
);

INSERT INTO mysql_customer_categories VALUES 
    ('New York', 'Premium', 0.15),
    ('Los Angeles', 'Standard', 0.10),
    ('Chicago', 'Premium', 0.15);

-- Join ClickHouse customers with MySQL categories
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
),
clickhouse_customers AS (
    SELECT 
        chdb_customers_get_id(n) AS id,
        chdb_customers_get_name(n) AS name,
        chdb_customers_get_city(n) AS city
    FROM row_numbers
)
SELECT 
    cc.id,
    cc.name,
    cc.city,
    mc.category,
    mc.discount_rate
FROM clickhouse_customers cc
JOIN mysql_customer_categories mc ON cc.city = mc.city;
```

### Example 3: Create a View for Convenience

```sql
-- Create a view that looks like a regular table
CREATE VIEW v_clickhouse_customers AS
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
)
SELECT 
    chdb_customers_get_id(n) AS id,
    chdb_customers_get_name(n) AS name,
    chdb_customers_get_city(n) AS city
FROM row_numbers;

-- Now use it like a regular table
SELECT * FROM v_clickhouse_customers WHERE city = 'New York';

-- Join with MySQL tables easily
SELECT 
    v.name,
    v.city,
    m.category
FROM v_clickhouse_customers v
JOIN mysql_customer_categories m ON v.city = m.city;
```

### Example 4: Generic Table Access

```sql
-- Access any ClickHouse table dynamically
SELECT 
    chdb_table_get_field('mysql_import.orders', 'product_name', 1) AS first_product,
    chdb_table_get_field('mysql_import.orders', 'price', 1) AS first_price;

-- Get complete row data
SELECT chdb_table_get_row('mysql_import.customers', 5) AS fifth_customer;
```

## Performance Considerations

1. **Row-by-row Access**: Each field access is a separate query to the API server
2. **Caching**: Row counts are cached to avoid repeated queries
3. **Views**: Creating views improves query readability but not performance
4. **Batch Operations**: For large datasets, consider using `chdb_query()` instead

### Performance Comparison

| Method | Use Case | Performance |
|--------|----------|-------------|
| Table Functions | Small tables (<1000 rows) | Acceptable |
| Direct chdb_query() | Large tables or complex queries | Better |
| Cached Views | Repeated access to same data | Best |

## Advanced Usage

### Dynamic Table Generation

```sql
-- Create a stored procedure to generate any ClickHouse table
DELIMITER //
CREATE PROCEDURE generate_clickhouse_table(
    IN table_name VARCHAR(255),
    IN fields VARCHAR(1000)
)
BEGIN
    SET @sql = CONCAT('
        WITH RECURSIVE row_numbers AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM row_numbers 
            WHERE n < chdb_table_row_count(''', table_name, ''')
        )
        SELECT ', fields, ' FROM row_numbers'
    );
    
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END//
DELIMITER ;

-- Use it
CALL generate_clickhouse_table(
    'mysql_import.customers',
    'chdb_table_get_field(''mysql_import.customers'', ''id'', n) AS id,
     chdb_table_get_field(''mysql_import.customers'', ''name'', n) AS name'
);
```

### Materialized Tables

For better performance with frequently accessed data:

```sql
-- Create a materialized copy
CREATE TABLE materialized_customers AS
WITH RECURSIVE row_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM row_numbers 
    WHERE n < chdb_table_row_count('mysql_import.customers')
)
SELECT 
    chdb_customers_get_id(n) AS id,
    chdb_customers_get_name(n) AS name,
    chdb_customers_get_city(n) AS city
FROM row_numbers;

-- Add indexes for performance
ALTER TABLE materialized_customers ADD PRIMARY KEY (id);
ALTER TABLE materialized_customers ADD INDEX idx_city (city);

-- Now joins are fast
SELECT m.*, c.category
FROM materialized_customers m
JOIN mysql_customer_categories c ON m.city = c.city;
```

## Limitations

1. **Performance**: Each row requires a separate API call
2. **No Indexes**: Can't use ClickHouse indexes efficiently
3. **No Pushdown**: WHERE clauses aren't pushed to ClickHouse
4. **Recursive CTE Limit**: MySQL has a default limit of 1000 for CTEs

To increase CTE limit:
```sql
SET SESSION cte_max_recursion_depth = 10000;
```

## Best Practices

1. **Use Views**: Create views for frequently accessed ClickHouse tables
2. **Limit Rows**: Always use LIMIT when testing
3. **Cache Results**: Consider materialized tables for static data
4. **Batch Queries**: For complex analytics, use `chdb_query()` directly
5. **Monitor Performance**: Watch for slow queries with many rows

## Comparison with Direct Query

### Table Function Approach
```sql
-- Good for joins and row-by-row access
SELECT c.name, m.category
FROM v_clickhouse_customers c
JOIN mysql_categories m ON c.city = m.city;
```

### Direct Query Approach
```sql
-- Better for aggregations and complex queries
SELECT CAST(chdb_query('
    SELECT city, COUNT(*), AVG(age)
    FROM mysql_import.customers
    GROUP BY city
') AS CHAR);
```

Choose the approach based on your specific use case!