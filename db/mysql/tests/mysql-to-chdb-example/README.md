# MySQL to ClickHouse Data Transfer Example

This example demonstrates how to query a MySQL database and load the data into ClickHouse using chdb in C++.

## Prerequisites

- MySQL server with root access (password: teste)
- MySQL client libraries
- C++ compiler with C++17 support
- chdb library (assumed to be in ../chdb)

## Project Structure

- `main.cpp` - Main application code
- `setup_mysql.sql` - SQL script to create sample database and tables
- `chdb.h` - Simplified chdb header for demonstration
- `Makefile` - Build configuration
- `CMakeLists.txt` - Alternative CMake build configuration

## Features

1. **MySQL Connection**: Connects to local MySQL database
2. **Data Extraction**: Fetches customers and orders data from MySQL
3. **ClickHouse Integration**: Loads data into ClickHouse using chdb
4. **Sample Queries**: Demonstrates various analytical queries:
   - Customer count by city
   - Total revenue by customer
   - Average order value by month
   - Top selling products
   - Customer age distribution

## Building and Running

### Using Make:
```bash
make
make run
```

### Using CMake:
```bash
mkdir build
cd build
cmake ..
make
./mysql_to_chdb
```

## Sample Data

The project creates:
- **customers** table: 10 sample customers with demographics
- **orders** table: 15 sample orders with products and prices

## Note

This example includes a simplified `chdb.h` header for demonstration. In a real implementation, you would use the actual chdb library headers and link against the real chdb library.