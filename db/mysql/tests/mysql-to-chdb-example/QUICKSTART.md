# Quick Start Guide

This guide gets you up and running with the MySQL to ClickHouse data transfer using chDB.

## Prerequisites Check

```bash
# Check MySQL is installed and running
mysql --version
mysql -u root -pteste -e "SELECT 1"

# Check build tools
g++ --version  # Need C++17 support
make --version
```

## Step-by-Step Instructions

### 1. Build chDB Library (One-time setup)

```bash
# Navigate to chDB directory
cd /home/cslog/chdb

# Build the library (this takes 30-60 minutes)
make build

# Verify library was created
ls -la libchdb.so
```

### 2. Set Up the Project

```bash
# Navigate to project directory
cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example

# Clean any previous builds
make clean-all

# Build the executables (using v2 API)
make feed_data_v2 query_data_v2
```

### 3. Create MySQL Sample Data

```bash
# Create database and tables with sample data
mysql -u root -pteste < setup_mysql.sql

# Verify data was created
mysql -u root -pteste sample_db -e "SELECT COUNT(*) FROM customers;"
# Should show: 10

mysql -u root -pteste sample_db -e "SELECT COUNT(*) FROM orders;"
# Should show: 15
```

### 4. Transfer Data to ClickHouse

```bash
# Run the data feeder
./feed_data_v2
```

Expected output:
```
Loaded chdb library from: /home/cslog/chdb/libchdb.so
Connected to MySQL successfully!

Fetching data from MySQL...
Fetched 10 customers
Fetched 15 orders

Loading data into ClickHouse...
Database created/verified
Customers table created
Orders table created
Inserted 10 customers
Inserted 15 orders
Final customer count: 10
Final order count: 15

Data feeding completed!
```

### 5. Query the Data

```bash
# Run analytical queries
./query_data_v2
```

Expected output includes:
- Customer count by city
- Top customers by revenue
- Monthly order statistics
- Pretty formatted customer list

## Common Commands

```bash
# Run both steps at once
make run-all-v2

# Clean everything and start fresh
make clean-all
make run-all-v2

# Just clean persisted data (keep binaries)
make clean-data

# Rebuild everything
make clean
make all
```

## Quick Troubleshooting

### Problem: "Failed to load libchdb.so"
```bash
# Make sure chDB is built
cd /home/cslog/chdb && make build
```

### Problem: "MySQL connection failed"
```bash
# Check MySQL is running
sudo systemctl status mysql

# Test connection
mysql -u root -pteste -e "SELECT 1"
```

### Problem: "No data in ClickHouse"
```bash
# Clean and retry
make clean-data
./feed_data_v2
./query_data_v2
```

## What's Next?

- Modify `setup_mysql.sql` to use your own data
- Edit queries in `query_data_v2.cpp` for your analysis
- Adjust connection settings in `common.h`
- Read `IMPLEMENTATION_GUIDE.md` for detailed documentation