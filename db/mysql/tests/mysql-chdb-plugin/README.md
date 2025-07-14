# MySQL chDB UDF Plugin

A comprehensive project exploring different approaches to integrate ClickHouse (via chDB) with MySQL, from direct embedding to API server architecture. This project demonstrates the evolution of integrating a 722MB analytical engine into MySQL without crashing it.

## 🚀 Quick Summary

**Goal**: Query ClickHouse data from MySQL using data originally loaded from MySQL tables.

**Challenge**: libchdb.so is 722MB - too large to embed directly into MySQL.

**Solution**: API server that loads chDB once and serves queries via socket connection.

**Result**: ✅ Successfully query ClickHouse data from MySQL with millisecond response times!

## 📚 Documentation Overview

This project evolved through multiple approaches, each documented step-by-step:

### ⭐ Final Solution Documentation
- **[docs/COMPLETE_INTEGRATION_GUIDE.md](docs/COMPLETE_INTEGRATION_GUIDE.md)** - 🌟 **Start Here!** Complete guide to the API server solution
- **[docs/API_UDF_GUIDE.md](docs/API_UDF_GUIDE.md)** - Using MySQL UDF with API server
- **[docs/JSON_TABLE_GUIDE.md](docs/JSON_TABLE_GUIDE.md)** - ⭐ **True table-valued functions (MySQL 8.0.19+)**
- **[docs/TABLE_FUNCTION_GUIDE.md](docs/TABLE_FUNCTION_GUIDE.md)** - Table simulation with recursive CTEs (older MySQL)

### Core Documentation (Wrapper Approach)
- **[WRAPPER_STRATEGY_EXPLAINED.md](WRAPPER_STRATEGY_EXPLAINED.md)** - External helper process approach
- **[SUCCESS_SUMMARY.md](SUCCESS_SUMMARY.md)** - What works and how to use it
- **[WORKING_EXAMPLE.md](WORKING_EXAMPLE.md)** - Practical examples and code snippets

### Journey Documentation (Historical Context)
1. **[EMBEDDED_VS_EXTERNAL.md](EMBEDDED_VS_EXTERNAL.md)** - Why direct embedding failed
2. **[CRASH_SOLUTION.md](CRASH_SOLUTION.md)** - How we solved the MySQL crash problem
3. **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Technical analysis of the issues
4. **[docs/api-server-approach.md](docs/api-server-approach.md)** - API server architecture details

### Setup Guides
- **[MANUAL_SETUP_STEPS.md](MANUAL_SETUP_STEPS.md)** - Manual installation instructions
- **[README_LIBCHDB.md](README_LIBCHDB.md)** - Using libchdb.so directly
- **[README_TVF_SETUP.md](README_TVF_SETUP.md)** - Table-valued function simulation

### Reference Documentation
- **[TVF_TEST_README.md](TVF_TEST_README.md)** - Detailed TVF simulation guide
- **[CLAUDE.md](CLAUDE.md)** - AI assistant context for this project

## 🎯 The Working Solution - API Server Approach

### From MySQL:
```sql
-- Simple query
SELECT CAST(chdb_query('SELECT COUNT(*) FROM mysql_import.customers') AS CHAR);
-- Output: 10

-- Analytics query  
SELECT CAST(chdb_query('
    SELECT city, COUNT(*) as cnt 
    FROM mysql_import.customers 
    GROUP BY city 
    ORDER BY cnt DESC
') AS CHAR) AS city_stats;
```

### Alternative: Direct Helper Usage
```bash
# Query ClickHouse data directly
./chdb_query_helper "SELECT COUNT(*) FROM mysql_import.customers"
# Output: 10
```

## 📖 Step-by-Step Journey

### Step 1: Initial Attempt - Direct Embedding
**Approach**: Load libchdb.so directly into MySQL process  
**Result**: 💥 MySQL crashed!  
**Learning**: 722MB is too large for MySQL plugins  
**Documentation**: [EMBEDDED_VS_EXTERNAL.md](EMBEDDED_VS_EXTERNAL.md)

### Step 2: Understanding the Problem
**Discovery**: libchdb.so contains entire ClickHouse engine  
**Issue**: Symbol conflicts, memory issues, threading conflicts  
**Documentation**: [CRASH_SOLUTION.md](CRASH_SOLUTION.md)

### Step 3: The Wrapper Solution
**Approach**: Lightweight MySQL plugin + external helper program  
**Result**: ✅ Success! Queries work without crashes  
**Documentation**: [WRAPPER_STRATEGY_EXPLAINED.md](WRAPPER_STRATEGY_EXPLAINED.md)

### Step 4: Integration Testing
**MySQL UDF**: Partial success (security restrictions)  
**Direct Usage**: Perfect! Helper program works standalone  
**Documentation**: [SUCCESS_SUMMARY.md](SUCCESS_SUMMARY.md)

### Step 5: API Server Solution (Final)
**Approach**: Persistent server loads chDB once, MySQL connects via socket  
**Result**: ✅ Perfect! Fast queries from MySQL without crashes  
**Documentation**: [docs/COMPLETE_INTEGRATION_GUIDE.md](docs/COMPLETE_INTEGRATION_GUIDE.md)

## 🏗️ Architecture

### What Didn't Work
```
MySQL → Load 722MB libchdb.so → 💥 CRASH!
```

### What Works (Wrapper Approach)
```
MySQL → Lightweight Plugin (90KB) → Helper Process → libchdb.so (722MB) → ClickHouse Data
```

### What Works Best (API Server)
```
MySQL → UDF Functions → Socket → API Server (with 722MB libchdb.so loaded once) → ClickHouse Data
         ↓                         ↑
         └─── Binary Protocol ─────┘
              [4 bytes][data]
```

## 🚀 Quick Start

### Prerequisites
- MySQL 8.0+ with development headers
- chDB built with: `cd /home/cslog/chdb && make buildlib`
- C++ compiler with C++11 support
- ClickHouse data from [mysql-to-chdb-example](../mysql-to-chdb-example)

### Option 1: API Server Solution (Recommended)

```bash
# 1. Start the API server (in mysql-to-chdb-example directory)
cd ../mysql-to-chdb-example
./chdb_api_server_simple

# 2. Build and install MySQL UDF (in another terminal)
cd ../mysql-chdb-plugin
./scripts/build_api_udf.sh
sudo cp build/chdb_api_functions.so /usr/lib/mysql/plugin/
mysql -u root -pteste < scripts/install_api_udf.sql

# 3. Test from MySQL
mysql -u root -pteste -e "SELECT CAST(chdb_query('SELECT COUNT(*) FROM mysql_import.customers') AS CHAR)"
```

### Option 2: Direct Helper Program

```bash
# Build the helper program
g++ -o chdb_query_helper chdb_query_helper.cpp -ldl -std=c++11

# Test it directly
./chdb_query_helper "SELECT COUNT(*) FROM mysql_import.customers"
# Output: 10
```

### Usage Examples

#### MySQL UDF Usage (API Server)
```sql
-- Count customers
SELECT chdb_count('mysql_import.customers');

-- Complex query
SELECT CAST(chdb_query('
    SELECT city, COUNT(*) as customers, AVG(age) as avg_age 
    FROM mysql_import.customers 
    GROUP BY city
') AS CHAR);

-- Join with MySQL data
SELECT 
    m.product_name,
    CAST(chdb_query(CONCAT('
        SELECT SUM(quantity) 
        FROM mysql_import.orders 
        WHERE product_id = ', m.id
    )) AS UNSIGNED) AS total_sold
FROM mysql_products m;

-- NEW: True table-valued functions (MySQL 8.0.19+)
SELECT c.*, m.category 
FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        city VARCHAR(100) PATH '$.city'
    )
) AS c
JOIN mysql_customer_categories m ON c.city = m.city;

-- Alternative: CTE simulation (older MySQL)
WITH RECURSIVE nums AS (
    SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums 
    WHERE n < chdb_table_row_count('mysql_import.customers')
),
clickhouse_customers AS (
    SELECT 
        chdb_customers_get_id(n) AS id,
        chdb_customers_get_name(n) AS name,
        chdb_customers_get_city(n) AS city
    FROM nums
)
SELECT c.*, m.category 
FROM clickhouse_customers c
JOIN mysql_customer_categories m ON c.city = m.city;
```

#### Direct Helper Usage
```bash
# Simple query
./chdb_query_helper "SELECT COUNT(*) FROM mysql_import.customers"

# Analytics query
./chdb_query_helper "SELECT city, COUNT(*) as cnt FROM mysql_import.customers GROUP BY city"
```

#### Python Integration
```python
import subprocess

def query_chdb(sql):
    result = subprocess.run(['./chdb_query_helper', sql], 
                          capture_output=True, text=True)
    return result.stdout.strip()

count = query_chdb("SELECT COUNT(*) FROM mysql_import.customers")
print(f"Total customers: {count}")
```

#### Shell Script Integration
```bash
#!/bin/bash
customers=$(./chdb_query_helper "SELECT COUNT(*) FROM mysql_import.customers")
echo "We have $customers customers"
```

## 📊 Complete Data Flow

```mermaid
graph LR
    A[MySQL Tables] -->|mysql-to-chdb-example| B[ClickHouse Data]
    B -->|chdb_query_helper| C[Query Results]
    C --> D[Your Application]
```

1. **Data Loading**: Use `mysql-to-chdb-example/feed_data_v2` to load MySQL data into ClickHouse format
2. **Data Querying**: Use `chdb_query_helper` to execute analytical queries
3. **Integration**: Use results in any application or language

## 🔧 Project Structure

```
mysql-chdb-plugin/
├── chdb_query_helper.cpp      # The working solution!
├── src/
│   ├── chdb_tvf_embedded.cpp  # Direct embedding attempt (crashes MySQL)
│   ├── chdb_tvf_wrapper.cpp   # Wrapper approach (safe but limited by MySQL)
│   └── test_tvf_plugin.cpp    # Table-valued function simulation
├── scripts/
│   ├── build_wrapper_tvf.sh   # Build the wrapper version
│   └── ...
└── docs/                       # All the documentation files
```

## 🛠️ Troubleshooting

### If MySQL crashes
```bash
# Restart MySQL
sudo systemctl restart mysql

# Clean up problematic functions
mysql -u root -p < clean_mysql_functions.sql
```

### If helper returns empty results
```bash
# Check if ClickHouse data exists
ls -la /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data/
```

### If libchdb.so not found
```bash
# Build chDB
cd /home/cslog/chdb
make buildlib
```

## 📈 Performance Comparison

| Approach | Load Time | Query Time | Status |
|----------|-----------|------------|---------|
| Direct Embedding | N/A | N/A | 💥 Crashes MySQL |
| Binary Execution | 2-3s per query | 10ms | ❌ Too slow |
| Wrapper Process | 2-3s per query | 10ms | ⚠️ Works but slow |
| **API Server** | 3s once | **5-50ms** | ✅ **Best solution!** |

### Why API Server is Best
- Loads 722MB library only once
- Serves unlimited queries with millisecond latency
- No MySQL crashes or memory issues
- Can handle concurrent connections

## 🎓 Lessons Learned

1. **Don't force large libraries into MySQL** - 722MB is too much
2. **Process isolation is your friend** - Separate processes prevent crashes
3. **Simple solutions often work best** - A helper program solved everything
4. **MySQL UDF limitations** - Security restrictions limit external execution

## 🤝 Contributing

This project demonstrates various integration approaches. Contributions welcome for:
- Alternative integration methods
- Performance optimizations
- Additional language bindings
- Better error handling

## 📄 License

This project is for educational purposes, demonstrating MySQL-ClickHouse integration techniques.

## 🙏 Acknowledgments

- chDB project for the embedded ClickHouse engine
- MySQL/Percona for the extensible UDF system
- The journey of failed attempts that led to the working solution!

---

**Remember**: Sometimes the best solution isn't the most elegant - it's the one that actually works! 🚀