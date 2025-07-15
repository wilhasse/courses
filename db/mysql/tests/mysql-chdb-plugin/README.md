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

### 📦 Available Plugins

This project includes multiple MySQL UDF plugins for different approaches:

1. **`chdb_api_udf.cpp`** ⭐ - **Connects to the API server on port 8125**
   - Lightweight plugin that sends queries to the running API server
   - No heavy libchdb.so loading in MySQL
   - Functions: `chdb_api_query()`
   
2. **`chdb_api_json_udf.cpp`** ⭐ - **JSON format for table-like results**
   - Automatically adds FORMAT JSON to queries
   - Use with JSON_TABLE for proper columnar output
   - Functions: `chdb_api_query_json()`

3. **`chdb_api_ip_udf.cpp`** ⭐ - **IP-configurable server address**
   - Connect to any chDB API server, not just localhost
   - Functions: `chdb_api_query_remote(host:port, sql)`, `chdb_api_query_local(sql)`

4. **`chdb_api_ip_json_udf.cpp`** ⭐ - **IP-configurable + JSON format**
   - Combine configurable server with JSON output
   - Functions: `chdb_api_query_json_remote(host:port, sql)`, `chdb_api_query_json_local(sql)`
   
5. **`simple_chdb_udf.cpp`** - Executes chDB binary via subprocess
   - Simple approach but spawns new process for each query
   - Functions: `chdb_query()`
   
6. **`chdb_tvf_wrapper.cpp`** - Wrapper approach with external helper
   - Uses external process to load libchdb.so
   - Functions: various TVF functions
   
7. **`chdb_api_functions.cpp`** - Extended API functions
   - Multiple convenience functions for API server
   - Functions: `chdb_query()`, `chdb_count()`, etc.

8. **`chdb_json_table_functions.cpp`** - JSON table functions for MySQL 8.0.19+
   - True table-valued functions using JSON_TABLE
   - Functions: `chdb_customers_json()`, etc.

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
./chdb_api_server_simple -d /chdb/data/
# Server will start on port 8125

# 2. Install MySQL UDF plugin (in another terminal)
cd ../mysql-chdb-plugin
./scripts/install_chdb_api.sh
# This will build and install the chdb_api_query function

# 3. Test from MySQL
mysql -u root -e "SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico') AS CHAR)"
```

#### Key Points About chdb_api_udf Plugin

- **Plugin**: `chdb_api_udf.cpp` - Lightweight plugin that connects to API server
- **Function**: `chdb_api_query(sql)` - Executes ClickHouse SQL via API
- **Port**: Connects to localhost:8125
- **Protocol**: Simple binary protocol (no protobuf needed)
- **Important**: Always use `CAST(... AS CHAR)` to convert binary output

#### Usage Examples

```sql
-- Basic queries
SELECT CAST(chdb_api_query('SELECT version()') AS CHAR);
SELECT CAST(chdb_api_query('SELECT 1 + 1') AS CHAR);
SELECT CAST(chdb_api_query('SELECT today()') AS CHAR);

-- Query your data
SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico') AS CHAR);

-- Complex analytics
SELECT CAST(chdb_api_query('
    SELECT 
        toYYYYMM(data) as month,
        COUNT(*) as records,
        AVG(valor) as avg_value
    FROM mysql_import.historico
    GROUP BY month
    ORDER BY month DESC
    LIMIT 10
') AS CHAR);

-- NEW: Get results as a proper table using JSON format
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT id_contr, seq, codigo 
        FROM mysql_import.historico 
        LIMIT 10
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq',
        codigo INT PATH '$.codigo'
    )
) AS jt;

-- With the 10MB default limit, you can now query larger datasets
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT DISTINCT id_contr 
        FROM mysql_import.historico 
        WHERE codigo = 22
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr'
    )
) AS jt;
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

### If you get "ERROR: Response too large"
This means your query result exceeds the current limit. Solutions:
```bash
# Option 1: Add LIMIT to your query
SELECT chdb_api_query_json('SELECT * FROM table LIMIT 1000');

# Option 2: Increase the result size limit
./scripts/rebuild_with_limit.sh 50  # Increase to 50MB

# Option 3: Use aggregation to reduce data
SELECT chdb_api_query_json('SELECT COUNT(*) FROM table GROUP BY column');
```

### If you get "Invalid JSON text" with JSON_TABLE
```sql
-- Check the raw output first
SELECT CAST(chdb_api_query_json('YOUR QUERY LIMIT 10') AS CHAR)\G

-- Common issues:
-- 1. Result too large (see above)
-- 2. Column names are case-sensitive (use lowercase)
-- 3. Empty result set
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

## ⚙️ Configuration

### Adjusting Result Size Limit

By default, the UDF functions have a 10MB result size limit (increased from the original 1MB). You can adjust this limit based on your needs:

```bash
# Set to 20MB
./scripts/rebuild_with_limit.sh 20

# Set to 50MB  
./scripts/rebuild_with_limit.sh 50

# Set to 100MB (use with caution)
./scripts/rebuild_with_limit.sh 100
```

**Recommendations:**
- **10MB** (default): Suitable for most queries
- **20-50MB**: For large analytical queries
- **100MB+**: Maximum recommended, may impact MySQL performance

**Note:** After changing the limit, all UDF functions will be rebuilt and reinstalled automatically.

### Connecting to Remote chDB API Servers

The IP-configurable UDF functions allow you to connect to chDB API servers on different machines:

```bash
# Install IP-configurable functions
./scripts/install_ip_udf.sh
```

**Available Functions:**
- `chdb_api_query_remote(host:port, sql)` - Query any server
- `chdb_api_query_local(sql)` - Query localhost:8125  
- `chdb_api_query_json_remote(host:port, sql)` - JSON format
- `chdb_api_query_json_local(sql)` - JSON format on localhost

**Usage Examples:**
```sql
-- Query remote server
SELECT CAST(chdb_api_query_remote('192.168.1.100:8125', 'SELECT COUNT(*) FROM table') AS CHAR);

-- Query with different port
SELECT CAST(chdb_api_query_remote('dbserver.local:9000', 'SELECT version()') AS CHAR);

-- Use JSON format for table results
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('192.168.1.100:8125', 
        'SELECT * FROM mysql_import.historico LIMIT 10'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq',
        codigo INT PATH '$.codigo'
    )
) AS jt;
```

**Benefits:**
- Same .so file works with any server
- No need to rebuild for different environments
- Can query multiple servers from one MySQL instance
- Supports hostnames and IP addresses

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