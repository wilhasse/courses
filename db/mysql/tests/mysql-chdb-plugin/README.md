# MySQL chDB UDF Plugin

A comprehensive project exploring different approaches to integrate ClickHouse (via chDB) with MySQL, from direct embedding to safe wrapper strategies. This project demonstrates the evolution of integrating a 722MB analytical engine into MySQL without crashing it.

## 🚀 Quick Summary

**Goal**: Query ClickHouse data from MySQL using data originally loaded from MySQL tables.

**Challenge**: libchdb.so is 722MB - too large to embed directly into MySQL.

**Solution**: A wrapper strategy using a helper program that runs in a separate process.

**Result**: ✅ Successfully query ClickHouse analytical data without crashing MySQL!

## 📚 Documentation Overview

This project evolved through multiple approaches, each documented step-by-step:

### Core Documentation
- **[WRAPPER_STRATEGY_EXPLAINED.md](WRAPPER_STRATEGY_EXPLAINED.md)** - 🌟 **Start Here!** Complete explanation of the working solution
- **[SUCCESS_SUMMARY.md](SUCCESS_SUMMARY.md)** - What works and how to use it
- **[WORKING_EXAMPLE.md](WORKING_EXAMPLE.md)** - Practical examples and code snippets

### Journey Documentation (Historical Context)
1. **[EMBEDDED_VS_EXTERNAL.md](EMBEDDED_VS_EXTERNAL.md)** - Why direct embedding failed
2. **[CRASH_SOLUTION.md](CRASH_SOLUTION.md)** - How we solved the MySQL crash problem
3. **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Technical analysis of the issues

### Setup Guides
- **[MANUAL_SETUP_STEPS.md](MANUAL_SETUP_STEPS.md)** - Manual installation instructions
- **[README_LIBCHDB.md](README_LIBCHDB.md)** - Using libchdb.so directly
- **[README_TVF_SETUP.md](README_TVF_SETUP.md)** - Table-valued function simulation

### Reference Documentation
- **[TVF_TEST_README.md](TVF_TEST_README.md)** - Detailed TVF simulation guide
- **[CLAUDE.md](CLAUDE.md)** - AI assistant context for this project

## 🎯 The Working Solution

```bash
# Query ClickHouse data (loaded from MySQL)
./chdb_query_helper "SELECT COUNT(*) FROM mysql_import.customers"
# Output: 10

# Complex analytics
./chdb_query_helper "SELECT AVG(age) FROM mysql_import.customers WHERE city = 'New York'"
# Output: 35.5
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

## 🏗️ Architecture

### What Didn't Work
```
MySQL → Load 722MB libchdb.so → 💥 CRASH!
```

### What Works
```
MySQL → Lightweight Plugin (90KB) → Helper Process → libchdb.so (722MB) → ClickHouse Data
```

## 🚀 Quick Start

### Prerequisites
- MySQL 8.0+ with development headers
- chDB built with: `cd /home/cslog/chdb && make buildlib`
- C++ compiler with C++11 support
- ClickHouse data from [mysql-to-chdb-example](../mysql-to-chdb-example)

### Build the Working Solution

```bash
# 1. Build the helper program
g++ -o chdb_query_helper chdb_query_helper.cpp -ldl -std=c++11

# 2. Test it directly
./chdb_query_helper "SELECT COUNT(*) FROM mysql_import.customers"
# Output: 10

# 3. (Optional) Build MySQL wrapper plugin
./build_wrapper_tvf.sh
```

### Usage Examples

#### Direct Usage (Recommended)
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

## 📈 Performance Considerations

- **Direct Embedding**: Would be fastest but crashes MySQL
- **Wrapper Process**: ~10-50ms overhead per query (acceptable for analytics)
- **Best Use Case**: Analytical queries, reporting, data exploration

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