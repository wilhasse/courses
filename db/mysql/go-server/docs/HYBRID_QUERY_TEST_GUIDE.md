# Hybrid Query System - Test Guide

This guide explains how to test the hybrid query system using the provided test programs.

## Test Environment

- **Remote MySQL Server**: 10.1.0.7 (no password required)
- **Local Server**: localhost:3306
- **Test Database**: testdb
- **Test Tables**: 
  - `employees` (cached from remote)
  - `employee_notes` (remains on remote)

## Available Tests

### 1. Clean Demo Test (`test/clean_hybrid_demo.go`)

The simplest test that demonstrates the core functionality:

```bash
make test-hybrid-clean
```

**What it does:**
- Loads employees table from remote MySQL into LMDB
- Executes simple SELECT on cached data
- Performs JOIN between cached employees and remote employee_notes
- Shows clean, formatted output

**Expected Output:**
```
1. Caching employees table from remote MySQL server (10.1.0.7)
   ✓ Successfully cached 15 employees

2. Query cached employees table:
   Employee #1: John Smith
   Employee #2: Sarah Johnson
   Employee #3: Michael Williams

3. JOIN cached employees with remote employee_notes:
   Found 5 employee notes:
   - Employee #1 John Smith: Great performance this quarter
   - Employee #2 Sarah Johnson: Completed certification program
   ...
```

### 2. Working Test (`test/working_hybrid.go`)

More comprehensive test with debugging information:

```bash
make test-hybrid-working
```

**Features:**
- Shows query analysis details
- Displays join conditions
- Handles type conversion safely
- Tests multiple scenarios

### 3. Adaptive Test (`test/adaptive_hybrid.go`)

Discovers remote table schema and adapts:

```bash
make test-hybrid-adaptive
```

**Features:**
- Inspects remote table structure
- Creates appropriate test queries
- Handles different column types

### 4. Final Test (`test/hybrid_final.go`)

Complete demonstration with all features:

```bash
make test-hybrid-final
```

## Setup Instructions

### 1. Create Remote Test Data

First, ensure the remote MySQL server has the test tables:

```bash
# Check what's on the remote server
mysql -h 10.1.0.7 -u root testdb -e "SHOW TABLES;"

# Create employee_notes table if needed
mysql -h 10.1.0.7 -u root testdb < test/create_remote_test_tables.sql
```

The `create_remote_test_tables.sql` creates:
```sql
CREATE TABLE employee_notes (
    note_id INT PRIMARY KEY AUTO_INCREMENT,
    emp_id INT NOT NULL,
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data linking to employees
INSERT INTO employee_notes (emp_id, note) VALUES
(1, 'Great performance this quarter'),
(2, 'Completed certification program'),
...
```

### 2. Run Tests

```bash
# Quick test
make test-hybrid-clean

# All tests
make test-hybrid-working
make test-hybrid-adaptive
make test-hybrid-final
```

## Understanding the Output

### Query Analysis
```
Query Analysis:
- Has cached table: true
- Cached tables: [{testdb employees e}]
- Remote tables: [{testdb employee_notes n}]
- Join conditions: [{e id n emp_id =}]
- Requires rewrite: true
```

### Query Rewriting
```
Original query:
SELECT e.*, n.note FROM employees e JOIN employee_notes n ON e.id = n.emp_id

Rewritten for MySQL:
SELECT n.note, n.emp_id FROM employee_notes n
```

### Join Execution
```
Remote result columns: [note emp_id]
Cached result columns: [id first_name last_name ...]
Join produced 5 rows
```

## Common Issues and Solutions

### 1. Connection Refused
```
Failed to create hybrid handler: dial tcp [::1]:3306: connect: connection refused
```
**Solution**: Ensure local MySQL server is not running on port 3306, or use a different port.

### 2. LMDB Directory Not Found
```
Failed to create LMDB client: no such file or directory
```
**Solution**: The test creates the directory automatically. If it fails, create manually:
```bash
mkdir -p ./test_cache
```

### 3. Remote Table Not Found
```
Table 'testdb.employee_notes' doesn't exist
```
**Solution**: Run the setup script:
```bash
mysql -h 10.1.0.7 -u root testdb < test/create_remote_test_tables.sql
```

### 4. Cartesian Product
```
Join produced 75 rows  # Should be 5-15 rows
```
**Solution**: This was fixed by ensuring join columns are included in rewritten queries.

## Performance Testing

To see the performance benefits:

```bash
# Time a query directly to remote
time mysql -h 10.1.0.7 -u root testdb -e "SELECT * FROM employees" > /dev/null

# Time through hybrid system (after caching)
# The cached query should be significantly faster
```

## Debugging

Enable debug logging to see internal operations:

```go
logger := zerolog.New(os.Stdout).Level(zerolog.DebugLevel)
```

This shows:
- Table loading progress
- Query analysis details
- Rewritten queries
- Join conditions
- Column mappings

## Test Data

The test uses realistic data:
- **employees**: 15 rows with id, first_name, last_name, department, etc.
- **employee_notes**: Performance reviews and notes linked by emp_id

This demonstrates a common use case where employee reference data is cached locally while transactional data (notes) remains on the remote server.