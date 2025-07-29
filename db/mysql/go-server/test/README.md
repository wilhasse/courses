# Hybrid Query System Test

This directory contains test programs for the hybrid query system that demonstrates joining data between LMDB cache and remote MySQL.

## Test Setup

### 1. Remote MySQL Server (10.1.0.7)

First, create the employees table on the remote server:

```bash
# Connect to remote MySQL
mysql -h 10.1.0.7 -u root testdb

# Or use the provided SQL script
mysql -h 10.1.0.7 -u root testdb < setup_remote_table.sql
```

The `setup_remote_table.sql` creates:
- `employees` table with sample employee data
- Fields: id, name, email, department, hire_date, salary

### 2. Local MySQL Server (localhost)

Create the permissions table locally:

```sql
CREATE TABLE permissions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    emp_id INT NOT NULL,
    permission VARCHAR(50) NOT NULL,
    granted_date DATE
);

INSERT INTO permissions (emp_id, permission, granted_date) VALUES
(1, 'READ', '2024-01-01'),
(1, 'WRITE', '2024-01-15'),
(2, 'READ', '2024-02-01'),
(3, 'ADMIN', '2024-03-01'),
(4, 'READ', '2024-01-10'),
(5, 'WRITE', '2024-02-20');
```

## Running the Tests

### Simple Test
```bash
make test-hybrid-simple
```

This test:
1. Loads employees table from 10.1.0.7 into LMDB cache
2. Executes a simple SELECT on the cached data
3. Attempts a JOIN with local permissions table

### Full Demo
```bash
make test-hybrid-demo
```

This comprehensive demo:
1. Loads remote employees table into LMDB
2. Creates local permissions table
3. Tests simple queries on cached data
4. Performs JOIN between cached employees and local permissions
5. Shows query analysis and rewriting
6. Displays performance comparison

### Integration Tests
```bash
make test-hybrid-integration
```

Runs the full test suite (requires INTEGRATION_TEST=true environment variable).

## How It Works

1. **Remote Table Caching**: The employees table from 10.1.0.7 is loaded into local LMDB cache
2. **Query Analysis**: When you run a JOIN query like:
   ```sql
   SELECT e.*, p.* 
   FROM employees e 
   JOIN permissions p ON e.id = p.emp_id
   ```
3. **Query Rewriting**: The system:
   - Detects that `employees` is cached
   - Rewrites the query to fetch only from `permissions` table
   - Gets `employees` data from LMDB cache
4. **In-Memory Join**: Results are joined in memory using the join condition

## Expected Output

```
=== Loading employees table from remote server into LMDB ===
✓ Employees table loaded into LMDB cache

=== Testing JOIN between cached employees and local permissions ===
Query Analysis:
- Has cached table: true
- Cached tables: [{testdb employees }]
- Remote tables: [{testdb permissions }]
- Is join query: true
- Requires rewrite: true

Join Query Results:
Employee ID: 1, Name: John Doe, Permission: READ, Granted: 2024-01-01
Employee ID: 1, Name: John Doe, Permission: WRITE, Granted: 2024-01-15
Employee ID: 2, Name: Jane Smith, Permission: READ, Granted: 2024-02-01
...
```

## Troubleshooting

1. **Connection Error to 10.1.0.7**:
   - Ensure the remote MySQL server is accessible
   - Check firewall rules
   - Verify root user can connect without password

2. **Local Permissions Table Missing**:
   - The demo creates it automatically
   - Or create it manually using the SQL above

3. **LMDB Errors**:
   - Run `make setup` to ensure LMDB is installed
   - Check disk space for cache directory