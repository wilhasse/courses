# Remote Database Working Example

This document shows a successful configuration and test of the virtual database feature.

## Configuration Used

- **Remote MySQL Server**: 172.16.120.11:3306
- **Remote Database**: cslog_siscom_prod
- **User**: appl_cslog
- **Password**: D981x@a

## Steps to Configure

1. **Start the MySQL server**:
```bash
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH
./bin/mysql-server
```

2. **Create the virtual database**:
```bash
# Format: dbname__remote__host__port__database__user__password
# Note: Replace dots with underscores in IP address, @ with AT in password
mysql -h 127.0.0.1 -P 3306 -u root -e "CREATE DATABASE siscom_prod__remote__172_16_120_11__3306__cslog_siscom_prod__appl_cslog__D981xATa;"
```

3. **Use the virtual database**:
```bash
mysql -h 127.0.0.1 -P 3306 -u root -e "USE siscom_prod; SHOW TABLES;"
```

## Working Features

✅ **Database Creation**: Virtual database created successfully
✅ **Table Listing**: `SHOW TABLES` works and displays all remote tables
✅ **Table Filtering**: `SHOW TABLES LIKE 'pattern%'` works
✅ **Basic Queries**: SELECT queries are forwarded to remote database

## Known Limitations

❌ **DESCRIBE TABLE**: Not yet implemented (returns error about unresolved plan)
❌ **Complex Queries**: Some advanced SQL features may not be fully supported
❌ **Performance**: All queries go over network, adding latency

## Example Session

```sql
-- List databases
mysql> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| siscom_prod        |  -- This is our virtual database
| testdb             |
+--------------------+

-- Use virtual database
mysql> USE siscom_prod;

-- List tables (returns 2232+ tables from remote)
mysql> SHOW TABLES;
[... lists all remote tables ...]

-- Filter tables
mysql> SHOW TABLES LIKE 'CLIENTE%';
+----------------------+
| Tables_in_siscom_prod |
+----------------------+
| CLIENTE              |
| CLIENTE_PGTO_MERCADO |
+----------------------+
```

## Technical Details

- The implementation uses `information_schema` queries to fetch table structure
- All DML operations (SELECT, INSERT, UPDATE, DELETE) are proxied to remote
- Connection pooling is handled by the go-sql-driver/mysql package
- Table schemas are cached after first access for performance