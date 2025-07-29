# Quick Start Guide

Get up and running with the MySQL server and virtual databases in 5 minutes.

## 1. Build & Run

```bash
# Clone and enter directory
cd mysql-server-example

# Build (automatic setup)
make build

# Run server with external access
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH
./bin/mysql-server --bind 0.0.0.0
```

## 2. Connect to Server

```bash
mysql -h localhost -P 3306 -u root
```

## 3. Create Virtual Database

Replace the connection details with your remote MySQL server:

```sql
-- Format: dbname__remote__host__port__database__user__password
-- Example for 192.168.1.100:3306 with database 'production'
CREATE DATABASE prod__remote__192_168_1_100__3306__production__dbuser__dbpass;

-- Real example that was tested:
CREATE DATABASE siscom_prod__remote__172_16_120_11__3306__cslog_siscom_prod__appl_cslog__D981xATa;
```

**Important**: 
- Replace dots with underscores: `192.168.1.100` → `192_168_1_100`
- Replace @ with AT: `pass@word` → `passATword`

## 4. Use Virtual Database

```sql
-- Switch to virtual database
USE prod;

-- Now all queries go to remote MySQL!
SHOW TABLES;
SELECT * FROM your_table LIMIT 10;
```

## Common Commands

### Server Management
```bash
# Start server (localhost only)
./bin/mysql-server

# Start server (external access)
./bin/mysql-server --bind 0.0.0.0

# Start with debug mode
./bin/mysql-server --debug --bind 0.0.0.0

# Custom port
./bin/mysql-server --port 3307 --bind 0.0.0.0
```

### Database Operations
```sql
-- List all databases (local + virtual)
SHOW DATABASES;

-- Create virtual database
CREATE DATABASE name__remote__host__port__db__user__pass;

-- Use virtual database
USE name;

-- List remote tables
SHOW TABLES;
SHOW TABLES LIKE 'user%';

-- Query remote data
SELECT * FROM table_name;
INSERT INTO table_name (col1, col2) VALUES ('val1', 'val2');
UPDATE table_name SET col1 = 'new' WHERE id = 1;
DELETE FROM table_name WHERE id = 1;

-- Drop virtual database
DROP DATABASE name;
```

## Troubleshooting

### Can't connect to server?
```bash
# Check if server is running
ps aux | grep mysql-server

# Check server logs
# Look for "Server listening on" message

# Make sure LMDB library is in path
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH
```

### Virtual database connection failed?
```bash
# Test direct connection to remote MySQL
mysql -h remote_host -P 3306 -u username -p

# Check firewall on remote server
# Ensure remote MySQL allows external connections
```

### Can't connect from another machine?
```bash
# Make sure server is bound to 0.0.0.0
./bin/mysql-server --bind 0.0.0.0

# Check local firewall
sudo ufw allow 3306

# Get server IP
ip addr show
```

## Next Steps

- Read the [Virtual Database User Guide](docs/VIRTUAL_DATABASE_USER_GUIDE.md) for detailed information
- Check [Remote Database Working Example](docs/REMOTE_DATABASE_WORKING_EXAMPLE.md) for a real-world example
- See [CLAUDE.md](CLAUDE.md) for developer information