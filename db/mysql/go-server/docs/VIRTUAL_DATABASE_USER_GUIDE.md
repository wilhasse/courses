# Virtual Database User Guide

This guide explains how to use the virtual database feature to connect to remote MySQL servers.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Creating Virtual Databases](#creating-virtual-databases)
- [Using Virtual Databases](#using-virtual-databases)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Security](#security)

## Overview

Virtual databases allow you to create local database proxies that forward all queries to remote MySQL servers. This is useful for:

- **Federation**: Query multiple MySQL servers from one interface
- **Development**: Mirror production schemas without copying data
- **Migration**: Gradually move from remote to local storage
- **Security**: Use read-only credentials for safe production access

## Quick Start

### 1. Start the Server

```bash
# Build the server
make build

# Start with external access enabled
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH
./bin/mysql-server --bind 0.0.0.0
```

### 2. Create a Virtual Database

```sql
-- Connect to the server
mysql -h localhost -P 3306 -u root

-- Create virtual database
-- Format: name__remote__host__port__database__user__password
CREATE DATABASE myapp__remote__192_168_1_100__3306__production__readonly__Pass123;
```

### 3. Use the Virtual Database

```sql
-- Switch to virtual database
USE myapp;

-- All queries are now forwarded to the remote server
SHOW TABLES;
SELECT * FROM users LIMIT 10;
```

## Creating Virtual Databases

### Naming Convention

Virtual databases use a special naming format:

```
<local_name>__remote__<host>__<port>__<remote_database>__<username>__<password>
```

### Character Escaping Rules

| Character | Replace With | Example |
|-----------|--------------|---------|
| `.` (dot) | `_` (underscore) | `192.168.1.1` → `192_168_1_1` |
| `@` | `AT` | `user@123` → `userAT123` |
| `:` | `COLON` | `pass:word` → `passCOLONword` |

### Examples

1. **Basic Example**
   ```sql
   -- Remote: mysql://dbuser:pass123@192.168.1.100:3306/myapp
   CREATE DATABASE app__remote__192_168_1_100__3306__myapp__dbuser__pass123;
   ```

2. **With Special Characters**
   ```sql
   -- Remote: mysql://app_user:P@ss#123@db.example.com:3306/production
   CREATE DATABASE prod__remote__db_example_com__3306__production__app_user__PATss#123;
   ```

3. **Non-Standard Port**
   ```sql
   -- Remote: mysql://reader:readonly@10.0.0.50:3307/analytics
   CREATE DATABASE analytics__remote__10_0_0_50__3307__analytics__reader__readonly;
   ```

## Using Virtual Databases

### Basic Operations

```sql
-- List all databases (includes virtual ones)
SHOW DATABASES;

-- Switch to virtual database
USE myapp;

-- List tables from remote
SHOW TABLES;

-- Query remote data
SELECT * FROM customers WHERE created_at > '2024-01-01';

-- Insert data (if user has permissions)
INSERT INTO logs (message, timestamp) VALUES ('Login', NOW());

-- Update remote data
UPDATE settings SET value = 'true' WHERE key = 'maintenance';

-- Delete from remote
DELETE FROM sessions WHERE expired_at < NOW();
```

### Supported Features

✅ **Data Query Operations**
- SELECT with all clauses (WHERE, JOIN, GROUP BY, etc.)
- INSERT single and multiple rows
- UPDATE with conditions
- DELETE with conditions

✅ **Information Commands**
- SHOW TABLES
- SHOW TABLES LIKE 'pattern%'
- USE database

✅ **Advanced Features**
- Connection pooling to remote servers
- Schema caching for performance
- Multiple simultaneous virtual databases

### Limitations

❌ **Not Supported**
- DESCRIBE/DESC table (use SHOW CREATE TABLE on remote)
- CREATE/ALTER/DROP TABLE on remote
- Stored procedures and functions
- Triggers and events
- Cross-database JOINs with local databases

## Examples

### Example 1: Development Environment

Mirror production database for local development:

```sql
-- Create read-only mirror of production
CREATE DATABASE prod_mirror__remote__prod_db_host__3306__production__readonly_user__ReadOnly123;

USE prod_mirror;

-- Safe to query production data
SELECT COUNT(*) FROM orders WHERE status = 'pending';
SELECT * FROM products WHERE active = 1 LIMIT 100;
```

### Example 2: Multi-Database Federation

Connect to multiple remote databases:

```sql
-- Customer database
CREATE DATABASE customers__remote__10_1_1_10__3306__customer_db__app__password1;

-- Orders database  
CREATE DATABASE orders__remote__10_1_1_20__3306__order_db__app__password2;

-- Analytics database
CREATE DATABASE analytics__remote__10_1_1_30__3306__analytics_db__reader__password3;

-- Query across virtual databases (each forwarded separately)
USE customers;
SELECT * FROM users WHERE country = 'US';

USE orders;
SELECT * FROM orders WHERE user_id IN (123, 456, 789);
```

### Example 3: Data Migration

Gradual migration from remote to local:

```sql
-- Phase 1: Create virtual database
CREATE DATABASE legacy__remote__old_server__3306__legacy_db__migrator__MigPass123;

-- Phase 2: Create local database
CREATE DATABASE legacy_new;

-- Phase 3: Copy data selectively
USE legacy;
-- Query remote data

USE legacy_new;
-- Insert into local storage

-- Phase 4: Switch application to local database
-- Phase 5: Drop virtual database when migration complete
DROP DATABASE legacy;
```

## Troubleshooting

### Connection Issues

**Error**: "failed to ping remote MySQL"
```bash
# Test direct connection first
mysql -h <remote_host> -P <port> -u <user> -p

# Check network connectivity
ping <remote_host>
telnet <remote_host> <port>
```

**Error**: "Access denied for user"
- Verify username and password
- Check user permissions on remote: `SHOW GRANTS FOR 'user'@'%';`
- Ensure remote MySQL allows connections from your IP

### Query Issues

**Error**: "Table doesn't exist"
```sql
-- On remote server, check table exists
SHOW TABLES LIKE 'table_name';

-- Check case sensitivity
-- MySQL on Linux is case-sensitive for table names
```

**Slow Queries**
- Network latency affects all queries
- Consider creating indexes on remote
- Use LIMIT for large result sets
- Cache frequently accessed data locally

### Debugging

Enable debug mode for detailed logging:

```bash
# Start server with debug flag
./bin/mysql-server --debug --bind 0.0.0.0

# Watch query forwarding
tail -f server.log | grep -i proxy
```

## Security

### Best Practices

1. **Use Read-Only Credentials**
   ```sql
   -- On remote MySQL, create read-only user
   CREATE USER 'readonly'@'%' IDENTIFIED BY 'StrongPassword123';
   GRANT SELECT ON production.* TO 'readonly'@'%';
   ```

2. **Firewall Configuration**
   ```bash
   # Only allow specific IPs to connect
   sudo ufw allow from 192.168.1.0/24 to any port 3306
   ```

3. **Use SSH Tunneling**
   ```bash
   # Create secure tunnel
   ssh -L 3306:localhost:3306 user@mysql-proxy-server
   
   # Connect through tunnel
   mysql -h localhost -P 3306 -u root
   ```

4. **Rotate Credentials Regularly**
   - Drop and recreate virtual databases with new passwords
   - Use strong, unique passwords for each connection

### Security Warnings

⚠️ **Credentials Visible**: Database names contain passwords. Limit access to database listing.

⚠️ **No Encryption**: Connections between proxy and remote are not encrypted by default.

⚠️ **Audit Trail**: Queries are forwarded with proxy server's IP, not original client.

## Advanced Configuration

### Environment Variables

```bash
# Set default behavior
export BIND_ADDR=0.0.0.0
export PORT=3306
export DEBUG=true
export LD_LIBRARY_PATH=/path/to/lmdb/lib

./bin/mysql-server
```

### Custom Port Configuration

```bash
# Run on different port
./bin/mysql-server --port 3307 --bind 0.0.0.0

# Connect to custom port
mysql -h localhost -P 3307 -u root
```

### High Availability Setup

For production use, consider:

1. **Load Balancer**: Place HAProxy/ProxySQL in front
2. **Multiple Instances**: Run multiple proxy servers
3. **Connection Pooling**: Configure max connections per remote
4. **Monitoring**: Track query latency and errors

## Performance Tuning

1. **Schema Caching**: Schemas are cached after first access
2. **Connection Pooling**: Reuses connections to remote servers
3. **Query Optimization**: 
   - Use specific column names instead of SELECT *
   - Add indexes on remote for frequent queries
   - Use LIMIT for large result sets

4. **Network Optimization**:
   - Place proxy close to remote MySQL (same datacenter)
   - Use persistent connections
   - Consider compression for large results

## Conclusion

Virtual databases provide a powerful way to federate multiple MySQL servers. They're ideal for development, migration, and read-only production access. Always follow security best practices and monitor performance for production use.