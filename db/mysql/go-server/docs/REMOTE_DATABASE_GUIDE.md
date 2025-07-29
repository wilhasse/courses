# Remote Database Proxy Guide

This guide explains how to create virtual databases that mirror schema from remote MySQL servers.

## Overview

The Remote Database feature allows you to create a local database that acts as a proxy to a remote MySQL database. When you query this local database, it:

1. Fetches table schemas from the remote database's `information_schema`
2. Forwards all queries (SELECT, INSERT, UPDATE, DELETE) to the remote database
3. Returns results as if the tables existed locally

## Creating a Remote Database

Due to SQL syntax limitations, use the special naming convention:

```sql
CREATE DATABASE dbname__remote__host__port__database__user__password;
```

### Example

To create a proxy to a remote MySQL database:

```sql
-- Connect to production database at 192.168.1.100:3306
CREATE DATABASE prodmirror__remote__192.168.1.100__3306__production__reader__secretpass;

-- Now you can use it like a regular database
USE prodmirror;
SHOW TABLES;  -- Shows tables from the remote 'production' database

-- Query remote tables
SELECT * FROM users;       -- Queries remote users table
SELECT * FROM products;    -- Queries remote products table
```

## How It Works

1. **Schema Discovery**: When you access a table, the proxy queries the remote database's `information_schema.COLUMNS` to get the table structure
2. **Query Forwarding**: All SQL operations are translated and forwarded to the remote MySQL server
3. **Result Translation**: Results from the remote server are converted to match go-mysql-server's type system

## Features

- **Read Operations**: Full support for SELECT queries
- **Write Operations**: INSERT, UPDATE, DELETE are forwarded to remote
- **Schema Caching**: Table schemas are cached after first access
- **Type Mapping**: Automatic conversion between MySQL types

## Limitations

1. **No DDL Operations**: Cannot CREATE/ALTER/DROP tables on remote database
2. **No Transactions**: Each query is executed independently
3. **Performance**: Network latency affects query performance
4. **Authentication**: Credentials are part of database name (temporary solution)

## Security Considerations

⚠️ **Warning**: The current implementation includes credentials in the database name. This is for demonstration purposes only. In production:

1. Use a secure credential store
2. Implement proper authentication mechanisms
3. Use SSL/TLS connections
4. Apply principle of least privilege for remote access

## Use Cases

1. **Development/Testing**: Mirror production schema without copying data
2. **Data Federation**: Query multiple MySQL servers from one interface
3. **Read Replicas**: Create read-only views of remote databases
4. **Migration**: Gradually move from remote to local storage

## Troubleshooting

### Connection Failed
- Verify remote MySQL server is accessible
- Check firewall rules
- Ensure user has proper permissions

### Table Not Found
- Verify table exists in remote database
- Check user has SELECT permission on table
- Try `SHOW TABLES` to list available tables

### Type Conversion Errors
- Some MySQL types may not map perfectly
- Check data types in remote schema
- Report issues for specific type mappings