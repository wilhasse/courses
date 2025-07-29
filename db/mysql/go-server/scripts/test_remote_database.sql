-- Test script for Remote Database functionality
-- This demonstrates how to create and use a remote database proxy

-- Example 1: Create a remote database proxy
-- Replace with your actual remote MySQL server details
-- Format: dbname__remote__host__port__database__user__password
CREATE DATABASE IF NOT EXISTS remote_test__remote__localhost__3307__testdb__root__password;

-- Use the remote database
USE remote_test;

-- Show tables from the remote database
SHOW TABLES;

-- Query remote tables (these queries are forwarded to the remote MySQL server)
-- Uncomment and adjust table names based on what exists in your remote database
-- SELECT * FROM users LIMIT 10;
-- SELECT COUNT(*) FROM products;

-- Example 2: Create multiple remote database connections
-- You can have multiple remote databases connected at once
-- CREATE DATABASE staging__remote__staging.example.com__3306__staging_db__reader__readpass;
-- CREATE DATABASE analytics__remote__analytics.example.com__3306__metrics__analyst__analypass;

-- Example 3: Join across remote and local databases
-- This works because all databases are accessed through the same SQL engine
-- SELECT 
--     l.id as local_id,
--     r.name as remote_name
-- FROM 
--     testdb.local_table l
--     JOIN remote_test.remote_table r ON l.remote_id = r.id;

-- Drop the test remote database when done
-- DROP DATABASE remote_test;

-- Note: To test this properly, you need:
-- 1. A running MySQL server on a different port (e.g., 3307) or host
-- 2. A database with some tables on that server
-- 3. Valid credentials to access it

-- For local testing, you can:
-- 1. Run another MySQL instance: docker run -p 3307:3306 -e MYSQL_ROOT_PASSWORD=password mysql:8
-- 2. Create a test database: mysql -h localhost -P 3307 -u root -ppassword -e "CREATE DATABASE testdb"
-- 3. Create test tables in that database
-- 4. Then use this script to connect to it