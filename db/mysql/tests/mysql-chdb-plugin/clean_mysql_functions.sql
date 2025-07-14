-- Clean up all UDF functions from mysql.func table
-- This removes the references to missing .so files

-- First, show what's currently registered
SELECT '=== Current UDF Functions ===' as info;
SELECT name, dl FROM mysql.func;

-- Remove all ch_ functions
DELETE FROM mysql.func WHERE name LIKE 'ch_%';
DELETE FROM mysql.func WHERE name = 'chdb_columns';
DELETE FROM mysql.func WHERE name = 'chdb_to_utf8';

-- Show what's left
SELECT '=== After Cleanup ===' as info;
SELECT name, dl FROM mysql.func;

-- Flush to apply changes
FLUSH PRIVILEGES;