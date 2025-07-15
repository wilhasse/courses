-- MySQL Commands to Check Installed UDF Functions

-- 1. Check all UDF functions in mysql.func table
SELECT * FROM mysql.func;

-- 2. Check only chDB related functions
SELECT name, dl, type FROM mysql.func WHERE name LIKE 'chdb%';

-- 3. Check specific function exists
SELECT COUNT(*) as exists FROM mysql.func WHERE name = 'chdb_api_query';

-- 4. Show function details with better formatting
SELECT 
    name AS function_name,
    dl AS plugin_file,
    type AS function_type,
    CASE 
        WHEN ret = 0 THEN 'STRING'
        WHEN ret = 1 THEN 'REAL'
        WHEN ret = 2 THEN 'INTEGER'
        ELSE 'UNKNOWN'
    END AS return_type
FROM mysql.func 
WHERE name LIKE 'chdb%'
ORDER BY name;

-- 5. Check if specific functions exist (returns 1 if exists, 0 if not)
SELECT 
    (SELECT COUNT(*) FROM mysql.func WHERE name = 'chdb_api_query') as has_chdb_api_query,
    (SELECT COUNT(*) FROM mysql.func WHERE name = 'chdb_api_query_json') as has_chdb_api_query_json,
    (SELECT COUNT(*) FROM mysql.func WHERE name = 'chdb_query_remote') as has_chdb_query_remote,
    (SELECT COUNT(*) FROM mysql.func WHERE name = 'chdb_query_json_remote') as has_chdb_query_json_remote;

-- 6. Show FUNCTION STATUS (for stored functions, not UDFs)
-- Note: This won't show UDFs, only stored procedures/functions
SHOW FUNCTION STATUS WHERE Db = 'mysql';

-- 7. Test if a function is working (will error if not installed)
-- SELECT chdb_api_query('SELECT 1');

-- 8. Group functions by plugin file
SELECT 
    dl AS plugin_file,
    GROUP_CONCAT(name ORDER BY name) AS functions
FROM mysql.func 
WHERE name LIKE 'chdb%'
GROUP BY dl;

-- 9. Check function with specific plugin file
SELECT name, dl 
FROM mysql.func 
WHERE dl IN ('chdb_api_udf.so', 'chdb_api_json_udf.so', 'chdb_api_ip_udf.so', 'chdb_api_ip_json_udf.so');

-- 10. Quick one-liner to list all chDB functions
SELECT GROUP_CONCAT(name) AS installed_chdb_functions FROM mysql.func WHERE name LIKE 'chdb%';