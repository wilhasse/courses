-- Uninstall chDB API UDF functions

DROP FUNCTION IF EXISTS chdb_query;
DROP FUNCTION IF EXISTS chdb_count;
DROP FUNCTION IF EXISTS chdb_sum;

SELECT 'chDB API UDF functions uninstalled successfully' AS status;