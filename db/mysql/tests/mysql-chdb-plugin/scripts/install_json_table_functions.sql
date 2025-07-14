-- Install chDB JSON table functions for MySQL 8.0.19+

-- Drop existing functions if they exist
DROP FUNCTION IF EXISTS chdb_table_json;
DROP FUNCTION IF EXISTS chdb_customers_json;
DROP FUNCTION IF EXISTS chdb_query_json;

-- Create JSON table functions
CREATE FUNCTION chdb_table_json RETURNS STRING 
    SONAME 'chdb_json_table_functions.so';

CREATE FUNCTION chdb_customers_json RETURNS STRING 
    SONAME 'chdb_json_table_functions.so';

CREATE FUNCTION chdb_query_json RETURNS STRING 
    SONAME 'chdb_json_table_functions.so';

SELECT 'chDB JSON table functions installed successfully' AS status;