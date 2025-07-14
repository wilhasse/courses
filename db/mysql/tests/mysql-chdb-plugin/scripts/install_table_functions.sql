-- Install chDB table simulation functions

-- Drop existing functions if they exist
DROP FUNCTION IF EXISTS chdb_table_row_count;
DROP FUNCTION IF EXISTS chdb_table_get_field;
DROP FUNCTION IF EXISTS chdb_table_get_row;
DROP FUNCTION IF EXISTS chdb_customers_get_id;
DROP FUNCTION IF EXISTS chdb_customers_get_name;
DROP FUNCTION IF EXISTS chdb_customers_get_city;

-- Create generic table functions
CREATE FUNCTION chdb_table_row_count RETURNS INTEGER 
    SONAME 'chdb_table_functions.so';

CREATE FUNCTION chdb_table_get_field RETURNS STRING 
    SONAME 'chdb_table_functions.so';

CREATE FUNCTION chdb_table_get_row RETURNS STRING 
    SONAME 'chdb_table_functions.so';

-- Create specialized customer table functions
CREATE FUNCTION chdb_customers_get_id RETURNS INTEGER 
    SONAME 'chdb_table_functions.so';

CREATE FUNCTION chdb_customers_get_name RETURNS STRING 
    SONAME 'chdb_table_functions.so';

CREATE FUNCTION chdb_customers_get_city RETURNS STRING 
    SONAME 'chdb_table_functions.so';

SELECT 'chDB table functions installed successfully' AS status;