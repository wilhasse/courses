# Formatting chDB Query Results in MySQL

Since MySQL UDFs return strings, not result sets, here are different approaches to work with multi-row data from chDB:

## 1. Basic Query (Tab-Separated)

The raw output is tab-separated values with newlines:

```sql
SELECT CAST(chdb_api_query('SELECT ID_CONTR,SEQ,CODIGO FROM mysql_import.historico LIMIT 10') AS CHAR);
```

## 2. Using SUBSTRING_INDEX for Single Row

Extract specific rows from the result:

```sql
-- Get first row
SELECT SUBSTRING_INDEX(
    CAST(chdb_api_query('SELECT ID_CONTR,SEQ,CODIGO FROM mysql_import.historico LIMIT 10') AS CHAR),
    '\n', 1
) AS first_row;

-- Get second row
SELECT SUBSTRING_INDEX(
    SUBSTRING_INDEX(
        CAST(chdb_api_query('SELECT ID_CONTR,SEQ,CODIGO FROM mysql_import.historico LIMIT 10') AS CHAR),
        '\n', 2
    ),
    '\n', -1
) AS second_row;
```

## 3. Create a Stored Procedure for Better Display

```sql
DELIMITER $$

DROP PROCEDURE IF EXISTS chdb_query_formatted$$

CREATE PROCEDURE chdb_query_formatted(IN query_str TEXT)
BEGIN
    DECLARE result TEXT;
    DECLARE line_count INT DEFAULT 0;
    DECLARE current_pos INT DEFAULT 1;
    DECLARE next_pos INT;
    DECLARE current_line TEXT;
    
    -- Execute the query
    SET result = CAST(chdb_api_query(query_str) AS CHAR);
    
    -- Create temporary table for results
    DROP TEMPORARY TABLE IF EXISTS chdb_temp_results;
    CREATE TEMPORARY TABLE chdb_temp_results (
        row_num INT AUTO_INCREMENT PRIMARY KEY,
        line TEXT
    );
    
    -- Split by newlines and insert into temp table
    WHILE current_pos <= LENGTH(result) DO
        SET next_pos = LOCATE('\n', result, current_pos);
        
        IF next_pos = 0 THEN
            SET next_pos = LENGTH(result) + 1;
        END IF;
        
        SET current_line = SUBSTRING(result, current_pos, next_pos - current_pos);
        
        IF LENGTH(TRIM(current_line)) > 0 THEN
            INSERT INTO chdb_temp_results (line) VALUES (current_line);
        END IF;
        
        SET current_pos = next_pos + 1;
    END WHILE;
    
    -- Display results
    SELECT row_num, line FROM chdb_temp_results;
    
    -- Clean up
    DROP TEMPORARY TABLE chdb_temp_results;
END$$

DELIMITER ;

-- Usage:
CALL chdb_query_formatted('SELECT ID_CONTR,SEQ,CODIGO FROM mysql_import.historico LIMIT 10');
```

## 4. Parse Tab-Separated Values

```sql
DELIMITER $$

DROP PROCEDURE IF EXISTS chdb_query_as_table$$

CREATE PROCEDURE chdb_query_as_table(IN query_str TEXT)
BEGIN
    DECLARE result TEXT;
    DECLARE current_pos INT DEFAULT 1;
    DECLARE next_pos INT;
    DECLARE current_line TEXT;
    DECLARE col1 TEXT;
    DECLARE col2 TEXT;
    DECLARE col3 TEXT;
    DECLARE tab_pos1 INT;
    DECLARE tab_pos2 INT;
    
    -- Execute the query
    SET result = CAST(chdb_api_query(query_str) AS CHAR);
    
    -- Create temporary table with proper columns
    DROP TEMPORARY TABLE IF EXISTS chdb_parsed_results;
    CREATE TEMPORARY TABLE chdb_parsed_results (
        ID_CONTR INT,
        SEQ INT,
        CODIGO INT
    );
    
    -- Parse each line
    WHILE current_pos <= LENGTH(result) DO
        SET next_pos = LOCATE('\n', result, current_pos);
        
        IF next_pos = 0 THEN
            SET next_pos = LENGTH(result) + 1;
        END IF;
        
        SET current_line = SUBSTRING(result, current_pos, next_pos - current_pos);
        
        IF LENGTH(TRIM(current_line)) > 0 THEN
            -- Parse tab-separated values
            SET tab_pos1 = LOCATE('\t', current_line);
            SET tab_pos2 = LOCATE('\t', current_line, tab_pos1 + 1);
            
            IF tab_pos1 > 0 AND tab_pos2 > 0 THEN
                SET col1 = SUBSTRING(current_line, 1, tab_pos1 - 1);
                SET col2 = SUBSTRING(current_line, tab_pos1 + 1, tab_pos2 - tab_pos1 - 1);
                SET col3 = SUBSTRING(current_line, tab_pos2 + 1);
                
                INSERT INTO chdb_parsed_results VALUES (
                    CAST(col1 AS UNSIGNED),
                    CAST(col2 AS UNSIGNED),
                    CAST(col3 AS UNSIGNED)
                );
            END IF;
        END IF;
        
        SET current_pos = next_pos + 1;
    END WHILE;
    
    -- Display as proper table
    SELECT * FROM chdb_parsed_results;
    
    -- Clean up
    DROP TEMPORARY TABLE chdb_parsed_results;
END$$

DELIMITER ;

-- Usage:
CALL chdb_query_as_table('SELECT ID_CONTR,SEQ,CODIGO FROM mysql_import.historico LIMIT 10');
```

## 5. Using JSON (MySQL 8.0+)

If you modify the chDB query to return JSON:

```sql
-- Query that returns JSON array
SELECT CAST(chdb_api_query('
    SELECT JSONCompactEachRow(ID_CONTR, SEQ, CODIGO) 
    FROM mysql_import.historico 
    LIMIT 10
') AS CHAR);

-- Or use JSON format
SELECT CAST(chdb_api_query('
    SELECT ID_CONTR, SEQ, CODIGO 
    FROM mysql_import.historico 
    LIMIT 10 
    FORMAT JSON
') AS CHAR);
```

## 6. Best Practice: Use JSON_TABLE (MySQL 8.0.19+)

The cleanest approach is to have chDB return JSON and use JSON_TABLE:

```sql
-- First, create a function that returns JSON format
-- Then use JSON_TABLE to parse it:

SELECT jt.*
FROM JSON_TABLE(
    chdb_api_query('
        SELECT toJSONString(groupArray(
            tuple(ID_CONTR, SEQ, CODIGO)
        )) 
        FROM mysql_import.historico 
        LIMIT 10
    '),
    '$[*]' COLUMNS (
        ID_CONTR INT PATH '$[0]',
        SEQ INT PATH '$[1]',
        CODIGO INT PATH '$[2]'
    )
) AS jt;
```

## 7. Command Line Alternative

For better formatted output, use the command line:

```bash
mysql -u root -e "SELECT CAST(chdb_api_query('SELECT ID_CONTR,SEQ,CODIGO FROM mysql_import.historico LIMIT 10') AS CHAR)\G"
```

## Summary

While MySQL UDFs can't return true result sets, you can:
1. Use stored procedures to parse and display the data
2. Return JSON from chDB and use JSON_TABLE
3. Process the results in your application code
4. Use the command line for better formatting

The stored procedure approach gives you the most flexibility for displaying tabular data within MySQL.