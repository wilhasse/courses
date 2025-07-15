-- Example: Formatting chDB results as a table

-- 1. Create a simple parsing procedure
DELIMITER $$

DROP PROCEDURE IF EXISTS show_historico$$

CREATE PROCEDURE show_historico(IN limit_rows INT)
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
    
    -- Build and execute the query
    SET @query = CONCAT('SELECT ID_CONTR,SEQ,CODIGO FROM mysql_import.historico LIMIT ', limit_rows);
    SET result = CAST(chdb_api_query(@query) AS CHAR);
    
    -- Create temporary table
    DROP TEMPORARY TABLE IF EXISTS temp_historico;
    CREATE TEMPORARY TABLE temp_historico (
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
            -- Find tab positions
            SET tab_pos1 = LOCATE(CHAR(9), current_line);
            SET tab_pos2 = LOCATE(CHAR(9), current_line, tab_pos1 + 1);
            
            IF tab_pos1 > 0 AND tab_pos2 > 0 THEN
                SET col1 = SUBSTRING(current_line, 1, tab_pos1 - 1);
                SET col2 = SUBSTRING(current_line, tab_pos1 + 1, tab_pos2 - tab_pos1 - 1);
                SET col3 = SUBSTRING(current_line, tab_pos2 + 1);
                
                INSERT INTO temp_historico VALUES (
                    CAST(col1 AS UNSIGNED),
                    CAST(col2 AS UNSIGNED),
                    CAST(col3 AS UNSIGNED)
                );
            END IF;
        END IF;
        
        SET current_pos = next_pos + 1;
    END WHILE;
    
    -- Display as table
    SELECT * FROM temp_historico;
    
    -- Clean up
    DROP TEMPORARY TABLE temp_historico;
END$$

DELIMITER ;

-- Usage examples:
CALL show_historico(10);
CALL show_historico(5);