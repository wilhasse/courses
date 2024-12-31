CREATE PROCEDURE `analyze_db`()
LANGUAGE SQL
NOT DETERMINISTIC
CONTAINS SQL
SQL SECURITY DEFINER
COMMENT ''
BEGIN
    -- Declarations must all come first
    DECLARE done INT DEFAULT 0;
    DECLARE TNAME CHAR(255);
    DECLARE start_time TIMESTAMP;
    DECLARE table_count INT;
    DECLARE current_table INT DEFAULT 0;
    
    -- Cursor declaration must come before handler declarations
    DECLARE table_names CURSOR for
        SELECT table_name 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = DATABASE() 
        ORDER BY DATA_LENGTH DESC;
    
    -- Handlers come after cursor declarations
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION 
    BEGIN
        UPDATE analyze_progress 
        SET status = 'ERROR', 
            end_time = NOW() 
        WHERE table_name = TNAME AND end_time IS NULL;
    END;
    
    -- Now we can start the actual procedure logic
    DROP TEMPORARY TABLE IF EXISTS analyze_progress;
    CREATE TEMPORARY TABLE analyze_progress (
        table_name VARCHAR(255),
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        status VARCHAR(100)
    );
    
    -- Get total number of tables
    SELECT COUNT(*) INTO table_count 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = DATABASE();
    
    OPEN table_names;
    
    table_loop: WHILE done = 0 DO
        FETCH NEXT FROM table_names INTO TNAME;
        IF done = 0 THEN
            SET current_table = current_table + 1;
            
            -- Log start of table analysis
            INSERT INTO analyze_progress (table_name, start_time, status)
            VALUES (TNAME, NOW(), 'PROCESSING');
            
            -- Execute ANALYZE TABLE
            SET @SQL_TXT = CONCAT("ANALYZE TABLE ", TNAME, "\\G");
            PREPARE stmt_name FROM @SQL_TXT;
            EXECUTE stmt_name;
            DEALLOCATE PREPARE stmt_name;
            
            -- Update progress
            UPDATE analyze_progress 
            SET status = 'COMPLETED', 
                end_time = NOW() 
            WHERE table_name = TNAME;
        END IF;
    END WHILE;
    
    -- Show final summary
    SELECT 
        table_name,
        start_time,
        end_time,
        TIMESTAMPDIFF(SECOND, start_time, COALESCE(end_time, NOW())) as duration_seconds,
        status
    FROM analyze_progress
    ORDER BY start_time;
    
    CLOSE table_names;
    
    -- Clean up
    DROP TEMPORARY TABLE IF EXISTS analyze_progress;
END