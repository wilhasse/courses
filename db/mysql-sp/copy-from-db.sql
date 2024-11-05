CREATE DEFINER=`root`@`%` PROCEDURE `copyDataFromTest`()
LANGUAGE SQL
NOT DETERMINISTIC
CONTAINS SQL
SQL SECURITY DEFINER
COMMENT ''
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE v_table VARCHAR(255);
    
    -- Cursor declaration
    DECLARE table_cursor CURSOR FOR 
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = DATABASE() 
    AND TABLE_TYPE = 'BASE TABLE';
    
    -- Handler declaration
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    
    -- Disable foreign keys
    SET SESSION cslog_write_persist_only = 1;
	 SET SESSION rocksdb_bulk_load_size = 10000;
    SET SESSION rocksdb_commit_in_the_middle = 1;
    
    -- Open cursor
    OPEN table_cursor;
    
    copy_loop: LOOP
        -- Get next table
        FETCH table_cursor INTO v_table;
        
        -- Exit if no more tables
        IF done = 1 THEN
            LEAVE copy_loop;
        END IF;
        
        -- Check destination first
        SET @dest_count_stmt = CONCAT('SELECT COUNT(*) INTO @dest_count FROM `', v_table, '`');
        PREPARE stmt FROM @dest_count_stmt;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        
        -- Only proceed if destination is empty
        IF @dest_count = 0 THEN
            -- Check source table
            SET @source_count_stmt = CONCAT('SELECT COUNT(*) INTO @source_count FROM siscob_teste.`', v_table, '`');
            PREPARE stmt FROM @source_count_stmt;
            EXECUTE stmt;
            DEALLOCATE PREPARE stmt;
            
            -- Copy if source has data
            IF @source_count > 0 THEN
                SET @copy_stmt = CONCAT(
                    'INSERT INTO `', v_table, '` ',
                    'SELECT * FROM siscob_teste.`', v_table, '`'
                );
                PREPARE stmt FROM @copy_stmt;
                EXECUTE stmt;
                DEALLOCATE PREPARE stmt;
                
                SELECT CONCAT('Copied ', @source_count, ' rows to table: ', v_table) AS status;
            END IF;
        END IF;
    END LOOP;
    
    -- Close cursor
    CLOSE table_cursor;
    
    -- Re-enable foreign keys
    SET FOREIGN_KEY_CHECKS = 1;
END