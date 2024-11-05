
CREATE PROCEDURE `changeToRocksDB`()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE TNAME CHAR(255);
    DECLARE error_msg TEXT;
    DECLARE cur CURSOR FOR
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
          AND ENGINE = 'InnoDB'
          AND TABLE_NAME <> 'LAYOUT_MEM';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1
        error_msg = MESSAGE_TEXT;
        SELECT CONCAT('Error changing engine for table ', TNAME, ': ', error_msg) AS error_message;
    END;
    
    CREATE TEMPORARY TABLE IF NOT EXISTS failed_tables (
        table_name VARCHAR(255),
        error_message TEXT
    );
    
    OPEN cur;
    
    read_loop: LOOP
        FETCH cur INTO TNAME;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        BEGIN
            DECLARE EXIT HANDLER FOR SQLEXCEPTION
            BEGIN
                GET DIAGNOSTICS CONDITION 1
                error_msg = MESSAGE_TEXT;
                INSERT INTO failed_tables VALUES (TNAME, error_msg);
            END;
            
            SET @sql = CONCAT('ALTER TABLE ', TNAME, ' ENGINE = RocksDB');
            SELECT CONCAT('Changing engine for table: ', TNAME) AS debug_message;
            PREPARE stmt FROM @sql;
            EXECUTE stmt;
            DEALLOCATE PREPARE stmt;
            SELECT CONCAT('Successfully changed engine for table: ', TNAME) AS success_message;
        END;
    END LOOP;
    
    CLOSE cur;
    
    SELECT 'Attempted to change all eligible tables to RocksDB' AS debug_message;
    
    SELECT * FROM failed_tables;
    DROP TEMPORARY TABLE failed_tables;
END;