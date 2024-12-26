DROP PROCEDURE IF EXISTS compress_all_tables;
CREATE PROCEDURE `compress_all_tables`()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE TNAME CHAR(255);
    DECLARE cur CURSOR FOR
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_TYPE = 'BASE TABLE' 
        AND ROW_FORMAT <> 'COMPRESSED';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    
    OPEN cur;
    
    read_loop: LOOP
        FETCH cur INTO TNAME;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        SET @sql = CONCAT('ALTER TABLE ', TNAME, ' ROW_FORMAT=COMPRESSED');
        SELECT CONCAT('Compressing table: ', TNAME) AS debug_message;
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END LOOP;
    
    CLOSE cur;  
   
    SELECT 'All tables have been compressed' AS debug_message;
END;