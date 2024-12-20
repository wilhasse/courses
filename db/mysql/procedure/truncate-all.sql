DROP PROCEDURE IF EXISTS truncate_all_tables;
CREATE PROCEDURE `truncate_all_tables`()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE TNAME CHAR(255);
    DECLARE cur CURSOR FOR
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_TYPE = 'BASE TABLE';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    SET FOREIGN_KEY_CHECKS = 0;
    
    OPEN cur;
    
    read_loop: LOOP
        FETCH cur INTO TNAME;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        SET @sql = CONCAT('TRUNCATE TABLE ', TNAME);
        SELECT CONCAT('Truncating table: ', TNAME) AS debug_message;
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END LOOP;
    
    CLOSE cur;
    
    SET FOREIGN_KEY_CHECKS = 1;
    
    SELECT 'All tables have been truncated' AS debug_message;
END;