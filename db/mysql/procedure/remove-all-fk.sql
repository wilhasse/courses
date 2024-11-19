DROP PROCEDURE IF EXISTS remove_all_fk;
CREATE PROCEDURE `removeAllFK`()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE TNAME, CNAME CHAR(255);
    DECLARE cur CURSOR FOR
        SELECT TABLE_NAME, CONSTRAINT_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
        WHERE CONSTRAINT_TYPE = 'FOREIGN KEY'
          AND TABLE_SCHEMA = DATABASE();
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    SET FOREIGN_KEY_CHECKS = 0;
    
    OPEN cur;
    
    read_loop: LOOP
        FETCH cur INTO TNAME, CNAME;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        SET @sql = CONCAT('ALTER TABLE ', TNAME, ' DROP FOREIGN KEY ', CNAME);
        SELECT CONCAT('Dropping foreign key: ', CNAME, ' from table: ', TNAME) AS debug_message;
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END LOOP;
    
    CLOSE cur;
    
    SET FOREIGN_KEY_CHECKS = 1;
    
    SELECT 'All foreign keys have been removed' AS debug_message;
END;
