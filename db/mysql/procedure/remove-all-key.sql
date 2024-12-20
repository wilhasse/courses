CREATE DEFINER=`root`@`localhost` PROCEDURE `removeAllKeys`()
LANGUAGE SQL
NOT DETERMINISTIC
CONTAINS SQL
SQL SECURITY DEFINER
COMMENT ''
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE TNAME, CNAME, CONSTRAINT_TYPE VARCHAR(255);
    
    DECLARE cur CURSOR FOR
        SELECT TABLE_NAME, CONSTRAINT_NAME, CONSTRAINT_TYPE
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
        WHERE TABLE_SCHEMA = DATABASE()
        AND CONSTRAINT_TYPE IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE');
        
    DECLARE idx_cur CURSOR FOR
        SELECT TABLE_NAME, INDEX_NAME 
        FROM INFORMATION_SCHEMA.STATISTICS 
        WHERE TABLE_SCHEMA = DATABASE()
        AND INDEX_NAME NOT IN (
            SELECT CONSTRAINT_NAME 
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
            WHERE TABLE_SCHEMA = DATABASE()
        )
        AND INDEX_NAME != 'PRIMARY'
        GROUP BY TABLE_NAME, INDEX_NAME;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
        
    SET FOREIGN_KEY_CHECKS = 0;
        
    OPEN cur;
    
    constraints_loop: LOOP
        FETCH cur INTO TNAME, CNAME, CONSTRAINT_TYPE;
        IF done THEN
            LEAVE constraints_loop;
        END IF;
        
        CASE CONSTRAINT_TYPE
            WHEN 'FOREIGN KEY' THEN
                SET @sql = CONCAT('ALTER TABLE ', TNAME, ' DROP FOREIGN KEY ', CNAME);
                SELECT CONCAT('Dropping foreign key: ', CNAME, ' from table: ', TNAME) AS debug_message;
            
            WHEN 'PRIMARY KEY' THEN
                SET @sql = CONCAT('ALTER TABLE ', TNAME, ' DROP PRIMARY KEY');
                SELECT CONCAT('Dropping primary key from table: ', TNAME) AS debug_message;
            
            WHEN 'UNIQUE' THEN
                SET @sql = CONCAT('ALTER TABLE ', TNAME, ' DROP INDEX ', CNAME);
                SELECT CONCAT('Dropping unique constraint: ', CNAME, ' from table: ', TNAME) AS debug_message;
        END CASE;
        
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END LOOP;
   
    CLOSE cur;
      
    SET done = 0;    
    
    OPEN idx_cur;
    
    indexes_loop: LOOP
        FETCH idx_cur INTO TNAME, CNAME;
        IF done THEN
            LEAVE indexes_loop;
        END IF;
        
        SET @sql = CONCAT('ALTER TABLE ', TNAME, ' DROP INDEX ', CNAME);
        SELECT CONCAT('Dropping index: ', CNAME, ' from table: ', TNAME) AS debug_message;
        
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END LOOP;
    
    CLOSE idx_cur;    
    
    SET FOREIGN_KEY_CHECKS = 1;
    
    SELECT 'All keys and indexes have been removed' AS debug_message;
END