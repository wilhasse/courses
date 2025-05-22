CREATE DEFINER=`root`@`localhost` PROCEDURE `drop_view`(
	IN `p_schema` VARCHAR(64),
	IN `p_view` VARCHAR(64)
)
LANGUAGE SQL
NOT DETERMINISTIC
CONTAINS SQL
SQL SECURITY DEFINER
COMMENT ''
BEGIN
    DECLARE v_type VARCHAR(16);
    DECLARE v_error_msg VARCHAR(255);
    
    
    SELECT TABLE_TYPE
      INTO v_type
      FROM INFORMATION_SCHEMA.TABLES
     WHERE TABLE_SCHEMA = p_schema
       AND TABLE_NAME   = p_view;
       
    IF v_type IS NULL THEN
        SET v_error_msg = CONCAT('Objeto ', p_schema, '.', p_view, ' não encontrado');
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = v_error_msg;
    ELSEIF v_type <> 'VIEW' THEN
        SET v_error_msg = CONCAT(p_schema, '.', p_view, ' não é uma VIEW');
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = v_error_msg;
    END IF;
    
    
    SET @stmt = CONCAT('DROP VIEW ', p_schema, '.', p_view);
    PREPARE stmt1 FROM @stmt;
    EXECUTE stmt1;
    DEALLOCATE PREPARE stmt1;
END