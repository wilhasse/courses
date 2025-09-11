DROP PROCEDURE IF EXISTS analyze_db;
DELIMITER $$

CREATE DEFINER=`root`@`localhost` PROCEDURE analyze_db()
BEGIN
  DECLARE done INT DEFAULT 0;
  DECLARE tname VARCHAR(255);

  DECLARE has_err INT DEFAULT 0;
  DECLARE err_msg TEXT;

  -- cursor: só tabelas (sem views), maiores primeiro
  DECLARE c CURSOR FOR
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = DATABASE()
      AND table_type = 'BASE TABLE'
    ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC;

  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

  -- captura erro do ANALYZE pra imprimir "ERROR: <msg>"
  DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
  BEGIN
    SET has_err = 1;
    GET DIAGNOSTICS CONDITION 1 err_msg = MESSAGE_TEXT;
  END;

  OPEN c;
  read_loop: LOOP
    FETCH c INTO tname;
    IF done = 1 THEN
      LEAVE read_loop;
    END IF;

    SET @sql_txt = CONCAT(
      'ANALYZE NO_WRITE_TO_BINLOG TABLE `',
      REPLACE(DATABASE(), '`','``'),
      '`.`',
      REPLACE(tname,'`','``'),
      '`'
    );

    PREPARE stmt FROM @sql_txt;
    EXECUTE stmt;           -- inevitavelmente retorna um pequeno resultset
    DEALLOCATE PREPARE stmt;

    IF has_err = 1 THEN
      SELECT CONCAT('Analyzing ', tname, ' ... ERROR: ', err_msg) AS status;
      SET has_err = 0; SET err_msg = NULL;
    ELSE
      SELECT CONCAT('Analyzing ', tname, ' ... OK') AS status;
    END IF;
  END LOOP;

  CLOSE c;
END$$

DELIMITER ;
