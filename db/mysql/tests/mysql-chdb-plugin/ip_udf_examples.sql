-- Examples of using configurable chDB API functions

-- ========== Basic Usage ==========

-- 1. Query localhost (when API server is on same machine)
SELECT CAST(chdb_query_local('SELECT version()') AS CHAR) AS version;

-- 2. Query remote server by IP
SELECT CAST(chdb_query_remote('192.168.1.100:8125', 'SELECT version()') AS CHAR) AS version;

-- 3. Query remote server by hostname
SELECT CAST(chdb_query_remote('dbserver.company.local:8125', 'SELECT version()') AS CHAR) AS version;

-- 4. Use default port (8125)
SELECT CAST(chdb_query_remote('192.168.1.100', 'SELECT version()') AS CHAR) AS version;

-- ========== JSON Format for Table Results ==========

-- 5. Local server with JSON format
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_query_json_local('SELECT id_contr, seq, codigo FROM mysql_import.historico LIMIT 10') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq',
        codigo INT PATH '$.codigo'
    )
) AS jt;

-- 6. Remote server with JSON format
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_query_json_remote('192.168.1.100:8125', '
        SELECT id_contr, seq, codigo 
        FROM mysql_import.historico 
        WHERE codigo = 22 
        LIMIT 100
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq',
        codigo INT PATH '$.codigo'
    )
) AS jt;

-- ========== Advanced Examples ==========

-- 7. Create a VIEW that queries remote server
CREATE OR REPLACE VIEW v_remote_historico AS
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_query_json_remote('192.168.1.100:8125', 
        'SELECT id_contr, seq, codigo FROM mysql_import.historico LIMIT 1000'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq',
        codigo INT PATH '$.codigo'
    )
) AS jt;

-- 8. Query multiple servers and UNION results
SELECT 'Server1' as source, jt.*
FROM JSON_TABLE(
    CONVERT(chdb_query_json_remote('server1.local:8125', 
        'SELECT COUNT(*) as count FROM mysql_import.historico'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (count INT PATH '$.count')
) AS jt
UNION ALL
SELECT 'Server2' as source, jt.*
FROM JSON_TABLE(
    CONVERT(chdb_query_json_remote('server2.local:8125', 
        'SELECT COUNT(*) as count FROM mysql_import.historico'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (count INT PATH '$.count')
) AS jt;

-- 9. Stored procedure to query configurable server
DELIMITER $$

DROP PROCEDURE IF EXISTS query_remote_historico$$

CREATE PROCEDURE query_remote_historico(
    IN server_address VARCHAR(255),
    IN filter_codigo INT
)
BEGIN
    SET @query = CONCAT(
        'SELECT jt.* FROM JSON_TABLE(',
        'CONVERT(chdb_query_json_remote(''', server_address, ''', ',
        '''SELECT id_contr, seq, codigo FROM mysql_import.historico ',
        'WHERE codigo = ', filter_codigo, ' LIMIT 100'') USING utf8mb4), ',
        '''$.data[*]'' COLUMNS (',
        'id_contr INT PATH ''$.id_contr'', ',
        'seq INT PATH ''$.seq'', ',
        'codigo INT PATH ''$.codigo'')) AS jt'
    );
    
    PREPARE stmt FROM @query;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END$$

DELIMITER ;

-- Usage:
-- CALL query_remote_historico('192.168.1.100:8125', 22);
-- CALL query_remote_historico('server2.local:9000', 767);

-- 10. Function to check server connectivity
DELIMITER $$

DROP FUNCTION IF EXISTS check_chdb_server$$

CREATE FUNCTION check_chdb_server(server_address VARCHAR(255))
RETURNS VARCHAR(100)
DETERMINISTIC
BEGIN
    DECLARE result VARCHAR(100);
    DECLARE version_str VARCHAR(255);
    
    -- Try to get version from server
    SET version_str = CAST(chdb_query_remote(server_address, 'SELECT version()') AS CHAR);
    
    IF version_str LIKE 'ERROR:%' THEN
        SET result = CONCAT('Server ', server_address, ' is NOT reachable');
    ELSE
        SET result = CONCAT('Server ', server_address, ' is OK: ', version_str);
    END IF;
    
    RETURN result;
END$$

DELIMITER ;

-- Usage:
-- SELECT check_chdb_server('192.168.1.100:8125');
-- SELECT check_chdb_server('localhost:8125');

-- Clean up examples
-- DROP VIEW IF EXISTS v_remote_historico;
-- DROP PROCEDURE IF EXISTS query_remote_historico;
-- DROP FUNCTION IF EXISTS check_chdb_server;