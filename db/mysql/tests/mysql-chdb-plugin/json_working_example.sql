-- Working examples using chdb_api_query_json with JSON_TABLE
-- Key: Use CONVERT(...USING utf8mb4) to convert binary to UTF-8

-- 1. Simple test - view the raw JSON
SELECT CAST(chdb_api_query_json('SELECT 1 as num, "hello" as msg') AS CHAR);

-- 2. Basic JSON_TABLE example with historico
-- IMPORTANT: Column names in ClickHouse are lowercase!
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('SELECT id_contr, seq, codigo FROM mysql_import.historico LIMIT 10') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq', 
        codigo INT PATH '$.codigo'
    )
) AS jt;

-- 3. With WHERE clause and ORDER BY
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT ID_CONTR, SEQ, CODIGO 
        FROM mysql_import.historico 
        WHERE ID_CONTR = 1 
        ORDER BY SEQ 
        LIMIT 10
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        ID_CONTR INT PATH '$.ID_CONTR',
        SEQ INT PATH '$.SEQ',
        CODIGO INT PATH '$.CODIGO'
    )
) AS jt;

-- 4. Aggregate in ClickHouse, display in MySQL
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT 
            ID_CONTR,
            COUNT(*) as record_count,
            MIN(SEQ) as min_seq,
            MAX(SEQ) as max_seq
        FROM mysql_import.historico 
        GROUP BY ID_CONTR
        ORDER BY ID_CONTR
        LIMIT 10
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        ID_CONTR INT PATH '$.ID_CONTR',
        record_count INT PATH '$.record_count',
        min_seq INT PATH '$.min_seq',
        max_seq INT PATH '$.max_seq'
    )
) AS jt;

-- 5. Create a VIEW for easy access
CREATE OR REPLACE VIEW v_historico_sample AS
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('SELECT ID_CONTR, SEQ, CODIGO FROM mysql_import.historico LIMIT 100') USING utf8mb4),
    '$.data[*]' COLUMNS (
        ID_CONTR INT PATH '$.ID_CONTR',
        SEQ INT PATH '$.SEQ',
        CODIGO INT PATH '$.CODIGO'
    )
) AS jt;

-- Now you can query the view like a regular table
SELECT * FROM v_historico_sample WHERE ID_CONTR = 1;

-- Clean up
DROP VIEW IF EXISTS v_historico_sample;