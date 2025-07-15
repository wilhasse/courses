-- Examples of using chdb_api_query_json with JSON_TABLE

-- 1. Basic JSON query
SELECT chdb_api_query_json('SELECT 1 AS num, "hello" AS msg');

-- 2. Query historico table and parse with JSON_TABLE
-- Note: We need to CONVERT the result to UTF8 because UDFs return binary strings
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('SELECT ID_CONTR, SEQ, CODIGO FROM mysql_import.historico LIMIT 10') USING utf8mb4),
    '$.data[*]' COLUMNS (
        ID_CONTR INT PATH '$.ID_CONTR',
        SEQ INT PATH '$.SEQ',
        CODIGO INT PATH '$.CODIGO'
    )
) AS jt;

-- 3. More complex example with filtering
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT 
            ID_CONTR,
            SEQ,
            CODIGO,
            COUNT(*) OVER (PARTITION BY ID_CONTR) as total_per_contr
        FROM mysql_import.historico 
        WHERE ID_CONTR < 5
        LIMIT 20
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        ID_CONTR INT PATH '$.ID_CONTR',
        SEQ INT PATH '$.SEQ',
        CODIGO INT PATH '$.CODIGO',
        total_per_contr INT PATH '$.total_per_contr'
    )
) AS jt;

-- 4. Join ClickHouse data with MySQL table
-- First, let's create a sample MySQL table
CREATE TABLE IF NOT EXISTS mysql_metadata (
    id_contr INT PRIMARY KEY,
    description VARCHAR(100)
);

INSERT IGNORE INTO mysql_metadata VALUES 
    (0, 'Contract Zero'),
    (1, 'Contract One'),
    (2, 'Contract Two');

-- Now join ClickHouse data with MySQL data
SELECT 
    jt.ID_CONTR,
    jt.SEQ,
    jt.CODIGO,
    m.description
FROM JSON_TABLE(
    chdb_api_query_json('
        SELECT ID_CONTR, SEQ, CODIGO 
        FROM mysql_import.historico 
        WHERE ID_CONTR <= 2
        LIMIT 10
    '),
    '$.data[*]' COLUMNS (
        ID_CONTR INT PATH '$.ID_CONTR',
        SEQ INT PATH '$.SEQ',
        CODIGO INT PATH '$.CODIGO'
    )
) AS jt
LEFT JOIN mysql_metadata m ON jt.ID_CONTR = m.id_contr;

-- 5. Aggregate ClickHouse data
SELECT 
    ID_CONTR,
    COUNT(*) as row_count,
    GROUP_CONCAT(DISTINCT CODIGO) as unique_codigos
FROM JSON_TABLE(
    chdb_api_query_json('
        SELECT ID_CONTR, SEQ, CODIGO 
        FROM mysql_import.historico 
        LIMIT 100
    '),
    '$.data[*]' COLUMNS (
        ID_CONTR INT PATH '$.ID_CONTR',
        SEQ INT PATH '$.SEQ',
        CODIGO INT PATH '$.CODIGO'
    )
) AS jt
GROUP BY ID_CONTR;

-- Clean up
DROP TABLE IF EXISTS mysql_metadata;