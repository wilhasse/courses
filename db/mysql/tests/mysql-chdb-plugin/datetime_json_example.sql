-- Examples of handling datetime columns with JSON_TABLE

-- 1. Basic example with datetime column
SELECT jt.* 
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('172.16.120.14:8125', 
        'SELECT DISTINCT id_contr, data FROM mysql_import.historico WHERE codigo=22 LIMIT 100'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        data DATETIME PATH '$.data'
    )
) AS jt;

-- 2. With date formatting in ClickHouse
SELECT jt.* 
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('172.16.120.14:8125', 
        'SELECT DISTINCT 
            id_contr, 
            data,
            formatDateTime(data, ''%Y-%m-%d'') as date_only,
            formatDateTime(data, ''%H:%i:%s'') as time_only
        FROM mysql_import.historico 
        WHERE codigo=22 
        LIMIT 100'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        data DATETIME PATH '$.data',
        date_only DATE PATH '$.date_only',
        time_only TIME PATH '$.time_only'
    )
) AS jt;

-- 3. With date calculations in ClickHouse
SELECT jt.* 
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('172.16.120.14:8125', 
        'SELECT DISTINCT 
            id_contr,
            data,
            toYear(data) as year,
            toMonth(data) as month,
            toDayOfWeek(data) as day_of_week,
            date_diff(''day'', data, today()) as days_ago
        FROM mysql_import.historico 
        WHERE codigo=22 
        LIMIT 100'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        data DATETIME PATH '$.data',
        year INT PATH '$.year',
        month INT PATH '$.month',
        day_of_week INT PATH '$.day_of_week',
        days_ago INT PATH '$.days_ago'
    )
) AS jt;

-- 4. Filter by date range in ClickHouse
SELECT jt.* 
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('172.16.120.14:8125', 
        'SELECT DISTINCT id_contr, data, codigo
        FROM mysql_import.historico 
        WHERE codigo=22 
          AND data >= ''2023-01-01''
          AND data < ''2024-01-01''
        ORDER BY data DESC
        LIMIT 100'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        data DATETIME PATH '$.data',
        codigo INT PATH '$.codigo'
    )
) AS jt;

-- 5. Group by date parts
SELECT jt.* 
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('172.16.120.14:8125', 
        'SELECT 
            toYYYYMM(data) as year_month,
            codigo,
            COUNT(DISTINCT id_contr) as unique_contracts,
            COUNT(*) as total_records
        FROM mysql_import.historico 
        WHERE codigo IN (22, 767)
        GROUP BY year_month, codigo
        ORDER BY year_month DESC, codigo
        LIMIT 100'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        year_month INT PATH '$.year_month',
        codigo INT PATH '$.codigo',
        unique_contracts INT PATH '$.unique_contracts',
        total_records INT PATH '$.total_records'
    )
) AS jt;

-- 6. Working with timestamps (if your data has timestamp precision)
SELECT jt.* 
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('172.16.120.14:8125', 
        'SELECT 
            id_contr,
            data,
            toString(data) as data_string,
            toUnixTimestamp(data) as unix_timestamp
        FROM mysql_import.historico 
        WHERE codigo=22 
        LIMIT 10'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        data DATETIME PATH '$.data',
        data_string VARCHAR(30) PATH '$.data_string',
        unix_timestamp BIGINT PATH '$.unix_timestamp'
    )
) AS jt;

-- 7. Join with MySQL data using the datetime
CREATE TEMPORARY TABLE IF NOT EXISTS temp_date_ranges (
    date_from DATE,
    date_to DATE,
    period_name VARCHAR(50)
);

INSERT INTO temp_date_ranges VALUES
    ('2023-01-01', '2023-03-31', 'Q1 2023'),
    ('2023-04-01', '2023-06-30', 'Q2 2023'),
    ('2023-07-01', '2023-09-30', 'Q3 2023'),
    ('2023-10-01', '2023-12-31', 'Q4 2023');

SELECT 
    jt.*,
    dr.period_name
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json_remote('172.16.120.14:8125', 
        'SELECT id_contr, data, codigo
        FROM mysql_import.historico 
        WHERE codigo=22 
        LIMIT 100'
    ) USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        data DATE PATH '$.data',
        codigo INT PATH '$.codigo'
    )
) AS jt
LEFT JOIN temp_date_ranges dr 
    ON jt.data BETWEEN dr.date_from AND dr.date_to;

DROP TEMPORARY TABLE IF EXISTS temp_date_ranges;

-- Note: JSON_TABLE datetime column types:
-- DATETIME - for full datetime values (YYYY-MM-DD HH:MM:SS)
-- DATE - for date only (YYYY-MM-DD)
-- TIME - for time only (HH:MM:SS)
-- TIMESTAMP - for timestamp values
-- VARCHAR(n) - to keep as string if needed