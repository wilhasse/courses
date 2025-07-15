#!/bin/bash

echo "=== Testing chDB JSON Functions ==="
echo

# Test 1: Basic JSON function
echo "1. Testing chdb_api_query_json function..."
mysql -u root -e "SELECT CAST(chdb_api_query_json('SELECT 1 as num, \"hello\" as msg') AS CHAR) AS json_result\G" 2>&1 | grep -v "Warning"

echo
echo "2. Testing JSON_TABLE with historico data..."
mysql -u root -e "
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('SELECT id_contr, seq, codigo FROM mysql_import.historico LIMIT 5') USING utf8mb4),
    '\$.data[*]' COLUMNS (
        id_contr INT PATH '\$.id_contr',
        seq INT PATH '\$.seq',
        codigo INT PATH '\$.codigo'
    )
) AS jt;" 2>&1 | grep -v "Warning"

echo
echo "3. Testing aggregation query..."
mysql -u root -e "
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT 
            id_contr,
            COUNT(*) as count,
            MIN(seq) as min_seq,
            MAX(seq) as max_seq
        FROM mysql_import.historico 
        GROUP BY id_contr
        ORDER BY id_contr
        LIMIT 5
    ') USING utf8mb4),
    '\$.data[*]' COLUMNS (
        id_contr INT PATH '\$.id_contr',
        count INT PATH '\$.count',
        min_seq INT PATH '\$.min_seq',
        max_seq INT PATH '\$.max_seq'
    )
) AS jt;" 2>&1 | grep -v "Warning"

echo
echo "=== All tests completed ==="