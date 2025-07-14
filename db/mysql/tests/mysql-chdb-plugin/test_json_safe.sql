-- Safe test for JSON functions with smaller buffer
USE teste;

-- Test 1: Simple test function
SELECT '=== Test 1: Simple test function ===' AS test;
SELECT chdb_test_json() AS json_result;

-- Test 2: Check if JSON is valid
SELECT '=== Test 2: JSON validation ===' AS test;
SELECT JSON_VALID(chdb_test_json()) AS is_valid_json;

-- Test 3: Try JSON_TABLE with test function
SELECT '=== Test 3: JSON_TABLE with test function ===' AS test;
SELECT * FROM JSON_TABLE(
    chdb_test_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        age INT PATH '$.age'
    )
) AS test_table;

-- Test 4: Try customers function (limited to 10 rows)
SELECT '=== Test 4: Limited customers function ===' AS test;
SELECT chdb_customers_json() AS customers_json;

-- Test 5: Check if customers JSON is valid
SELECT '=== Test 5: Customers JSON validation ===' AS test;
SELECT JSON_VALID(chdb_customers_json()) AS is_valid_json;

-- Test 6: Try JSON_TABLE with customers (if valid)
SELECT '=== Test 6: JSON_TABLE with customers ===' AS test;
SELECT * FROM JSON_TABLE(
    chdb_customers_json(),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name',
        city VARCHAR(100) PATH '$.city'
    )
) AS customers
LIMIT 3;