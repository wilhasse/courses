-- Simple test for JSON functions
USE teste;

-- Test 1: Check if the function returns data
SELECT '=== Test 1: Basic function test ===' AS test;
SELECT chdb_customers_json() AS json_result;

-- Test 2: Check if JSON is valid
SELECT '=== Test 2: JSON validation ===' AS test;
SELECT JSON_VALID(chdb_customers_json()) AS is_valid_json;

-- Test 3: Try to use CAST to ensure proper string encoding
SELECT '=== Test 3: CAST test ===' AS test;
SELECT CAST(chdb_customers_json() AS CHAR CHARACTER SET utf8mb4) AS cast_json;

-- Test 4: Try JSON_TABLE with CAST
SELECT '=== Test 4: JSON_TABLE with CAST ===' AS test;
SELECT * FROM JSON_TABLE(
    CAST(chdb_customers_json() AS CHAR CHARACTER SET utf8mb4),
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name'
    )
) AS customers
LIMIT 3;

-- Test 5: Setup MySQL table for JOIN testing
SELECT '=== Test 5: Setting up MySQL orders table ===' AS test;
CREATE TABLE IF NOT EXISTS customer_orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    status VARCHAR(50)
);

-- Insert sample orders (with ON DUPLICATE KEY UPDATE for repeatability)
INSERT INTO customer_orders (customer_id, order_date, total_amount, status) VALUES
    (1, '2024-01-15', 299.99, 'completed'),
    (2, '2024-01-18', 150.00, 'pending'),
    (3, '2024-01-20', 789.50, 'completed'),
    (1, '2024-01-22', 99.99, 'shipped'),
    (4, '2024-01-25', 450.00, 'completed'),
    (2, '2024-01-28', 200.25, 'cancelled')
ON DUPLICATE KEY UPDATE customer_id=VALUES(customer_id);

SELECT 'Orders table populated with sample data' AS status;

-- Test 6: JOIN MySQL table with ClickHouse data via JSON_TABLE
SELECT '=== Test 6: JOIN MySQL orders with ClickHouse customers ===' AS test;
SELECT 
    o.id,
    o.customer_id,
    o.order_date,
    o.total_amount,
    o.status,
    ch.name,
    ch.email,
    ch.city
FROM customer_orders o
JOIN (
    SELECT * FROM JSON_TABLE(
        CAST(chdb_customers_json() AS CHAR CHARACTER SET utf8mb4),
        '$[*]' COLUMNS (
            id INT PATH '$.id',
            name VARCHAR(100) PATH '$.name',
            email VARCHAR(100) PATH '$.email',
            city VARCHAR(100) PATH '$.city'
        )
    ) AS customers
) ch ON o.customer_id = ch.id
ORDER BY o.order_date;

-- Test 7: Aggregation query - Customer order summaries
SELECT '=== Test 7: Customer order summaries (MySQL + ClickHouse) ===' AS test;
SELECT 
    ch.name,
    ch.city,
    COUNT(o.id) AS total_orders,
    SUM(o.total_amount) AS total_spent,
    AVG(o.total_amount) AS avg_order_value
FROM customer_orders o
JOIN (
    SELECT * FROM JSON_TABLE(
        CAST(chdb_customers_json() AS CHAR CHARACTER SET utf8mb4),
        '$[*]' COLUMNS (
            id INT PATH '$.id',
            name VARCHAR(100) PATH '$.name',
            city VARCHAR(100) PATH '$.city'
        )
    ) AS customers
) ch ON o.customer_id = ch.id
WHERE o.status = 'completed'
GROUP BY ch.id, ch.name, ch.city
ORDER BY total_spent DESC;

-- Test 8: LEFT JOIN to show all customers including those without orders
SELECT '=== Test 8: All customers with their order counts ===' AS test;
SELECT 
    ch.id,
    ch.name,
    ch.city,
    COALESCE(order_stats.order_count, 0) AS order_count,
    COALESCE(order_stats.total_spent, 0) AS total_spent
FROM (
    SELECT * FROM JSON_TABLE(
        CAST(chdb_customers_json() AS CHAR CHARACTER SET utf8mb4),
        '$[*]' COLUMNS (
            id INT PATH '$.id',
            name VARCHAR(100) PATH '$.name',
            city VARCHAR(100) PATH '$.city'
        )
    ) AS customers
) ch
LEFT JOIN (
    SELECT 
        customer_id,
        COUNT(*) AS order_count,
        SUM(total_amount) AS total_spent
    FROM customer_orders
    GROUP BY customer_id
) order_stats ON ch.id = order_stats.customer_id
ORDER BY ch.id;
