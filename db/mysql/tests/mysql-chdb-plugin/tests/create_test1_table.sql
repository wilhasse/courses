-- Create TEST1 table with sample data

DROP TABLE IF EXISTS TEST1;

CREATE TABLE TEST1 (
    id INT PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    created_date DATE NOT NULL
);

-- Insert sample data
INSERT INTO TEST1 (id, category, amount, created_date) VALUES
(1, 'Electronics', 599.99, '2024-01-15'),
(2, 'Books', 45.50, '2024-01-20'),
(3, 'Clothing', 120.00, '2024-02-01'),
(4, 'Food', 85.25, '2024-02-10'),
(5, 'Electronics', 1299.00, '2024-02-15');

-- Verify data
SELECT * FROM TEST1;