-- Create database
CREATE DATABASE IF NOT EXISTS sample_db;
USE sample_db;

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INT,
    city VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    product_name VARCHAR(100),
    quantity INT,
    price DECIMAL(10,2),
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- Insert sample data into customers
INSERT INTO customers (name, email, age, city) VALUES
    ('John Doe', 'john@example.com', 35, 'New York'),
    ('Jane Smith', 'jane@example.com', 28, 'Los Angeles'),
    ('Bob Johnson', 'bob@example.com', 42, 'Chicago'),
    ('Alice Brown', 'alice@example.com', 31, 'Houston'),
    ('Charlie Wilson', 'charlie@example.com', 45, 'Phoenix'),
    ('Diana Lee', 'diana@example.com', 29, 'San Antonio'),
    ('Edward Davis', 'edward@example.com', 38, 'San Diego'),
    ('Fiona Garcia', 'fiona@example.com', 33, 'Dallas'),
    ('George Miller', 'george@example.com', 50, 'San Jose'),
    ('Helen Martinez', 'helen@example.com', 27, 'Austin');

-- Insert sample data into orders
INSERT INTO orders (customer_id, product_name, quantity, price, order_date) VALUES
    (1, 'Laptop', 1, 1200.00, '2024-01-15'),
    (1, 'Mouse', 2, 25.00, '2024-01-15'),
    (2, 'Keyboard', 1, 80.00, '2024-01-20'),
    (3, 'Monitor', 2, 300.00, '2024-01-22'),
    (4, 'Webcam', 1, 120.00, '2024-01-25'),
    (5, 'Headphones', 1, 150.00, '2024-02-01'),
    (2, 'USB Hub', 1, 40.00, '2024-02-05'),
    (6, 'Laptop', 1, 1500.00, '2024-02-10'),
    (7, 'Mouse Pad', 3, 15.00, '2024-02-12'),
    (8, 'External SSD', 1, 200.00, '2024-02-15'),
    (9, 'Graphics Card', 1, 800.00, '2024-02-18'),
    (10, 'RAM Module', 2, 120.00, '2024-02-20'),
    (1, 'Power Supply', 1, 150.00, '2024-02-22'),
    (3, 'CPU Cooler', 1, 80.00, '2024-02-25'),
    (5, 'Motherboard', 1, 350.00, '2024-03-01');