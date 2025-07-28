-- Initial database setup for go-mysql-server with LMDB storage
-- This script creates the sample database and tables with data

-- Create the test database
CREATE DATABASE IF NOT EXISTS testdb;
USE testdb;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

-- Create products table  
CREATE TABLE IF NOT EXISTS products (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50)
);

-- Create orders table for demonstrating joins
CREATE TABLE IF NOT EXISTS orders (
    id INT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    order_date TIMESTAMP NOT NULL
);

-- Insert sample users
INSERT INTO users (id, name, email, created_at) VALUES (1, 'Alice', 'alice@example.com', '2023-01-01 00:00:00');
INSERT INTO users (id, name, email, created_at) VALUES (2, 'Bob', 'bob@example.com', '2023-01-02 00:00:00');
INSERT INTO users (id, name, email, created_at) VALUES (3, 'Charlie', 'charlie@example.com', '2023-01-03 10:00:00');
INSERT INTO users (id, name, email, created_at) VALUES (4, 'David', 'david@example.com', '2023-01-04 11:00:00');
INSERT INTO users (id, name, email, created_at) VALUES (5, 'Eve', 'eve@example.com', '2023-01-05 12:00:00');

-- Insert sample products
INSERT INTO products (id, name, price, category) VALUES (1, 'Laptop', 999.99, 'Electronics');
INSERT INTO products (id, name, price, category) VALUES (2, 'Book', 19.99, 'Education');
INSERT INTO products (id, name, price, category) VALUES (3, 'Coffee Mug', 12.50, 'Kitchen');
INSERT INTO products (id, name, price, category) VALUES (4, 'Smartphone', 699.99, 'Electronics');
INSERT INTO products (id, name, price, category) VALUES (5, 'Tablet', 399.99, 'Electronics');
INSERT INTO products (id, name, price, category) VALUES (6, 'Headphones', 149.99, 'Electronics');
INSERT INTO products (id, name, price, category) VALUES (7, 'Monitor', 299.99, 'Electronics');
INSERT INTO products (id, name, price, category) VALUES (8, 'Keyboard', 89.99, 'Electronics');

-- Insert sample orders
INSERT INTO orders (id, user_id, product_id, quantity, order_date) VALUES (1, 1, 1, 1, '2023-02-01 09:00:00');
INSERT INTO orders (id, user_id, product_id, quantity, order_date) VALUES (2, 1, 4, 1, '2023-02-01 09:05:00');
INSERT INTO orders (id, user_id, product_id, quantity, order_date) VALUES (3, 2, 6, 2, '2023-02-02 14:30:00');
INSERT INTO orders (id, user_id, product_id, quantity, order_date) VALUES (4, 3, 7, 1, '2023-02-03 16:15:00');
INSERT INTO orders (id, user_id, product_id, quantity, order_date) VALUES (5, 3, 8, 1, '2023-02-03 16:20:00');