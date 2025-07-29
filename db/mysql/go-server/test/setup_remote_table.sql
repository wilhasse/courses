-- Run this on the remote MySQL server (10.1.0.7) to create test data
-- mysql -h 10.1.0.7 -u root testdb < setup_remote_table.sql

-- Create employees table
CREATE TABLE IF NOT EXISTS employees (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    department VARCHAR(50),
    hire_date DATE,
    salary DECIMAL(10,2)
);

-- Insert sample employee data
INSERT INTO employees (name, email, department, hire_date, salary) VALUES
('John Doe', 'john.doe@company.com', 'Engineering', '2020-01-15', 75000.00),
('Jane Smith', 'jane.smith@company.com', 'Marketing', '2019-03-22', 65000.00),
('Bob Johnson', 'bob.johnson@company.com', 'Sales', '2021-06-10', 55000.00),
('Alice Brown', 'alice.brown@company.com', 'Engineering', '2020-09-01', 80000.00),
('Charlie Wilson', 'charlie.wilson@company.com', 'HR', '2018-11-30', 60000.00),
('Diana Martinez', 'diana.martinez@company.com', 'Finance', '2022-02-14', 70000.00),
('Eric Davis', 'eric.davis@company.com', 'Engineering', '2021-07-20', 72000.00),
('Fiona Garcia', 'fiona.garcia@company.com', 'Marketing', '2020-04-05', 63000.00),
('George Lee', 'george.lee@company.com', 'Sales', '2019-08-18', 58000.00),
('Helen Anderson', 'helen.anderson@company.com', 'Engineering', '2022-01-10', 85000.00);

-- Show the data
SELECT * FROM employees;