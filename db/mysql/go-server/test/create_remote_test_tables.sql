-- Create employee_notes table on remote server to test JOIN
-- Run this on 10.1.0.7:
-- mysql -h 10.1.0.7 -u root testdb < test/create_remote_test_tables.sql

CREATE TABLE IF NOT EXISTS employee_notes (
    note_id INT PRIMARY KEY AUTO_INCREMENT,
    emp_id INT NOT NULL,
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_emp_id (emp_id)
);

-- Insert some test notes for existing employees
INSERT INTO employee_notes (emp_id, note) VALUES
(1, 'Great performance this quarter'),
(2, 'Completed certification program'),
(3, 'Leading new project initiative'),
(1, 'Promoted to senior position'),
(4, 'Excellent teamwork'),
(5, 'Mentoring new team members'),
(2, 'Presented at company conference'),
(6, 'Improved process efficiency by 30%'),
(7, 'Received customer excellence award'),
(8, 'Completed advanced training'),
(1, 'Q3 goals exceeded'),
(3, 'Strong technical leadership'),
(9, 'Great collaboration skills'),
(10, 'Innovation award recipient');

-- Show the data
SELECT * FROM employee_notes;