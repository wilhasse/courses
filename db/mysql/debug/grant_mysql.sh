#!/bin/bash

# MySQL socket path
MYSQL_SOCK="/var/run/mysqld/mysqld.sock"

# Check if socket file exists
if [ ! -S "$MYSQL_SOCK" ]; then
    echo "Error: MySQL socket file not found at $MYSQL_SOCK"
    exit 1
fi

# SQL commands to be executed
cat << 'EOF' > /tmp/mysql_setup.sql
CREATE USER 'root'@'%';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
EOF

# Execute the SQL commands
mysql -u root -S "$MYSQL_SOCK" < /tmp/mysql_setup.sql

# Check if the commands were executed successfully
if [ $? -eq 0 ]; then
    echo "MySQL user setup completed successfully"
else
    echo "Error: Failed to execute MySQL commands"
fi

# Clean up temporary file
rm /tmp/mysql_setup.sql
