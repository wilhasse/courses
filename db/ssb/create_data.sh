#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <IP> <CREATE>"
    exit 1
fi

# Assign the arguments to variables
IP=$1
CREATE="$2"

# MySQL connection details for remote server (where we load data)
REMOTE_USER="root"

# Function to create MySQL connection string for remote server
mysql_connect_remote() {
    local connect_string="mysql -h $IP -u $REMOTE_USER"
    if [ -n "$MYSQL_PWD" ]; then
        connect_string+=" -p$MYSQL_PWD"
    fi
    echo "$connect_string"
}

# Function to run MySQL command for data loading
run_mysql_command() {

    echo "Drop database..."
    $(mysql_connect_remote) -e "DROP DATABASE ssb"

    echo "Create database..."
    $(mysql_connect_remote) -e "CREATE DATABASE ssb"

    echo "Create tables create_table_${CREATE}.sql"
       cat create_table_${CREATE}.sql | $(mysql_connect_remote) ssb

    echo "----------------------------------------"
}

# Start timing the entire process
total_start_time=$(date +%s)

# Run commands
run_mysql_command

# Calculate total execution time
total_end_time=$(date +%s)
total_execution_time=$(echo "$total_end_time - $total_start_time" | bc)
echo "Total execution time: $total_execution_time seconds"
