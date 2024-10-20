#!/bin/bash

# Check if IP address is provided
if [ $# -eq 0 ]; then
    echo "Please provide an IP address as an argument."
    exit 1
fi

# Assign the first argument to the IP variable
IP=$1

# MySQL connection details for remote server (where we load data)
REMOTE_USER="root"
REMOTE_DATABASE="ssb"

# MySQL connection details for local server (to store execution time)
LOCAL_USER="root"
LOCAL_DATABASE="ssb_results"

# Function to create MySQL connection string for remote server
mysql_connect_remote() {
    local connect_string="mysql -h $IP -u $REMOTE_USER"
    if [ -n "$MYSQL_PWD" ]; then
        connect_string+=" -p$MYSQL_PWD"
    fi
    connect_string+=" $REMOTE_DATABASE --local-infile=1"
    echo "$connect_string"
}

# Function to create MySQL connection string for local server
mysql_connect_local() {
    local connect_string="mysql -u $LOCAL_USER"
    if [ -n "$LOCAL_MYSQL_PWD" ]; then
        connect_string+=" -p$LOCAL_MYSQL_PWD"
    fi
    connect_string+=" $LOCAL_DATABASE"
    echo "$connect_string"
}

# Create table to store execution time if it doesn't exist
$(mysql_connect_local) -e "
CREATE TABLE IF NOT EXISTS query_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    remote_ip VARCHAR(15),
    query_id INT,
    execution_time INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"

# Function to run MySQL command for data loading
run_mysql_command() {
    local file=$1
    local table=$2

    # Load table
    echo "Truncating and loading data into $table table..."
    $(mysql_connect_remote) -e "SET FOREIGN_KEY_CHECKS=0;TRUNCATE TABLE $table;"
    $(mysql_connect_remote) -e "LOAD DATA LOCAL INFILE '~/ssb/data/$file' INTO TABLE $table FIELDS TERMINATED BY '|' LINES TERMINATED BY '\n';"
    echo "----------------------------------------"
}

# Start timing the entire process
total_start_time=$(date +%s)

# Run commands
run_mysql_command "supplier.tbl" "supplier"
run_mysql_command "customer.tbl" "customer"
run_mysql_command "date.tbl" "date"
run_mysql_command "lineorder.tbl" "lineorder"

# Calculate total execution time
total_end_time=$(date +%s)
total_execution_time=$((total_end_time - total_start_time))
echo "Total execution time: $total_execution_time seconds"

# Save total execution time to local database as query_id 0
$(mysql_connect_local) -e "
INSERT INTO query_performance (remote_ip, query_id, execution_time) 
VALUES ('$IP', 0, $total_execution_time);"

echo "Data loading completed. Total load time saved to local database."
