#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <IP> <LABEL> [LOAD_METHOD]"
    echo "LOAD_METHOD: 1 for LOAD DATA LOCAL INFILE, 0 for pigz method (default)"
    exit 1
fi

# Assign the arguments to variables
IP=$1
LABEL="$2"
LOAD_METHOD=${3:-0}  # Default to 0 if not provided

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
    id INT NOT NULL AUTO_INCREMENT,
    query_id INT,
    label VARCHAR(200),
    remote_ip VARCHAR(15),
    execution_time FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id) USING BTREE
);"

# Function to run MySQL command for data loading
run_mysql_command() {
    local table=$1
    echo "Truncating and loading data into $table table..."
    $(mysql_connect_remote) -e "SET FOREIGN_KEY_CHECKS=0;TRUNCATE TABLE $table;"

    if [ "$LOAD_METHOD" -eq 1 ]; then
        echo "Using LOAD DATA LOCAL INFILE method..."
        $(mysql_connect_remote) -e "LOAD DATA LOCAL INFILE '~/ssb/data/${table}.tbl' INTO TABLE $table FIELDS TERMINATED BY '|' LINES TERMINATED BY '\n';"
    else
        echo "Using pigz method..."
        pigz -c -d ~/ssb/data/${table}.sql.gz | $(mysql_connect_remote)
    fi

    echo "----------------------------------------"
}

# Start timing the entire process
total_start_time=$(date +%s)

# Run commands
run_mysql_command "supplier"
run_mysql_command "customer"
run_mysql_command "date"
run_mysql_command "lineorder"

# Calculate total execution time
total_end_time=$(date +%s)
total_execution_time=$(echo "$total_end_time - $total_start_time" | bc)
echo "Total execution time: $total_execution_time seconds"

# Save total execution time to local database as query_id 0
$(mysql_connect_local) -e "
INSERT INTO query_performance (remote_ip, label, query_id, execution_time)
VALUES ('$IP', '$LABEL', 0, $total_execution_time);"

echo "Data loading completed. Total load time saved to local database."
