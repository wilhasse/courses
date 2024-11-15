#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <IP> <LABEL> <DATA_DIR> [APPEND] [LOAD_METHOD]"
    echo "APPEND: 1 for no truncate data"
    echo "LOAD_METHOD: 1 for LOAD DATA LOCAL INFILE, 0 for pigz method (default)"
    exit 1
fi

# Assign the arguments to variables
IP=$1
LABEL="$2"
DIR="$3"
APPEND="${4:-0}"
LOAD_METHOD=${5:-1}  # Default to 1 LOAD DATA

# MySQL connection details for remote server (where we load data)
REMOTE_DATABASE="ssb"

REMOTE_USER="root"
REMOTE_PORT=3306

#REMOTE_USER="polardbx_root"
#REMOTE_PWD="xYXptQYe"
#REMOTE_PORT=50129

# MySQL connection details for local server (to store execution time)
LOCAL_USER="root"
LOCAL_DATABASE="ssb_results"

# Function to create MySQL connection string for remote server
mysql_connect_remote() {
    local connect_string="mysql -h $IP -u $REMOTE_USER"
    if [ -n "$REMOTE_PWD" ]; then
        connect_string+=" -p$REMOTE_PWD"
    fi
    connect_string+=" -P$REMOTE_PORT $REMOTE_DATABASE --local-infile=1"
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

    if [ "$APPEND" -eq 0 ]; then
        echo "Truncating data in $table..."
        $(mysql_connect_remote) -e "SET FOREIGN_KEY_CHECKS=0;TRUNCATE TABLE $table;"
    fi

    echo "Loading data into $table table..."
    if [ "$LOAD_METHOD" -eq 1 ]; then
        echo "Using LOAD DATA LOCAL INFILE method..."
        $(mysql_connect_remote) -e "SET autocommit = 0;SET UNIQUE_CHECKS = 0;SET FOREIGN_KEY_CHECKS = 0;LOAD DATA LOCAL INFILE '${DIR}/${table}.tbl' INTO TABLE $table FIELDS TERMINATED BY '|' LINES TERMINATED BY '\n';commit"
    else
        echo "Using pigz method..."
        pigz -c -d ${DIR}/${table}.sql.gz | $(mysql_connect_remote)
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
