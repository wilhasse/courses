#!/bin/bash

# Check if IP address is provided
if [ $# -eq 0 ]; then
    echo "Please provide an IP address as an argument."
    exit 1
fi

# Assign the first argument to the IP variable
IP=$1

# MySQL connection details
USER="root"
PASSWORD=""
DATABASE="ssb"

# Function to run MySQL command and measure execution time
run_mysql_command() {
    local file=$1
    local table=$2

    echo "Loading data into $table table..."
    start_time=$(date +%s)

    mysql -h "$IP" -u "$USER" -p"$PASSWORD" "$DATABASE" --local-infile=1 -e "LOAD DATA LOCAL INFILE '~/ssb/data/$file' INTO TABLE $table FIELDS TERMINATED BY '|' LINES TERMINATED BY '\n';"

    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Time taken to load $table: $execution_time seconds"
    echo "----------------------------------------"
}

# Start timing the entire process
total_start_time=$(date +%s)

# Run commands
run_mysql_command "supplier.tbl" "supplier"
run_mysql_command "customer.tbl" "customer"
run_mysql_command "date.tbl" "date"
run_mysql_command "lineorder.tbl" "lineorder"

# Calculate and display total execution time
total_end_time=$(date +%s)
total_execution_time=$((total_end_time - total_start_time))
echo "Total execution time: $total_execution_time seconds"

echo "Data loading completed."
