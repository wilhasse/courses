#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <IP> <OUTPUT_DIR>"
    echo "Example: $0 192.168.1.100 /path/to/output"
    exit 1
fi

# Assign arguments to variables
IP=$1
OUTPUT_DIR="$2"

# Database connection details
DB_NAME="ssb"
DB_USER="root"
DB_PORT=3306
#DB_USER="polardbx_root"
#DB_PASSWORD="xYXptQYe"
#DB_PORT=50129

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to create MySQL connection string
mysql_connect() {
    local connect_string="mysql -h $IP -u $DB_USER"
    if [ -n "$DB_PASSWORD" ]; then
        connect_string+=" -p$DB_PASSWORD"
    fi
    connect_string+=" -P$DB_PORT $DB_NAME"
    echo "$connect_string"
}

# Function to create mysqldump connection string
mysqldump_connect() {
    local connect_string="mysqldump -h $IP -u $DB_USER"
    if [ -n "$DB_PASSWORD" ]; then
        connect_string+=" -p$DB_PASSWORD"
    fi
    connect_string+=" -P$DB_PORT $DB_NAME"
    echo "$connect_string"
}

# Function to dump a single table
dump_table() {
    local table=$1
    echo "Dumping table: $table"

    # Use mysqldump with extended-insert and without create table statements
    # Then compress with gzip
    $(mysqldump_connect) \
        --no-create-info \
        --no-create-db \
        --no-tablespaces \
        --compact \
        --extended-insert \
        --skip-add-locks \
        --skip-comments \
        --skip-set-charset \
        --skip-add-drop-table \
        "$table" | grep -e "INSERT" | gzip > "$OUTPUT_DIR/${table}.sql.gz"

    echo "Finished dumping $table"
    echo "----------------------------------------"
}

# Start timing the entire process
total_start_time=$(date +%s)

# Get list of tables
tables=("supplier" "customer" "date" "lineorder")

# Dump each table
for table in "${tables[@]}"; do
    dump_table "$table"
done

# Calculate total execution time
total_end_time=$(date +%s)
total_execution_time=$(echo "$total_end_time - $total_start_time" | bc)
echo "Total execution time: $total_execution_time seconds"

# Print output location
echo "Dump files have been saved to: $OUTPUT_DIR"
