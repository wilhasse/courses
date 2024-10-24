#!/bin/bash

# Check if IP address is provided
if [ $# -eq 0 ]; then
    echo "Please provide the IP address of the remote host as an argument."
    exit 1
fi

# Remote host IP
REMOTE_IP=$1
LABEL="$2"

# MySQL connection details for the remote database to run queries on
QUERY_DB_NAME="ssb"
QUERY_DB_USER="root"
QUERY_DB_PASS=""

# MySQL connection details for the local database to store results
RESULT_DB_NAME="ssb_results"
RESULT_DB_USER="root"
RESULT_DB_PASS=""

# Function to create MySQL connection string
mysql_connect() {
    local db_user=$1
    local db_pass=$2
    local db_name=$3
    local host=$4

    connect_string="mysql"
    [ -n "$host" ] && connect_string+=" -h $host"
    connect_string+=" -u $db_user"
    if [ -n "$db_pass" ]; then
        connect_string+=" -p$db_pass"
    else
        connect_string+=""
    fi
    connect_string+=" $db_name"
    echo "$connect_string"
}

# Create a table to store the results if it doesn't exist (on local machine)
$(mysql_connect "$RESULT_DB_USER" "$RESULT_DB_PASS" "$RESULT_DB_NAME") <<EOF
CREATE TABLE IF NOT EXISTS query_performance (
    id INT NOT NULL AUTO_INCREMENT,
    query_id INT,
    label VARCHAR(200),
    remote_ip VARCHAR(15),
    execution_time FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id) USING BTREE
);
EOF

# Function to run query and measure time
run_query() {
    query_id=$1
    start_time=$(date +%s.%N)
    $(mysql_connect "$QUERY_DB_USER" "$QUERY_DB_PASS" "$QUERY_DB_NAME" "$REMOTE_IP") < queries/${query_id}.sql > ~/ssb/result/${query_id}_res.txt
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)

    $(mysql_connect "$RESULT_DB_USER" "$RESULT_DB_PASS" "$RESULT_DB_NAME") <<EOF
INSERT INTO query_performance (query_id, label, remote_ip, execution_time)
VALUES ($query_id, "$LABEL", '$REMOTE_IP', $execution_time);
EOF
    echo "Query $query_id executed in $execution_time seconds"
}

# Run queries
for i in {1..13}
do
    run_query $i
done

echo "All queries executed on $REMOTE_IP. Results stored in query_performance table in local $RESULT_DB_NAME database."
