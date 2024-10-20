#!/bin/bash -x

# Check if IP address is provided
if [ $# -eq 0 ]; then
    echo "Please provide the IP address of the remote host as an argument."
    exit 1
fi

# Remote host IP
REMOTE_IP=$1

# MySQL connection details for the remote database to run queries on
QUERY_DB_NAME="ssb"
QUERY_DB_USER="root"
QUERY_DB_PASS=""

# MySQL connection details for the local database to store results
RESULT_DB_NAME="performance_results"
RESULT_DB_USER="root"
RESULT_DB_PASS=""

# Create a table to store the results if it doesn't exist (on local machine)
mysql -u $RESULT_DB_USER -p$RESULT_DB_PASS $RESULT_DB_NAME <<EOF
CREATE TABLE IF NOT EXISTS query_performance (
    query_id INT,
    execution_time FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF

# Function to run query and measure time
run_query() {
    query_id=$1
    start_time=$(date +%s.%N)
    mysql -h $REMOTE_IP -u $QUERY_DB_USER -p$QUERY_DB_PASS $QUERY_DB_NAME < "queries/${query_id}.sql" > "~/ssb/result/${query_id}_res.txt"
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)

    # Insert result into the local performance results database
    mysql -u $RESULT_DB_USER -p$RESULT_DB_PASS $RESULT_DB_NAME <<EOF
INSERT INTO query_performance (query_id, execution_time) VALUES ($query_id, $execution_time);
EOF

    echo "Query $query_id executed in $execution_time seconds"
}

# Run queries
for i in {1..13}
do
    run_query $i
done

echo "All queries executed on $REMOTE_IP. Results stored in query_performance table in local $RESULT_DB_NAME database."
