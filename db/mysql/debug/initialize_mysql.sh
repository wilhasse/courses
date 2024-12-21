#!/bin/bash

# Check if an argument was provided
if [ $# -eq 1 ] && [ "$1" -eq 1 ]; then
    RUN_ADDITIONAL=true
else
    RUN_ADDITIONAL=false
fi

# stop database
sudo mysqladmin -u root shutdown -S /data/mysql/mysql.sock

# Set these variables according to your setup
MYSQL_BASEDIR="/data/percona-server/build/runtime_output_directory"
MYSQL_DATADIR="/data/mysql"
MYSQL_USER="cslog"  # The user under which MySQL will run

# Create the data directory if it doesn't exist
sudo rm -rf  $MYSQL_DATADIR
sudo rm mysqld.trace
sudo rm mysql-error.log
sudo rm mysql.log
sudo mkdir -p $MYSQL_DATADIR

# Set the correct ownership
sudo chown -R $MYSQL_USER:$MYSQL_USER $MYSQL_DATADIR

# Set up debug logs with correct permissions
./mysql_log.sh

# Initialize the data directory
sudo $MYSQL_BASEDIR/mysqld --initialize-insecure --user=$MYSQL_USER --datadir=$MYSQL_DATADIR

echo "MySQL data directory initialized. The root password is blank by default."
echo "Please change it after your first login."

# Execute additional scripts if argument is 1
if [ "$RUN_ADDITIONAL" = true ]; then
    echo "Running additional MySQL setup scripts..."
    ./run_mysql.sh &

    # Function to run grant command and check for success
    run_grant_command() {
        ./grant_mysql.sh
        return $?
    }

    # Initialize variables
    max_attempts=3
    attempt=1
    wait_time=2  # seconds to wait between attempts

    echo "Granting root privileges to MySQL..."

    # Try the command up to max_attempts times
    while [ $attempt -le $max_attempts ]; do
        if run_grant_command; then
            echo "MySQL grant command successful on attempt $attempt"
            break
        else
            if [ $attempt -lt $max_attempts ]; then
                echo "Attempt $attempt failed. MySQL might not be ready. Waiting $wait_time seconds before retry..."
                sleep $wait_time
                # Increase wait time for next attempt
                wait_time=$((wait_time * 2))
                attempt=$((attempt + 1))
            else
                echo "Error: Failed to grant MySQL privileges after $max_attempts attempts"
                exit 1
            fi
        fi
    done
fi

