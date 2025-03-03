#!/bin/bash

# Set these variables according to your setup
MYSQL_BASEDIR="/data/percona-server/build/runtime_output_directory"
MYSQL_DATADIR="/data/mysql"
MYSQL_USER="cslog"  # The user under which MySQL will run
MYSQLD_SOCKET_DIR="/var/run/mysqld"

# Check if mysqld directory exists, if not create it
if [ ! -d "$MYSQLD_SOCKET_DIR" ]; then
    echo "Creating mysqld socket directory..."
    sudo mkdir -p "$MYSQLD_SOCKET_DIR"
    # Set ownership
    sudo chown -R "$MYSQL_USER:$MYSQL_USER" "$MYSQLD_SOCKET_DIR"
    # Set permissions (commonly 755 for socket directories)
    sudo chmod 755 "$MYSQLD_SOCKET_DIR"
fi

# Set the correct ownership for data directory
sudo chown -R "$MYSQL_USER:$MYSQL_USER" "$MYSQL_DATADIR"

# Initialize the data directory
sudo "$MYSQL_BASEDIR/mysqld" \
    --defaults-file=/data/my.cnf \
    --user="$MYSQL_USER" \
    --datadir="$MYSQL_DATADIR" \
    --log-error=/data/mysql-error.log \
    --socket="$MYSQLD_SOCKET_DIR/mysqld.sock"
