#!/bin/bash -x

# Set these variables according to your setup
MYSQL_BASEDIR="/data/percona-server/build/runtime_output_directory"
MYSQL_DATADIR="/data/mysql"
MYSQL_USER="cslog"  # The user under which MySQL will run

# Set the correct ownership
sudo chown -R $MYSQL_USER:$MYSQL_USER $MYSQL_DATADIR

# Initialize the data directory
sudo $MYSQL_BASEDIR/mysqld --defaults-file=/data/my.cnf --user=$MYSQL_USER --datadir=$MYSQL_DATADIR --log-error=/data/mysql_error.log
