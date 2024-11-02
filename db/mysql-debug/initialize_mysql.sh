#!/bin/bash -x

# Set these variables according to your setup
MYSQL_BASEDIR="/data/percona-server/build/runtime_output_directory"
MYSQL_DATADIR="/data/mysql"
MYSQL_USER="cslog"  # The user under which MySQL will run

# Create the data directory if it doesn't exist
sudo rm -rf  $MYSQL_DATADIR
sudo rm mysqld.trace
sudo rm mysql_error.log
sudo rm mysql.log
sudo mkdir -p $MYSQL_DATADIR

# Set the correct ownership
sudo chown -R $MYSQL_USER:$MYSQL_USER $MYSQL_DATADIR

# Initialize the data directory
sudo $MYSQL_BASEDIR/mysqld --initialize-insecure --user=$MYSQL_USER --datadir=$MYSQL_DATADIR

echo "MySQL data directory initialized. The root password is blank by default."
echo "Please change it after your first login."
