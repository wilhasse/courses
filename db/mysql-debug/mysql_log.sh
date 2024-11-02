#!/bin/bash

# Set your MySQL user and group
MYSQL_USER=${MYSQL_USER:-"cslog"}
MYSQL_GROUP=${MYSQL_GROUP:-"cslog"}

# Debug log location
DEBUG_LOG="/data/mysql-debug.log"
ERROR_LOG="/data/mysql-error.log"

# Create log directory if it doesn't exist
sudo mkdir -p $(dirname $DEBUG_LOG)
sudo mkdir -p $(dirname $ERROR_LOG)

# Remove old logs if they exist
[ -f $DEBUG_LOG ] && sudo rm $DEBUG_LOG
[ -f $ERROR_LOG ] && sudo rm $ERROR_LOG

# Create new log files
sudo touch $DEBUG_LOG $ERROR_LOG

# Set correct ownership
sudo chown $MYSQL_USER:$MYSQL_GROUP $DEBUG_LOG
sudo chown $MYSQL_USER:$MYSQL_GROUP $ERROR_LOG

# Set correct permissions
sudo chmod 664 $DEBUG_LOG
sudo chmod 664 $ERROR_LOG

echo "Debug logs initialized with correct permissions:"
ls -l $DEBUG_LOG $ERROR_LOG
