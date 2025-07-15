#!/bin/bash

# Usage example for the Go historico loader

# Set your MySQL connection parameters
HOST="localhost"
USER="your_user"
PASSWORD="your_password"
DATABASE="your_database"

# Optional parameters
ROW_COUNT=""  # Set to skip COUNT(*) query, e.g., ROW_COUNT="-row-count 300266692"
OFFSET=""     # Set to resume from offset, e.g., OFFSET="-offset 1000000"

echo "Starting Go historico loader..."
echo "This implementation uses chdb-go library with proper batch processing"
echo ""

# Run the loader
./historico_loader_go \
    -host "$HOST" \
    -user "$USER" \
    -password "$PASSWORD" \
    -database "$DATABASE" \
    $ROW_COUNT \
    $OFFSET

echo "Done!"