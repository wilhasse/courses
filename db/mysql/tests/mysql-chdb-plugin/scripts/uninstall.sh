#!/bin/bash
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "Uninstalling MySQL chDB UDF Plugin..."

# Check if MySQL is running
if ! pgrep -x "mysqld" > /dev/null; then
    print_warning "MySQL server doesn't appear to be running"
    print_warning "Make sure MySQL is started before uninstalling the plugin"
fi

# Drop the UDF function
print_status "Dropping UDF function..."
echo "You will be prompted for MySQL root password..."

mysql -u root -p << EOF
DROP FUNCTION IF EXISTS chdb_query;
EOF

if [ $? -eq 0 ]; then
    print_status "UDF function dropped successfully"
else
    print_error "Failed to drop UDF function"
    exit 1
fi

# Find MySQL plugin directory
MYSQL_PLUGIN_DIR=""
for dir in "/usr/lib/mysql/plugin" "/usr/local/mysql/lib/plugin" "/opt/mysql/lib/plugin"; do
    if [ -d "$dir" ]; then
        MYSQL_PLUGIN_DIR="$dir"
        break
    fi
done

if [ -z "$MYSQL_PLUGIN_DIR" ]; then
    print_error "MySQL plugin directory not found"
    exit 1
fi

# Remove plugin file
if [ -f "$MYSQL_PLUGIN_DIR/mysql_chdb_plugin.so" ]; then
    print_status "Removing plugin file..."
    sudo rm -f "$MYSQL_PLUGIN_DIR/mysql_chdb_plugin.so"
    print_status "Plugin file removed"
else
    print_warning "Plugin file not found in $MYSQL_PLUGIN_DIR"
fi

print_status "✅ Plugin uninstalled successfully!"