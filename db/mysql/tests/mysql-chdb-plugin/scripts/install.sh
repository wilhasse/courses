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

echo "Installing MySQL chDB UDF Plugin..."

# Check if plugin exists
if [ ! -f "build/mysql_chdb_plugin.so" ]; then
    print_error "Plugin not found. Run build.sh first."
    exit 1
fi

# Check if MySQL is running
if ! pgrep -x "mysqld" > /dev/null; then
    print_warning "MySQL server doesn't appear to be running"
    print_warning "Make sure MySQL is started before installing the plugin"
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
    echo "Common locations:"
    echo "  /usr/lib/mysql/plugin"
    echo "  /usr/local/mysql/lib/plugin" 
    echo "  /opt/mysql/lib/plugin"
    exit 1
fi

print_status "MySQL plugin directory: $MYSQL_PLUGIN_DIR"

# Copy plugin to MySQL plugin directory
print_status "Copying plugin to MySQL plugin directory..."
sudo cp build/mysql_chdb_plugin.so "$MYSQL_PLUGIN_DIR/"
sudo chmod 755 "$MYSQL_PLUGIN_DIR/mysql_chdb_plugin.so"

# Install the UDF function
print_status "Installing UDF function..."
echo "You will be prompted for MySQL root password..."

mysql -u root -p << EOF
DROP FUNCTION IF EXISTS chdb_query;
CREATE FUNCTION chdb_query RETURNS STRING SONAME 'mysql_chdb_plugin.so';
EOF

if [ $? -eq 0 ]; then
    print_status "✅ Plugin installed successfully!"
    echo ""
    echo "Test the plugin with:"
    echo "  mysql -u root -p -e \"SELECT CAST(chdb_query('SELECT version()') AS CHAR);\""
    echo ""
    echo "Or run the test suite:"
    echo "  mysql -u root -p < tests/test_udf.sql"
else
    print_error "Failed to install UDF function"
    exit 1
fi