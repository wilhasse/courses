#!/bin/bash
set -e

echo "Building MySQL chDB Plugin..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for MySQL development headers
print_status "Checking for MySQL development headers..."
if [ ! -d "/usr/include/mysql" ] && [ ! -d "/usr/local/mysql/include" ]; then
    print_error "MySQL development headers not found"
    echo "Install MySQL development headers:"
    echo "  Ubuntu/Debian: sudo apt-get install mysql-server-dev"
    echo "  CentOS/RHEL:   sudo yum install mysql-devel"
    exit 1
fi
print_status "MySQL headers found"

# Check for chDB build
CHDB_BINARY="/home/cslog/chdb/buildlib/programs/clickhouse"
if [ ! -f "$CHDB_BINARY" ]; then
    print_error "chDB binary not found at: $CHDB_BINARY"
    echo "Build chDB first following the AVX build guide"
    exit 1
fi
print_status "chDB binary found: $CHDB_BINARY"

# Test chDB binary
print_status "Testing chDB binary..."
if ! $CHDB_BINARY local --query "SELECT 1" >/dev/null 2>&1; then
    print_error "chDB binary test failed"
    exit 1
fi
print_status "chDB binary test passed"

# Check build dependencies
print_status "Checking build dependencies..."
if ! command -v cmake >/dev/null 2>&1; then
    print_error "cmake not found"
    exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
    print_error "g++ not found"
    exit 1
fi

# Create build directory
print_status "Creating build directory..."
rm -rf build
mkdir -p build
cd build

# Configure
print_status "Configuring build..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-mavx -mavx2 -mbmi -mbmi2" \
    -DCHDB_BINARY="$CHDB_BINARY"

# Build
print_status "Building plugin..."
make -j$(nproc)

# Verify build
if [ ! -f "mysql_chdb_plugin.so" ]; then
    print_error "Build failed - plugin not created"
    exit 1
fi

print_status "Build completed successfully!"
echo ""
echo "Plugin created: $(pwd)/mysql_chdb_plugin.so"
echo "Size: $(ls -lh mysql_chdb_plugin.so | awk '{print $5}')"
echo ""
echo "Next steps:"
echo "1. Install the plugin: sudo ./install.sh"
echo "2. Test the plugin: mysql -u root -p < ../tests/test_queries.sql"