#!/bin/bash

echo "=== chDB Installation Verification ==="
echo "Date: $(date)"
echo

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

success_count=0
total_tests=0

check() {
    local test_name="$1"
    local command="$2"
    local expected="$3"
    
    total_tests=$((total_tests + 1))
    echo -n "Checking $test_name... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        success_count=$((success_count + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

# Test 1: Library exists
check "library file exists" "test -f /usr/local/lib/libchdb.so"

# Test 2: Library is in ldconfig
check "library in ldconfig" "/sbin/ldconfig -p | grep -q libchdb"

# Test 3: Can compile with chdb
check "compilation test" "echo 'int main(){}' | gcc -x c - -lchdb -o /tmp/test_chdb 2>/dev/null && rm -f /tmp/test_chdb"

# Test 4: Dynamic loading works
cat > /tmp/dlopen_test.c << 'EOF'
#include <dlfcn.h>
int main() {
    void* h = dlopen("libchdb.so", RTLD_LAZY);
    if (h) { dlclose(h); return 0; }
    return 1;
}
EOF
check "dynamic loading" "gcc -o /tmp/dlopen_test /tmp/dlopen_test.c -ldl && /tmp/dlopen_test"
rm -f /tmp/dlopen_test /tmp/dlopen_test.c

# Test 5: Basic query works
if [ -f ./test_chdb_installation ]; then
    check "basic query execution" "./test_chdb_installation | grep -q 'Query successful'"
fi

# Test 6: API tools exist
check "execute_sql exists" "test -f ./execute_sql"
check "chdb_api_server_simple exists" "test -f ./chdb_api_server_simple"
check "chdb_api_client_simple exists" "test -f ./chdb_api_client_simple"

# Test 7: Data directories
echo
echo "Data Directories:"
for dir in /chdb/data /tmp/chdb ./clickhouse_data; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}✅${NC} $dir (${size})"
    else
        echo -e "  ${YELLOW}⚠️${NC}  $dir (not found)"
    fi
done

# Summary
echo
echo "=== Summary ==="
echo "Tests passed: $success_count/$total_tests"

if [ $success_count -eq $total_tests ]; then
    echo -e "${GREEN}✅ chDB is fully installed and operational!${NC}"
else
    echo -e "${YELLOW}⚠️  Some tests failed. chDB may need attention.${NC}"
    echo
    echo "To fix installation issues:"
    echo "  1. Install chDB: curl -sL https://lib.chdb.io | bash"
    echo "  2. Update library cache: sudo ldconfig"
    echo "  3. Rebuild tools: make clean && make all"
fi

# Version info
echo
echo "System Information:"
echo "  OS: $(uname -s) $(uname -r)"
echo "  Architecture: $(uname -m)"
if [ -f ./test_chdb_installation ]; then
    version=$(./test_chdb_installation 2>/dev/null | grep "Result:" -A1 | tail -1 | awk '{print $1}')
    if [ -n "$version" ]; then
        echo "  ClickHouse version: $version"
    fi
fi

# Library info
lib_size=$(ls -lh /usr/local/lib/libchdb.so 2>/dev/null | awk '{print $5}')
if [ -n "$lib_size" ]; then
    echo "  Library size: $lib_size"
fi