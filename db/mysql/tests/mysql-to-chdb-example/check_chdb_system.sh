#!/bin/bash

echo "=== chDB System Check ==="
echo

# Check if libchdb.so is in ldconfig
echo "1. Checking ldconfig..."
if ldconfig -p | grep -q libchdb; then
    echo "✅ libchdb.so found in ldconfig:"
    ldconfig -p | grep libchdb | head -5
else
    echo "❌ libchdb.so NOT found in ldconfig"
fi
echo

# Check common installation paths
echo "2. Checking installation paths..."
PATHS=(
    "/usr/local/lib/libchdb.so"
    "/usr/lib/libchdb.so"
    "/opt/chdb/lib/libchdb.so"
    "$HOME/chdb/libchdb.so"
    "./libchdb.so"
)

found=0
for path in "${PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✅ Found: $path"
        ls -lh "$path"
        found=1
    fi
done

if [ $found -eq 0 ]; then
    echo "❌ No libchdb.so found in common paths"
fi
echo

# Check library dependencies
echo "3. Checking library dependencies..."
if command -v ldd >/dev/null 2>&1; then
    lib_path=$(ldconfig -p | grep libchdb.so | head -1 | awk '{print $NF}')
    if [ -n "$lib_path" ]; then
        echo "Library dependencies for $lib_path:"
        ldd "$lib_path" | head -10
    fi
else
    echo "⚠️  ldd command not available"
fi
echo

# Check environment variables
echo "4. Checking environment..."
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<not set>}"
echo "PATH: $PATH" | head -1
echo

# Test with simple program
echo "5. Testing dynamic loading..."
cat > /tmp/test_dlopen.c << 'EOF'
#include <stdio.h>
#include <dlfcn.h>
int main() {
    void* h = dlopen("libchdb.so", RTLD_LAZY);
    if (h) {
        printf("✅ dlopen succeeded\n");
        dlclose(h);
        return 0;
    } else {
        printf("❌ dlopen failed: %s\n", dlerror());
        return 1;
    }
}
EOF

gcc -o /tmp/test_dlopen /tmp/test_dlopen.c -ldl 2>/dev/null
if [ -f /tmp/test_dlopen ]; then
    /tmp/test_dlopen
    rm -f /tmp/test_dlopen /tmp/test_dlopen.c
else
    echo "❌ Failed to compile test program"
fi
echo

# Check chDB data directories
echo "6. Checking data directories..."
DATA_DIRS=(
    "/chdb/data"
    "/tmp/chdb"
    "./clickhouse_data"
    "$HOME/.chdb"
)

for dir in "${DATA_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Found data directory: $dir"
        du -sh "$dir" 2>/dev/null | head -1
    fi
done
echo

# Summary
echo "=== Summary ==="
if ldconfig -p | grep -q libchdb && [ -f /usr/local/lib/libchdb.so ]; then
    echo "✅ chDB appears to be properly installed"
    echo "   You can use it in your programs by linking with -lchdb or using dlopen()"
else
    echo "⚠️  chDB installation may need attention"
    echo "   Run: curl -sL https://lib.chdb.io | bash"
    echo "   Then: sudo ldconfig"
fi