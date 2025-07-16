#!/bin/bash

# Script to update buffer size in all chDB API UDF source files

if [ $# -ne 1 ]; then
    echo "Usage: $0 <size_in_mb>"
    echo "Example: $0 50  # Sets buffer to 50MB"
    echo "Example: $0 100 # Sets buffer to 100MB"
    echo
    echo "Current buffer sizes:"
    grep -h "MAX_RESULT_SIZE" src/chdb_api*.cpp | grep -v "MAX_ALLOWED" | sort -u
    exit 1
fi

SIZE_MB=$1
SIZE_BYTES=$((SIZE_MB * 1024 * 1024))

echo "=== Updating chDB API UDF Buffer Size ==="
echo "Setting buffer size to ${SIZE_MB}MB (${SIZE_BYTES} bytes)"
echo

# Files to update
FILES=(
    "src/chdb_api_udf.cpp"
    "src/chdb_api_json_udf.cpp"
    "src/chdb_api_ip_udf.cpp"
    "src/chdb_api_ip_json_udf.cpp"
)

# Update each file
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Updating $file..."
        sed -i "s/#define MAX_RESULT_SIZE [0-9]*/#define MAX_RESULT_SIZE $SIZE_BYTES/" "$file"
        
        # Show the change
        grep "MAX_RESULT_SIZE" "$file" | head -1
    fi
done

echo
echo "Buffer size updated to ${SIZE_MB}MB"
echo
echo "To apply changes:"
echo "1. Rebuild the plugins:"
echo "   ./scripts/install_ip_udf.sh"
echo "   ./scripts/install_json_udf.sh"
echo "   ./scripts/install_chdb_api.sh"
echo
echo "2. Or manually rebuild specific plugins"
echo
echo "WARNING: Large buffers consume more memory per connection."
echo "Recommended sizes:"
echo "  - 10-20MB for normal queries"
echo "  - 50-100MB for large result sets"
echo "  - 200MB+ only if absolutely necessary"