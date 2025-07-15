#!/bin/bash

# Check storage usage and compression status for chdb tables

CHDB_PATH="${1:-/tmp/chdb}"

echo "=== Storage Analysis for chdb at $CHDB_PATH ==="
echo ""

# Check overall size
echo "Total size of chdb data:"
du -sh "$CHDB_PATH"
echo ""

echo "Size by database:"
du -sh "$CHDB_PATH/data/"* 2>/dev/null | sort -h
echo ""

echo "Size of historico table:"
du -sh "$CHDB_PATH/data/mysql_import/historico" 2>/dev/null
echo ""

# Create SQL to check compression and parts
cat > /tmp/check_storage.sql << 'EOF'
-- Check parts count and state
SELECT 
    'Parts Analysis' as section,
    count() as total_parts,
    countIf(active) as active_parts,
    formatReadableSize(sum(bytes_on_disk)) as total_size,
    formatReadableSize(avg(bytes_on_disk)) as avg_part_size
FROM system.parts 
WHERE database='mysql_import' AND table='historico';

-- Check column compression
SELECT 
    name,
    formatReadableSize(data_compressed_bytes) AS compressed,
    formatReadableSize(data_uncompressed_bytes) AS uncompressed,
    round(data_compressed_bytes / data_uncompressed_bytes, 2) AS compression_ratio
FROM system.columns
WHERE database = 'mysql_import' AND table = 'historico'
ORDER BY data_compressed_bytes DESC;

-- Check merge status
SELECT 
    'Merge Status' as section,
    is_currently_executing,
    num_parts,
    formatReadableSize(total_size_bytes_compressed) as compressed_size,
    formatReadableSize(memory_usage) as memory_usage
FROM system.merges
WHERE database = 'mysql_import';
EOF

echo "To run storage analysis in chdb:"
echo "cat /tmp/check_storage.sql"
echo ""
echo "Commands to optimize storage:"
echo "1. Allow merges to run: SYSTEM START MERGES mysql_import.historico;"
echo "2. Force immediate merge: OPTIMIZE TABLE mysql_import.historico FINAL;"
echo "3. Wait for completion: SYSTEM SYNC REPLICA mysql_import.historico;"