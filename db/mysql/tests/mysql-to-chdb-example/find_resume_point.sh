#!/bin/bash

# Find the last imported row to resume from

cat > /tmp/find_resume.sql << 'EOF'
-- Get the last row imported
SELECT 
    'Last imported row:' as info,
    id_contr,
    seq,
    data
FROM mysql_import.historico
ORDER BY id_contr DESC, seq DESC
LIMIT 1;

-- Get total count
SELECT 
    'Total rows imported:' as info,
    COUNT(*) as count
FROM mysql_import.historico;

-- Get storage status
SELECT 
    'Storage status:' as info,
    count() as parts,
    formatReadableSize(sum(bytes_on_disk)) as size
FROM system.parts 
WHERE database='mysql_import' AND table='historico' AND active;
EOF

echo "Run this SQL in your chdb session to find resume point:"
echo "cat /tmp/find_resume.sql"
echo ""
echo "Then resume with:"
echo "./historico_loader_go \\"
echo "    -host 172.16.120.10 \\"
echo "    -user your_user \\"
echo "    -password 'your_password' \\"
echo "    -database your_database \\"
echo "    -skip-texto \\"
echo "    -row-count 300266692 \\"
echo "    -offset 16950000 \\"
echo "    -chdb-path /data/chdb"