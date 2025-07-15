-- Check table structure
SHOW CREATE TABLE mysql_import.historico;

-- Check if it's partitioned
SELECT 
    partition,
    count() as rows,
    formatReadableSize(sum(bytes_on_disk)) as size
FROM system.parts
WHERE database = 'mysql_import' AND table = 'historico' AND active
GROUP BY partition
ORDER BY partition;

-- Option 1: Drop and recreate without partitions
-- DROP TABLE mysql_import.historico;
-- CREATE TABLE mysql_import.historico (
--     id_contr Int32, seq UInt16, id_funcionario Int32,
--     id_tel Int32, data DateTime, codigo UInt16, modo String
-- ) ENGINE = MergeTree() 
-- ORDER BY (id_contr, seq);

-- Option 2: Create new table and copy data
-- CREATE TABLE mysql_import.historico_new (
--     id_contr Int32, seq UInt16, id_funcionario Int32,
--     id_tel Int32, data DateTime, codigo UInt16, modo String
-- ) ENGINE = MergeTree() 
-- ORDER BY (id_contr, seq);
-- INSERT INTO mysql_import.historico_new SELECT * FROM mysql_import.historico;
-- RENAME TABLE mysql_import.historico TO mysql_import.historico_old, mysql_import.historico_new TO mysql_import.historico;
-- DROP TABLE mysql_import.historico_old;