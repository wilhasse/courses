# Basic

Backup an entire database and compress it using pigz:

```bash
mysqldump -u username --single-transaction -p database_name | pigz > backup_name.sql.gz
```

Backup selected tables from a database:
```bash
mysqldump -u username --single-transaction -p database_name table1 table2 | pigz > backup_tables.sql.gz
```

A table with specific period range:

```bash
mysqldump -u username --single-transaction -p database_name -t table1 --w "DATA >= '2025-01-01' AND DATA < '2025-02-01'" | pigz > backup_mes.sql.gz
```

### Common Flags
* `-u`: MySQL username
* `-p`: Prompt for password
* `-h`: Host name
* `-P`: Port number

# Other

Full with routines (store procedure):

```bash
mysqldump table -u root --routines -p -q | pigz -c > dump_db.sql.gz
```

Only store procedure

```bash
mysqldump -u root -p --routines --no-data --skip-triggers --no-create-info database > dump_prod_st.sql
```
Only views

```bash
mysql -u backup database -e "SHOW FULL TABLES WHERE TABLE_TYPE LIKE 'VIEW'" \
| grep 'VIEW' \
| awk '{ print $1 }' \
| while read view; do \
    mysql -u backup database -e "SHOW CREATE VIEW \`$view\`;" >> views.sql; \
done
```

Sql to drop routines and views

```sql
# routines
SELECT CONCAT('DROP ', routine_type, ' IF EXISTS ', routine_name, ';') 
FROM information_schema.routines 
WHERE routine_schema = 'database';

#views
SELECT CONCAT('DROP VIEW IF EXISTS ', table_name, ';')
FROM information_schema.views
WHERE table_schema = 'database';
```
