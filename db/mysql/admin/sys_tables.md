# Motivation

While uprgrading to 8.0 we got an error

```bash
2024-12-06T21:00:47.066685Z 1 [System] [MY-013576] [InnoDB] InnoDB initialization has started
2024-12-06T21:00:54.209951Z 1 [System] [MY-013577] [InnoDB] InnoDB initialization has ended
2024-12-06T21:00:54.909135Z 1 [ERROR] [MY-011006] [Server] Got error 197 from SE while migrating tablespaces.
2024-12-06T21:00:54.936665Z 0 [ERROR] [MY-010020] [Server] Data Dictionary initialization failed.
2024-12-06T21:00:54.936731Z 0 [ERROR] [MY-010119] [Server] Aborting
```

Inspecting innodb schema information
We found an orphan table in data table

## MySQL 5.7

```bash
mysql> SELECT * FROM information_schema.innodb_sys_datafiles WHERE SPACE='1754';
+-------+----------------------------------+
| SPACE | PATH                             |
+-------+----------------------------------+
|  1754 | ./cslog_fattor_morto/CPD_REL.ibd |
+-------+----------------------------------+
1 row in set (0,01 sec)

mysql> SELECT * FROM information_schema.innodb_sys_tablespaces WHERE SPACE=1754;
+-------+----------------------------+------+-------------+------------+-----------+---------------+------------+---------------+-----------+----------------+
| SPACE | NAME                       | FLAG | FILE_FORMAT | ROW_FORMAT | PAGE_SIZE | ZIP_PAGE_SIZE | SPACE_TYPE | FS_BLOCK_SIZE | FILE_SIZE | ALLOCATED_SIZE |
+-------+----------------------------+------+-------------+------------+-----------+---------------+------------+---------------+-----------+----------------+
|  1754 | cslog_fattor_morto/CPD_REL |   41 | Barracuda   | Compressed |     16384 |          8192 | Single     |             0 |         0 |              0 |
+-------+----------------------------+------+-------------+------------+-----------+---------------+------------+---------------+-----------+----------------+
1 row in set (0,02 sec)

mysql> SELECT * FROM information_schema.innodb_sys_tables WHERE SPACE='1754';
Empty set (0,01 sec)
```

All the metadata table is located in ibdata1
You can't change it even with root user

```bash
mysql mysql -u root -p
mysql> DELETE FROM information_schema.innodb_sys_tablespaces WHERE SPACE=1754;
ERROR 1044 (42000): Access denied for user 'root'@'localhost' to database 'information_schema'
```

## MySQL 8.0

```bash
SELECT NAME, SPACE, SPACE_TYPE, FILE_SIZE, ALLOCATED_SIZE 
FROM information_schema.INNODB_TABLESPACES 
WHERE SPACE = 1754;

SELECT t.SPACE, t.NAME AS TABLE_NAME, 
       ts.NAME AS TABLESPACE_NAME, 
       ts.FILE_SIZE, 
       ts.SPACE_TYPE
FROM information_schema.INNODB_TABLES t
JOIN information_schema.INNODB_TABLESPACES ts ON t.SPACE = ts.SPACE
WHERE t.SPACE = 1754;

```
