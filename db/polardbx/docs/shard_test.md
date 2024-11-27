# Test

Create table

```sql
CREATE DATABASE sharding_db PARTITION_MODE=sharding DEFAULT CHARACTER SET UTF8;
USE sharding_db;

CREATE TABLE `sbtest_sharding_id` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `k` int(11) NOT NULL DEFAULT '0',
  `c` char(120) NOT NULL DEFAULT '',
  `pad` char(60) NOT NULL DEFAULT '',
  PRIMARY KEY (`id`)
) DBPARTITION BY HASH(id);
  
CREATE TABLE `sbtest_sharding_k` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `k` int(11) NOT NULL DEFAULT '0',
  `c` char(120) NOT NULL DEFAULT '',
  `pad` char(60) NOT NULL DEFAULT '',
  PRIMARY KEY (`id`)
) DBPARTITION BY HASH(k);

CREATE TABLE `sbtest_sharding_c` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `k` int(11) NOT NULL DEFAULT '0',
  `c` char(120) NOT NULL DEFAULT '',
  `pad` char(60) NOT NULL DEFAULT '',
  PRIMARY KEY (`id`)
) DBPARTITION BY HASH(c);
```

Inspect topology

```sql
SHOW TOPOLOGY FROM sbtest_sharding_id;
SHOW TOPOLOGY FROM sbtest_sharding_k;
SHOW TOPOLOGY FROM sbtest_sharding_c;
```

Create sysbench test

```bash
sudo apt-get install sysbench
/usr/share/sysbench/oltp_read_write.lua --tables=1 --table_size=100000 --mysql-user=polardbx_root --mysql-password=llWVSOXm --mysql-host=127.0.0.1 --mysql-port=54076 --mysql-db=sharding_db prepare
sysbench 1.0.18 (using system LuaJIT 2.1.0-beta3)

Creating table 'sbtest1'...
Inserting 100000 records into 'sbtest1'

Creating a secondary index on 'sbtest1'...
```

Populate sharded tables

```bash
mysql> use sharding_db;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
mysql> INSERT INTO sbtest_sharding_id SELECT * FROM sbtest1;
Query OK, 100000 rows affected (12,46 sec)

mysql> INSERT INTO sbtest_sharding_k SELECT * FROM sbtest1;
Query OK, 100000 rows affected (13,19 sec)

mysql> INSERT INTO sbtest_sharding_c SELECT * FROM sbtest1;
Query OK, 100000 rows affected (14,00 sec)
```

Test queries joining those tables:

```bash
mysql> SELECT sid.id,sid.k,sid.pad FROM sbtest_sharding_id sid INNER JOIN sbtest_sharding_k sk ON sid.k = sk.k LIMIT 2;
+----+-------+-------------------------------------------------------------+
| id | k     | pad                                                         |
+----+-------+-------------------------------------------------------------+
|  5 | 49982 | 34551750492-67990399350-81179284955-79299808058-21257255869 |
|  5 | 49982 | 34551750492-67990399350-81179284955-79299808058-21257255869 |
+----+-------+-------------------------------------------------------------+
2 rows in set (0,08 sec)
```

Query plan:

```bash
EXPLAIN SELECT sid.* FROM sbtest_sharding_id sid INNER JOIN sbtest_sharding_k sk ON sid.k = sk.k LIMIT 10;
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LOGICAL EXECUTIONPLAN                                                                                                                                                     |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Limit(offset=0, fetch=?0)                                                                                                                                                 |
|   Project(id="id", k="k", c="c", pad="pad")                                                                                                                               |
|     HashJoin(condition="k = k", type="inner")                                                                                                                             |
|       Gather(concurrent=true)                                                                                                                                             |
|         LogicalView(tables="[000000-000007].sbtest_sharding_id_daEt", shardCount=8, 
          sql="SELECT `id`, `k`, `c`, `pad` FROM `sbtest_sharding_id` AS `sbtest_sharding_id`") |
|       Gather(concurrent=true)                                                                                                                                             |
|         LogicalView(tables="[000000-000007].sbtest_sharding_k_6GuZ", shardCount=8, 
          sql="SELECT `k` FROM `sbtest_sharding_k` AS `sbtest_sharding_k`")                      |
| HitCache:true                                                                                                                                                             |
| Source:SPM_ACCEPT                                                                                                                                                         |
| BaselineInfo Id: 1669894141                                                                                                                                               |
| PlanInfo Id: 473260719                                                                                                                                                    |
| TemplateId: 63888ffd                                                                                                                                                      |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
12 rows in set (0,02 sec)
```
