# Testing

Checking schema

```shell
mysql> select * from information_schema.schemata;
+--------------+--------------------+----------------------------+------------------------+----------+--------------------+
| CATALOG_NAME | SCHEMA_NAME        | DEFAULT_CHARACTER_SET_NAME | DEFAULT_COLLATION_NAME | SQL_PATH | DEFAULT_ENCRYPTION |
+--------------+--------------------+----------------------------+------------------------+----------+--------------------+
| def          | information_schema | utf8mb4                    | utf8mb4_general_ci     | NULL     | NO                 |
+--------------+--------------------+----------------------------+------------------------+----------+--------------------+
1 row in set (0,03 sec)
```

```shell
Create a sample database and table with 8 partitions

mysql> create database polarx_example  mode=auto;
Query OK, 1 row affected (0,69 sec)

mysql> use polarx_example;
Database changed
mysql> create table example (
    ->   `id` bigint(11) auto_increment NOT NULL,
    ->   `name` varchar(255) DEFAULT NULL,
    ->   `score` bigint(11) DEFAULT NULL,
    ->   primary key (`id`)
    -> ) engine=InnoDB default charset=utf8 
    -> partition by hash(id) 
    -> partitions 8;

Query OK, 0 rows affected (1,17 sec)

mysql> 
mysql> insert into example values(null,'lily',375),(null,'lisa',400),(null,'ljh',500);
Query OK, 3 rows affected (0,04 sec)
```

Selecting data and verifying topology

```shell
mysql> select * from example;
+----+------+-------+
| id | name | score |
+----+------+-------+
|  2 | lisa |   400 |
|  1 | lily |   375 |
|  3 | ljh  |   500 |
+----+------+-------+
3 rows in set (0,02 sec)

mysql> show topology from example;
+------+-----------------------------+--------------------+----------------+-------------------+-----------------------+---------------+-------------------+
| ID   | GROUP_NAME                  | TABLE_NAME         | PARTITION_NAME | SUBPARTITION_NAME | PHY_DB_NAME           | DN_ID         | STORAGE_POOL_NAME |
+------+-----------------------------+--------------------+----------------+-------------------+-----------------------+---------------+-------------------+
|    0 | POLARX_EXAMPLE_P00000_GROUP | example_GVLK_00000 | p1             |                   | polarx_example_p00000 | pxc_test-dn-0 | _default          |
|    1 | POLARX_EXAMPLE_P00001_GROUP | example_GVLK_00001 | p2             |                   | polarx_example_p00001 | pxc_test-dn-1 | _default          |
|    2 | POLARX_EXAMPLE_P00000_GROUP | example_GVLK_00002 | p3             |                   | polarx_example_p00000 | pxc_test-dn-0 | _default          |
|    3 | POLARX_EXAMPLE_P00001_GROUP | example_GVLK_00003 | p4             |                   | polarx_example_p00001 | pxc_test-dn-1 | _default          |
|    4 | POLARX_EXAMPLE_P00000_GROUP | example_GVLK_00004 | p5             |                   | polarx_example_p00000 | pxc_test-dn-0 | _default          |
|    5 | POLARX_EXAMPLE_P00001_GROUP | example_GVLK_00005 | p6             |                   | polarx_example_p00001 | pxc_test-dn-1 | _default          |
|    6 | POLARX_EXAMPLE_P00000_GROUP | example_GVLK_00006 | p7             |                   | polarx_example_p00000 | pxc_test-dn-0 | _default          |
|    7 | POLARX_EXAMPLE_P00001_GROUP | example_GVLK_00007 | p8             |                   | polarx_example_p00001 | pxc_test-dn-1 | _default          |
+------+-----------------------------+--------------------+----------------+-------------------+-----------------------+---------------+-------------------+
8 rows in set (0,00 sec)
```

Checking binlog polardb-cdc

```shell
mysql> show master status ;
+---------------+----------+--------------+------------------+-------------------+
| FILE          | POSITION | BINLOG_DO_DB | BINLOG_IGNORE_DB | EXECUTED_GTID_SET |
+---------------+----------+--------------+------------------+-------------------+
| binlog.000001 |    38796 |              |                  |                   |
+---------------+----------+--------------+------------------+-------------------+
1 row in set (0,25 sec)

mysql> show binlog events in 'binlog.000001' from 4;
| LOG_NAME      | POS   | EVENT_TYPE  | SERVER_ID  | END_LOG_POS | INFO           
...
| binlog.000001 |   720 | Query       | 2525424724 |         762 | BEGIN
| binlog.000001 |   762 | Rows_query  | 2525424724 |         805 | # TSO HEARTBEAT TXN                                                   
...
```

Other information

```shell
mysql> show storage ;
+-----------------+------------------+------------+-----------+----------+-------------+--------+-----------+-------+--------+
| STORAGE_INST_ID | LEADER_NODE      | IS_HEALTHY | INST_KIND | DB_COUNT | GROUP_COUNT | STATUS | DELETABLE | DELAY | ACTIVE |
+-----------------+------------------+------------+-----------+----------+-------------+--------+-----------+-------+--------+
| pxc_test-dn-0   | 10.1.1.121:17038 | true       | MASTER    | 2        | 3           | 0      | false     | null  | null   |
| pxc_test-dn-1   | 10.1.1.129:16297 | true       | MASTER    | 2        | 2           | 0      | true      | null  | null   |
| pxc_test-gms    | 10.1.1.132:15722 | true       | META_DB   | 2        | 2           | 0      | false     | null  | null   |
+-----------------+------------------+------------+-----------+----------+-------------+--------+-----------+-------+--------+
3 rows in set (0,01 sec)

mysql> show mpp ;
+----------+------------------+------+--------+
| ID       | NODE             | ROLE | LEADER |
+----------+------------------+------+--------+
| pxc_test | 10.1.1.121:57780 | W    | Y      |
| pxc_test | 10.1.1.129:61620 | W    | N      |
+----------+------------------+------+--------+
2 rows in set (0,00 sec)
```
