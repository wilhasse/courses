# Install

Create a file where mysqld is located:

```bash
root@dbgen:/home/cslog# cat /usr/sbin/mysqld.my 
{
    "components": "file://component_keyring_file"
}
```

And when the plugin is located define the keyring_file:

```bash
root@dbgen:/usr/lib/mysql/plugin# cat component_keyring_file.cnf
{
    "path": "/var/lib/mysql-keyring/keyring_file", "read_only": false
}
```

Restart mysql

# Test

```bash
mysql> SELECT * FROM performance_schema.keyring_component_status;
+---------------------+-------------------------------------+
| STATUS_KEY          | STATUS_VALUE                        |
+---------------------+-------------------------------------+
| Component_name      | component_keyring_file              |
| Author              | Oracle Corporation                  |
| License             | GPL                                 |
| Implementation_name | component_keyring_file              |
| Version             | 1.0                                 |
| Component_status    | Active                              |
| Data_file           | /var/lib/mysql-keyring/keyring_file |
| Read_only           | No                                  |
+---------------------+-------------------------------------+
8 rows in set (0,00 sec)
```

Create a encrypted table

```bash
CREATE TABLE `customer` (
	`c_custkey` BIGINT NOT NULL,
	`c_name` VARCHAR(30) NULL DEFAULT NULL,
	`c_address` VARCHAR(30) NULL DEFAULT NULL,
	`c_city` CHAR(20) NULL DEFAULT NULL,
	`c_nation` CHAR(20) NULL DEFAULT NULL,
	`c_region` CHAR(20) NULL DEFAULT NULL,
	`c_phone` CHAR(20) NULL DEFAULT NULL,
	`c_mktsegment` CHAR(20) NULL DEFAULT NULL,
	PRIMARY KEY (`c_custkey`)
)
 ENGINE=InnoDB ENCRYPTION='Y';
```

See plugin:

```bash
mysql> SELECT * FROM performance_schema.keyring_keys;
+--------------------------------------------------+-----------+----------------+
| KEY_ID                                           | KEY_OWNER | BACKEND_KEY_ID |
+--------------------------------------------------+-----------+----------------+
| INNODBKey-5a863196-9ea2-11ef-9b22-bc2411aca338-1 |           |                |
+--------------------------------------------------+-----------+----------------+
1 row in set (0,00 sec)
```

Physical file now it is json format:

```bash
cat /var/lib/mysql-keyring/keyring_file 
{"version":"1.0","elements":[{"user":"","data_id":"INNODBKey-5a863196-9ea2-11ef-9b22-bc2411aca338-1","data_type":"AES","data":"DB2A4B86EB78997B28F7305FC7D8BEA8448BEFC0C9BC392724A696F51FDBEC58","extension":[]}]}
```