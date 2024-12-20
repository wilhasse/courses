# Guide:

https://github.com/polardb/polardbx-sql/blob/main/docs/en/quickstart-development.md

# PolarDBX Engine

Start MySQL with normal user (not root)：

```shell
mkdir -p /data/my3306/{data,log,run,tmp,mysql,conf}
# my.cnf in config git, copy to conf
cd polardbx-engine/runtime_output_directory
./mysqld --defaults-file=/data/my3306/conf/my.cnf --initialize-insecure
./mysqld --defaults-file=/data/my3306/conf/my.cnf
```

Run and create root user: 

```shell
mysql -u root -p -S /data/my3306/run/mysql.sock
mysql> 
create user root@'%';
grant all privileges on *.* to root@'%' with grant option;
set password for root@'%' = 'abc123';
```

# PolarDBX SQL

server.properties in config copy to
/home/admin/drds-server/env/config.properties

Pay attention to IP and port and protocol version its need to match between sql and engine.

```ini
# MetaDB Address
metaDbAddr=10.1.1.148:4886
# MetaDB X-Protocol Port
metaDbXprotoPort=32886
galaxyXProtocol=2
```

```java
String galaxyXProtocol = serverProps.getProperty("galaxyXProtocol");
if (!StringUtil.isEmpty(galaxyXProtocol)) {
    final int i = Integer.parseInt(galaxyXProtocol);
    if (1 == i) {
        XConfig.GALAXY_X_PROTOCOL = true;
        XConfig.OPEN_XRPC_PROTOCOL = false;
    } else if (2 == i) {
        XConfig.GALAXY_X_PROTOCOL = false;
        XConfig.OPEN_XRPC_PROTOCOL = true;
    } else {
        XConfig.GALAXY_X_PROTOCOL = false;
        XConfig.OPEN_XRPC_PROTOCOL = false;
    }
} else {
    XConfig.GALAXY_X_PROTOCOL = false;
    XConfig.OPEN_XRPC_PROTOCOL = false;
}
```

```shell
cd polardbx-sql/polardbx-server/target/polardbx-server/
bin/startup.sh \
	-I \
	-P asdf1234ghjk5678 \
    -d 10.1.1.158:4886:32886 \
    -u polardbx_root \
    -S "123456" \
    -r "abc123"
```

Note:  
where 10.1.1.158 is the PolarDBX Engine (mysql fork)  
-r "" - password the root user you created in engine  

It generates metaDBPasswd, add to server.properties

```sql
Generate password for user: my_polarx && 3!$!P4#cG3^P0!$oY6$8$xF7^!pY8%
Encrypted password: Ba2IPQCS5MxTiIeLqpkTGcEBN/wH4CG7RX1hysD1yFs=
The property file is resident at resource file, skip saving password into it
 ======== Paste following configurations to conf/server.properties ! =======
 etaDbPasswd=Ba2IPQCS5MxTiIeLqpkTGcEBN/wH4CG7RX1hysD1yFs=
```

It also creates the user in mysql engine

```sql
create metadb database: polardbx_meta_db_polardbx
create user (my_polarx) on node (127.0.0.1:4886)
create user (my_polarx) on node (127.0.0.1:4886:32886)
Root user for polarx with password: polardbx_root && 123456
Encrypted password for polarx: UY1tQsgNvP8GJGGP8vHKKA==

Initialize polardbx success
```

Run again:

```shell
./bin/startup.sh -P asdf1234ghjk5678
#[enter] go back to prompt 
ps -ef | grep java
```

Access polardbx sql:

```sql
mysql -h127.1 -P8527 -upolardbx_root
```
