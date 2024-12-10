# Introduction

MySQL Server test

# Run

Build

```bash
mvn clean package
```

Run

```bash
java -jar target/simple-mysql-server-1.0-SNAPSHOT-jar-with-dependencies.jar
Initializing system components...
Created server executor with 8 threads
Network layer initialized
SimpleServer started on port 3306
Sent handshake packet without SSL capability
New connection accepted
```

Client

```bash
mysql -h127.0.0.1 -uroot -p --ssl-mode=DISABLED --enable-cleartext-plugin
Enter password:
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 3523477504
Server version: 5.7.0-SimpleServer Simple MySQL Protocol Implementation

Copyright (c) 2000, 2024, Oracle and/or its affiliates.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> SELECT 1;
+------+
| 1    |
+------+
|    1 |
+------+
1 row in set (0.00 sec)
```
