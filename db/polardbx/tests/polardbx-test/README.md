# Introduction

PolarDB-X Plugin Java Test

# Build and Run

```bash
mvn clean package
```

Dependency Tree

```bash
mvn dependency:tree
```

Choosing the class to run:

```bash
D:\courses\db\polardbx\tests\polardbx-test>java -jar target/polardbx-test-1.0-SNAPSHOT.jar
Please provide a command number or name:
  1) parsesql     - Run SQL parsing tests
  2) simplequery  - Run simple database query tests
  3) server       - Run simple server
```

# Simple Server

Run the class using option above or directly using maven SimpleServer:

```bash
# compile and run
mvn clean compile exec:java -Dexec.mainClass="SimpleServer" -X

# only run -X full stack trace
mvn exec:java -Dexec.mainClass="SimpleServer" -X
```

Run front SQL

``` bash
# client
mysql -u root -p123456 -h localhost -P 8507
```

Executing a query:

Client

```bash
mysql> select count(*) from lineorder;
+----------+
| count(*) |
+----------+
| 6001171  |
+----------+
1 row in set (16.56 sec)

```

Server execute as a plugin

```bash
mysql> show processlist;
+------+------+------------------+-----+---------+-----+----------+-------------------+----------+-----------+---------------+
| Id   | User | Host             | db  | Command | Time| State    | Info              | Time_ms  | Rows_sent | Rows_examined |
+------+------+------------------+-----+---------+-----+----------+-------------------------------+----------+-----+---------+
| 3762 | teste| shared_session   | NULL| Sleep   |  10 | NULL     | PLUGIN                                 | 9241 |   0 |  0 |
| 3763 | teste| 10.1.1.139:60876 | ssb | Query   |   9 | executing| PLUGIN: select count(*) from lineorder | 8725 |   0 |  0 |
```
