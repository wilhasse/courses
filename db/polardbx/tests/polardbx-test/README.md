# Introduction

PolarDB-X Plugin Java Test

# Build

```bash
mvn clean package
```

Check dependency Tree

```bash
mvn dependency:tree
```

Choosing the class to run:

```bash
D:\courses\db\polardbx\tests\polardbx-test>java -jar target/polardbx-test-1.0-SNAPSHOT.jar
Please provide a command number or name:
  1) query        - Run database query tests
  2) parse        - Run SQL parsing tests
  3) server       - Run simple server
  4) server_split - Run simple server splitting query
  5) server_parallel - Run server parallel splitting query
```

# Direct Query (XProtocol)

```bash
java -jar target/polardbx-test-1.0-SNAPSHOT.jar 1

20:55:28.048 [main] INFO com.alibaba.polardbx.common.utils.logger.LoggerFactory - using logger: com.alibaba.polardbx.common.utils.logger.slf4j.Slf4jLoggerAdapter
20:55:28.060 [main] INFO XLog -  [TDDL] XProtocol NIOWorker start with 16 threads and 16777216 bytes buf per thread., tddl version: 1.0-SNAPSHOT
Initializing connection to 10.1.1.148:33660
20:55:28.285 [main] INFO XLog -  [TDDL] XConnectionManager new datasource to teste@10.1.1.148:33660 id is 0 NOW_GLOBAL_SESSION: 0, tddl version: 1.0-SNAPSHOT
Attempting to establish connection...

Enter SQL query (or 'exit' to quit):
```

# Parse SQL

``` bash
D:\courses\db\polardbx\tests\polardbx-test>java -jar target/polardbx-test-1.0-SNAPSHOT.jar 2 "SELECT User FROM user WHERE User = 'root'"
21:21:52.052 [main] INFO com.alibaba.polardbx.common.utils.logger.LoggerFactory - using logger: com.alibaba.polardbx.common.utils.logger.slf4j.Slf4jLoggerAdapter
Parsed AST: SELECT `User`
FROM `user`
WHERE (`User` = 'root')
Validated SQL: SELECT `user`.`User`
FROM `user`
WHERE (`user`.`User` = 'root')
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

