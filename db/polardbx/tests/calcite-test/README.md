# Introduction

Apache Calcite Test

# Run

Build

```bash
mvn compile
mvn exec:java
```

```shell
[INFO] --- exec:3.1.0:java (default-cli) @ calcite-simple-example ---
Executing query: SELECT * FROM USERS
Execution plan:
5:LogicalProject(ID=[$0], NAME=[$1])
  4:LogicalTableScan(table=[[USERS]])

Executing query: SELECT * FROM users WHERE ID = 1
Execution plan:
13:LogicalProject(ID=[$0], NAME=[$1])
  12:LogicalFilter(condition=[=($0, 1)])
    11:LogicalTableScan(table=[[USERS]])

Executing query: SELECT name FROM USERS WHERE id > 1
Execution plan:
21:LogicalProject(NAME=[$1])
  20:LogicalFilter(condition=[>($0, 1)])
    19:LogicalTableScan(table=[[USERS]])

Executing query: SELECT COUNT(*) FROM users
Execution plan:
27:LogicalAggregate(group=[{}], EXPR$0=[COUNT()])
  26:LogicalTableScan(table=[[USERS]])
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  1.373 s
[INFO] Finished at: 2024-12-08T18:59:47-03:00
[INFO] ------------------------------------------------------------------------
```
