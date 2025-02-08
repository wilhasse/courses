# Kunpeng BoostKit Database Enablement Suite

*Translated using Claude

# MySQL Parallel Query Optimization Feature Guide

Document Version: 04
Release Date: 2024-11-28
Huawei Technologies Co., Ltd.

## Copyright Notice

Copyright © Huawei Technologies Co., Ltd. 2025. All rights reserved.

No part of this document may be reproduced or transmitted in any form or by any means without prior written consent of Huawei Technologies Co., Ltd.

## Trademark Notice

HUAWEI and other Huawei trademarks are trademarks of Huawei Technologies Co., Ltd.
All other trademarks and registered trademarks mentioned in this document are the property of their respective holders.

## Notice

The purchased products, services and features are stipulated by the contract made between Huawei and the customer. All or part of the products, services and features described in this document may not be within the purchase scope or the usage scope. Unless otherwise specified in the contract, all statements, information, and recommendations in this document are provided "AS IS" without warranties, guarantees or representations of any kind, either express or implied.

The information in this document is subject to change without notice. Every effort has been made in the preparation of this document to ensure accuracy of the contents, but all statements, information, and recommendations in this document do not constitute a warranty of any kind, express or implied.

## Security Statement

### Vulnerability Management Process

Huawei's product vulnerability management is specified in the "Vulnerability Management Process". For detailed content, please refer to:
https://www.huawei.com/cn/psirt/vul-response-process

For enterprise customers to obtain vulnerability information, please visit:
https://securitybulletin.huawei.com/enterprise/cn/security-advisory

## 1 Introduction

### 1.1 Application Scenarios

The MySQL parallel query optimization solution primarily targets OLAP scenarios in databases. OLAP scenarios refer to multi-dimensional analysis and querying of large-scale data, typically requiring scalability, data consistency, high performance, and high security. In such scenarios, database query response time and throughput are crucial for ensuring normal application operation.

Through MySQL parallel query optimization solution, parallel data reading is implemented using multiple cores and threads to execute SQL statements, accelerating query execution speed. MySQL parallel query optimization is mainly used in data analysis, BI reporting, and decision support business scenarios. Currently, it supports parallel queries for four types of single-table scans:

- JT_ALL
- JT_INDEX_SCAN
- JT_REF
- JT_RANGE

Building upon single-table conditions, it supports simple multi-table parallel queries, does not support subqueries, and supports some semi-join queries. The solution is supported through a whitelist approach. Specifically:

#### Single-table Whitelist:

```sql
select {column_name|Aggregate} from table where {=|>|<|>=|<=|like|between...and|in} group by {column_name} having {column_name} order by {column_name|Aggregate} limit x
```

Note:

- In the whitelist format, select and from are required, while where, group by, having, order by, and limit are optional.
- Aggregate represents: sum min max avg count

#### Multi-table Whitelist:

```sql
select {column_name|Aggregate} from table1 table2 ... where {=|>|<|>=|<=|like|between...and|in} group by {column_name} having {column_name} order by {column_name} limit x
```

Note:

- In the whitelist format, select and from are required, while where, group by, having, order by, and limit are optional.
- If the query involves system tables, temporary tables, non-InnoDB tables, stored procedures, or serializable isolation level, the parallel functionality will not be effective.

#### Semi-join Queries:

Some semi-join queries can be transformed into simple queries by MySQL's optimizer. If the innermost table in the execution plan is an outer table, then these types of SQL can also support parallel queries.

#### Aggregate Arithmetic Operations:

Supports arithmetic operations on aggregates, for example: sum()/sum(), a*sum(), where a is a constant.

### Security Hardening Statement

MySQL parallel query optimization supports MySQL 8.0.20 and MySQL 8.0.25 versions. It is recommended to pay attention to CVE vulnerabilities of corresponding versions on the MySQL official website and apply vulnerability fixes as required.

### 1.2 Parallel Query Feature Introduction

#### 1.2.1 Implementation Principle

Parallel query involves two key events: table partitioning and execution plan transformation.

##### Table Partitioning

The scanned data is divided into multiple parts for parallel scanning by multiple threads. InnoDB engine is an index-organized table, with data stored on disk in B+tree form. The partitioning logic starts from the root node page and scans down level by level. When the number of branches at a certain level exceeds the configured number of threads, partitioning stops. In implementation, two partitioning passes are actually performed.

The first pass divides partitions based on the number of branches from the root node page. For each branch, the leftmost leaf node record serves as the lower bound, and this record is marked as the upper bound for the adjacent previous branch. Through this method, the B+tree is divided into several subtrees, with each subtree being a scan partition.

To resolve the load imbalance issue from the first partitioning, a second partitioning is performed on the remaining partitions. After the second partitioning, multiple blocks with smaller data volumes can be obtained, allowing for more balanced scanning data distribution among threads.

##### Execution Plan Transformation

MySQL's execution plan is a left-deep tree. Before parallel execution, MySQL uses one thread to recursively execute this left-deep tree, then performs sort or aggregation on the join results. The goal of parallelization is to use multiple threads to execute this execution plan tree in parallel. The first non-const primary table is partitioned, and each thread's execution plan is identical to the original execution plan, except that the first table is only a portion of that table. This way, each thread executes a portion of the execution plan, these threads are called worker threads. After execution, the results are handed to the leader for consolidation, then sort, aggregation is performed, or results are sent directly to the client.

#### 1.2.2 Key Function Flow

In the parallel framework, there are leader threads and worker threads.

The leader thread's make_pq_leader_plan function is responsible for determining whether a statement can be executed in parallel, generating the leader's own execution plan based on the original execution plan, and then calling the ParallelScanIterator iterator to execute.

##### Init

During initialization, the leader thread first calls add_scan to partition the table that can be executed in parallel, dividing it into multiple data shards and placing all shards in a queue. Then it calls mysql_thread_create to create several worker threads, which loop to take a shard from the queue for execution until all shards have been executed.

##### Read

During the Read process, the leader thread calls the gather module to get data from the message queue. If needed, it can also perform additional operations such as count counting, sum operations, or other aggregate operations. Finally, the data is passed upward to the client.

##### End

The End operation is the final cleanup operation of the iterator, such as releasing memory and returning the status of read data.

Worker threads are started and created by the leader thread, with the number of worker threads controlled by the parallel degree parameter. Worker threads first call make_pq_worker_plan to generate their own execution plan. In this process, a replaceable iterator in the original execution plan is replaced with the parallel iterator PQblockScanIterator. Then it calls the read function of PQblockScanIterator, which calls the interaction interface with the InnoDB storage engine to get data from InnoDB, and finally calls the send_data function to send data to the message queue for the leader thread to use.

## 2 Patch Usage Instructions

Specific operation steps are as follows:

Step 1: Download MySQL source code according to Table 2-1 and store it in the target path, for example "/home".

Table 2-1 MySQL Different Version Source Code Download Addresses

| Version      | Download Address                                             |
| ------------ | ------------------------------------------------------------ |
| MySQL 8.0.20 | https://github.com/mysql/mysql-server/archive/mysql-8.0.20.tar.gz |
| MySQL 8.0.25 | https://github.com/mysql/mysql-server/archive/mysql-8.0.25.tar.gz |

Note: Code downloaded from Github does not include the boost folder. You can download source code containing boost from the MySQL official website and obtain the boost folder from it. The path to this boost folder will be needed during compilation.

Step 2: Download MySQL parallel query optimization feature Patch package according to Table 2-2.

Table 2-2 MySQL Different Version Patch Package Description

| Supported Version | Patch Package                 | Description                                                  |
| ----------------- | ----------------------------- | ------------------------------------------------------------ |
| MySQL 8.0.20      | code-pq.patch                 | Source code Patch, contains all code needed for parallel query functionality |
|                   | mtr-pq.patch                  | Patch for mtr tests in mysql-test, ensures all mtr tests pass |
| MySQL 8.0.25      | code-pq-forMySQL-8.0.25.patch | Source code Patch, contains all code needed for parallel query functionality |
|                   | mtr-pq-forMySQL-8.0.25.patch  | Patch for mtr tests in mysql-test, ensures all mtr tests pass |

- Current Patch package is based on MySQL 8.0.20 and 8.0.25 versions from the Gitee community.
- Current Patch package has completed functional verification on the Aarch64 Linux platform.
- Current Patch package does not support x86 hardware platform.

Step 3: Extract the source code package and enter the MySQL source code directory.

```bash
tar -zxvf mysql-boost-8.0.20.tar.gz
cd mysql-8.0.20
```

Step 4: In the source code root directory, use git initialization command to establish git management information.

```bash
git init
git add -A
git commit -m "Initial commit"
```

Note:

- Generally, git comes with the system. If you need to install git, please first refer to the "MySQL Migration Guide" for configuring Yum source related content, then execute the following command to install git.

```bash
yum install git
```

- If git submission user information is not configured, you need to configure user email and username information before git commit.

```bash
git config user.email "123@example.com"
git config user.name "123"
```

Step 5: Apply MySQL parallel query optimization feature patch.

```bash
git apply --whitespace=nowarn -p1 < mtr-pq.patch
git apply --whitespace=nowarn -p1 < code-pq.patch
```

If there is no echo error message, the patch has been successfully applied.

Step 6: Follow the normal steps for compiling and installing MySQL source code. For detailed information, please refer to the "MySQL Migration Guide".

## 3 Parallel Query Parameter Description

Six parallel-related parameters have been added as shown in Table 3-1.

Table 3-1 Parallel-related Parameters and Their Descriptions

| Parameter               | Description                                                  | Value                                                        |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| parallel_cost_threshold | Global and session level parameter, used to set the threshold for SQL statement execution of parallel query. Only when the query's estimated cost is higher than this threshold will parallel query be executed. When the SQL statement's estimated cost is lower than this threshold, the original query process will be executed. | - Value range: 0 ~ ULONG_MAX<br>- Default value: 1000        |
| parallel_default_dop    | Global and session level parameter, used to set the maximum concurrency for parallel query of each SQL statement. The query concurrency will be dynamically adjusted based on the table size. If the binary tree is too small (number of table partitions less than parallel degree), then the query concurrency will be set based on the table's partition count. The maximum parallel degree of each query will not exceed the value set by parallel_default_dop parameter. | - Value range: 0 ~ 1024<br>- Default value: 4                |
| parallel_max_threads    | Global level, used to set the total number of parallel query threads in the system. | - Value range: 0 ~ ULONG_MAX<br>- Default value: 64          |
| parallel_memory_limit   | Global level, used to set the total memory size limit for leader thread and worker threads during parallel execution. | - Value range: 0 ~ ULONG_MAX<br>- Default value: 100*1024*1024 |
| parallel_queue_timeout  | Global and session level, used to set the timeout period for parallel query waiting in the system. If system resources are insufficient, for example, if the running parallel query threads have reached the parallel_max_threads value, parallel query statements will wait. If the timeout is reached and resources still haven't been obtained, the original query process will be executed. | - Value range: 0 ~ ULONG_MAX, unit: ms<br>- Default value: 0 |
| force_parallel_execute  | Global and session level, used to set the switch for parallel query. | - bool value can be set to on or off. on indicates enabling parallel query feature, off indicates disabling parallel query feature.<br>- Default value: off |

Four status variables have also been added as shown in Table 3-2:

Table 3-2 Status Variables and Their Descriptions

| Status Variable    | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| PQ_threads_running | Global level, current total number of running parallel execution threads. |
| PQ_memory_used     | Global level, current total memory usage of parallel execution. |
| PQ_threads_refused | Global level, total number of queries that could not execute parallel execution due to total thread number limitation. |
| PQ_memory_refused  | Global level, total number of queries that could not execute parallel execution due to total memory limitation. |

## 4 Parallel Query Usage Instructions

You can use the parallel query optimization feature through either of the following two methods:

### Method 1: Set System Parameters

Control whether to enable parallel query by setting the global parameter force_parallel_execute; control how many threads to use for parallel query by setting the global parameter parallel_default_dop. These parameters can be modified at any time during use without restarting the database.

For example, to enable parallel execution with a concurrency of 4:

```sql
force_parallel_execute=on;
parallel_default_dop=4;
```

You can adjust the value of parallel_cost_threshold parameter based on actual conditions. If set to 0, all queries will use parallel execution; if set to a non-zero value, only queries with estimated cost value greater than this threshold will use parallel execution.

Note: If parallel query optimization feature is enabled but found not effective, please refer to the solution for "Parallel Query Optimization Feature Not Effective After Enabling".

### Method 2: Use Hint Syntax

Use hint syntax to control whether individual statements execute in parallel. When the system default parallel execution is off, hint can be used to accelerate specific SQL, but the parallel degree specified by hint cannot be greater than parallel_max_threads, otherwise SQL statement parallel query cannot be enabled. Conversely, certain SQL types can also be restricted from entering parallel execution.

- `SELECT /*+ PQ */ ... FROM ...` indicates using default concurrency 4 for parallel query.
- `SELECT /*+ PQ(8) */ ... FROM ...` indicates using concurrency 8 for parallel query.
- `SELECT /*+ NO_PQ */ ... FROM ...` indicates this statement does not use parallel query.

Performance improvement effects before and after using MySQL parallel query optimization feature can be obtained through TPC-H testing. For detailed test steps, please refer to "TPC-H Test Guide (for MySQL)".

From the test data, after adopting MySQL parallel query optimization feature, parallel degree can be increased, and query performance can be improved by more than 1 time (performance improvement is related to parallel degree).

## 5 Potential Incompatibility with Serial Results

Parallel execution results may be incompatible with serial execution in the following aspects:

### Error or Warning Message Count May Increase

For queries that show error/warning messages in serial execution, under parallel execution, each worker thread may show error/warning messages, leading to an increase in total error/warning message count.

### Precision Issues

During parallel execution process, there might be more intermediate results stored than in serial execution. If intermediate results are floating-point type, floating-point precision errors may occur, leading to slight differences in final results.

### Result Set Order Differences

When multiple worker threads execute queries, the returned result set order may be inconsistent with serial execution order. If using a group by statement query, the order within groups after grouping