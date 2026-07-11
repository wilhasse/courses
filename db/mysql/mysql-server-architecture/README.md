# MySQL Server Architecture Deep-Dive

A 13-chapter, step-by-step study guide to the MySQL **server layer** — everything above the
storage engine — based on the source of **Percona Server 8.0.46** (MySQL 8.0 LTS + Percona
enhancements). It follows the journey of a SQL statement top-down: socket → parser →
optimizer → executor → handler API → binlog → replica, with `file.c:line` references and
Mermaid diagrams throughout.

**Companion series:** [InnoDB Architecture Deep-Dive](../innodb-architecture/README.md)
covers everything *below* the handler API. The two series meet at Chapter 6 here — read
either first, but together they cover the whole database.

**Source studied:** [percona/percona-server](https://github.com/percona/percona-server)
(8.0 branch). File references like `sql/sql_parse.cc:1361` point into that tree; line
numbers are anchors from 8.0.46 and may drift slightly across releases.

## Start here

➡️ **[Chapter 0 — Overview: The Journey of a SQL Statement](./00-overview.md)**

📖 Also available as a
[website](https://wilhasse.github.io/courses/mysql-server-architecture/) and a
[PDF](./mysql-server-architecture.pdf) for offline reading.

## Chapters

| # | Chapter | Question it answers |
|---|---------|--------------------|
| [00](./00-overview.md) | Overview | How do the layers fit together? |
| [01](./01-connections-and-dispatch.md) | Startup, Connections & Dispatch | How does a packet become a running command in a THD? |
| [02](./02-parser.md) | The Parser | How does SQL text become a parse tree and LEX? |
| [03](./03-resolver-prepare.md) | Resolution & Prepare | How are names bound, tables opened, queries transformed? |
| [04](./04-optimizer.md) | The Optimizer | How is the cheapest plan found? |
| [05](./05-executor.md) | The Iterator Executor | How does the plan actually run? |
| [06](./06-handler-api.md) | The Handler API | How does the server talk to InnoDB — or any engine? |
| [07](./07-mdl-and-transactions.md) | MDL & Transaction Coordination | What serializes DDL vs DML? How do binlog + engine commit atomically? |
| [08](./08-binlog.md) | The Binary Log | How are changes recorded, grouped, GTID-named? |
| [09](./09-replication.md) | Replication | How do replicas stay in sync — and in parallel? |
| [10](./10-data-dictionary.md) | Data Dictionary & Atomic DDL | Where does the schema live since 8.0? |
| [11](./11-percona-additions.md) | What Percona Adds | Thread pool, MyRocks, audit, backup locks… |
| [12](./12-journey-capstone.md) | Capstone | One UPDATE traced through every layer of both series |

## How to study

1. Clone percona-server (8.0 branch) and keep it open beside each chapter.
2. Build with debug symbols; each chapter's **Try it** exercise uses gdb breakpoints,
   `EXPLAIN FORMAT=TREE`/`ANALYZE`, `optimizer_trace`, `performance_schema`, or
   `mysqlbinlog` to observe the mechanism live.
3. Finish with the capstone (Chapter 12): if you can narrate its trace from memory, you
   understand MySQL.
