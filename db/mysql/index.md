# MySQL Internals Deep-Dive

Two step-by-step courses covering a complete MySQL database — from the bytes on disk to
the replica stream — grounded in real source code with `file.c:line` references and
diagrams throughout.

## The two series

<div class="grid cards" markdown>

- **[MySQL Server Architecture Deep-Dive](mysql-server-architecture/README.md)**

    *Everything above the engine, top-down.* Based on Percona Server 8.0: connections,
    parser, optimizer, iterator executor, handler API, two-phase commit, binlog,
    replication, and the 8.0 data dictionary — 13 chapters.

    📄 [PDF version](mysql-server-architecture/mysql-server-architecture.pdf)

- **[InnoDB Architecture Deep-Dive](innodb-architecture/README.md)**

    *The storage engine, bottom-up.* Based on Embedded InnoDB 1.0.6 (the pre-Oracle
    Innobase codebase): tablespaces, pages, buffer pool, redo log & recovery, B+trees,
    MVCC, locking, and background threads — 13 chapters.

    📄 [PDF version](innodb-architecture/innodb-architecture.pdf)

</div>

## How to read them

- **New to database internals?** Start with the InnoDB series — it builds a complete
  storage engine in your head from first principles, in a codebase small enough to read.
- **Know storage engines, want the SQL layer?** Start with the server series — it follows
  one statement from TCP packet to result set.
- The two series **meet at the handler API** (server series, Chapter 6): the point where
  `SELECT` becomes B+tree operations. The final capstone
  ([one UPDATE through every layer](mysql-server-architecture/12-journey-capstone.md))
  ties both together.

## Studied source trees

- [wilhasse/oss-embedded-innodb](https://github.com/wilhasse/oss-embedded-innodb) —
  Embedded InnoDB 1.0.6, buildable, with runnable test programs.
- [percona/percona-server](https://github.com/percona/percona-server) (8.0 branch) —
  Percona Server 8.0.46.

Every chapter ends with a hands-on **Try it** exercise (gdb breakpoints, `EXPLAIN
FORMAT=TREE`, `mysqlbinlog`, hex-dumping `ibdata1`...) so you can watch each mechanism run.
