# What I Studied

> The knowledge base built along the way — from study journals, source-code reading, and
> eventually writing full courses. Kept here so it can be recalled and revisited.

## The arc

The studies followed the problem downward, then sideways:

```
MySQL as a user  →  MySQL internals  →  storage engines  →  query engines
                                                          →  columnar / parallel analytics
                                                          →  distributed SQL
                                                          →  CDC & data pipelines
```

## MySQL & InnoDB internals (the foundation)

- **Replication & HA**: GTID replication, group replication, ROW binlog format,
  parallel/multi-threaded replication and writeset dependency tracking, skip/repair of
  GTID errors; ProxySQL, Vitess, PlanetScale Boost as ecosystem studies.
- **InnoDB internals**: redo logs, recovery, page format, buffer pool; parsing `.ibd`
  files directly (pyinnodb, innodb-java-reader, inno_space); MyRocks/RocksDB as the LSM
  counterpoint; X-Engine; writing minimal storage engines from scratch.
- **Server internals**: the parser, optimizer and executor — studied by reading
  Percona Server source with a debugger (see [`db/mysql/study/`](https://github.com/wilhasse/courses/tree/main/db/mysql/study)
  and [`db/mysql/debug/`](https://github.com/wilhasse/courses/tree/main/db/mysql/debug)
  for the build/gdb harness).
- **Performance & ops**: query tuning, XtraBackup, upgrade case studies (Uber, GitHub),
  jemalloc, io_uring.

This thread culminated in the two full courses on this site — the
[InnoDB Architecture Deep-Dive](../mysql/innodb-architecture/README.md) (built on the
pre-Oracle Embedded InnoDB codebase) and the
[MySQL Server Architecture Deep-Dive](../mysql/server-architecture/README.md) (built on
Percona Server 8.0) — written to consolidate years of scattered notes into something
that can actually be recalled.

## Query engines & columnar analytics (the direction)

- **How query engines work**: Andy Grove's book/DataFusion, CMU database course,
  "survey of query execution engines"; hash join in MySQL 8.
- **Columnar engines**: DuckDB internals, MonetDB, ClickHouse; pg_duckdb and Hydra
  (columnar Postgres); PolarDB columnar join and Alibaba's CCI columnar indexes.
- **Parallel query**: Huawei Kunpeng BoostKit's parallel query for MySQL, GreatSQL's
  parallel features — intra-query parallelism as an alternative to changing engines.
- **Data interchange & pipelines**: Apache Arrow (+ Flight SQL), Parquet, Dremio, open
  table formats (Iceberg/Delta/Hudi); Kafka/Debezium/Maxwell/Canal for binlog CDC;
  Spark basics.

## Distributed SQL (the road not taken)

TiDB/TiKV (deeply — it was Strategy 1 in the [labs](./03-labs.md)), CockroachDB/Pebble,
PolarDB-X architecture and history (IOE → TDDL → PolarDB-X), Cassandra/ScyllaDB. The
takeaway that shaped everything after: distribution buys scale at the price of exactly
the operational complexity the [constraints](./01-the-idea.md) forbid.

## Build-your-own-database fundamentals

The "Build Your Own Database From Scratch" track
([`db/basic/`](https://github.com/wilhasse/courses/tree/main/db/basic)): B+Tree vs
LSM-tree, write amplification, fsync semantics, atomic rename, copy-on-write and
double-write — the vocabulary needed to evaluate every engine above; later benchmarked
concretely in the [LMDB vs B+Tree lab](./03-labs.md).

## Where the raw notes live

- Yearly study journals (2023–2025): chronological logs of every topic + reference —
  summarized in [Notes](../notes/index.md).
- Per-project study docs inside each lab folder in the
  [courses repo](https://github.com/wilhasse/courses) (e.g. `db/polardbx/study/`,
  `db/mysql/study/`).

---
**Previous:** [The Idea](./01-the-idea.md) · **Next:** [Labs & Attempts](./03-labs.md)
