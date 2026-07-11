# Chapter 8 — The Binary Log

> MySQL's second write-ahead log: the replication stream, its group commit, and GTIDs.
> Source: `sql/binlog.cc`, `libbinlogevents/`, `sql/rpl_gtid*.h`,
> `sql/rpl_commit_stage_manager.h`

## 8.1 Why a second log?

InnoDB already has a redo log (InnoDB series, Ch. 5) — but it's physical (page-level),
engine-private, and circular. Replication needs the opposite: a **logical, engine-neutral,
append-only** record of committed transactions that other servers can replay. That's the
binlog. Every write is thus logged twice (redo + binlog) — the price of the pluggable-engine
architecture, kept consistent by the 2PC of Chapter 7.

## 8.2 Events on the wire

A binlog file is a sequence of typed **events** (`libbinlogevents/`), each with a 19-byte
header (`LOG_EVENT_HEADER_LEN`, `binlog_event.h:433`): timestamp, type, server_id (the loop
breaker for circular replication), length, next-position, flags — plus an optional CRC32.

One row-format transaction on disk:

```
GTID_LOG_EVENT          "this transaction is UUID:42"  (+ last_committed/sequence_number
TABLE_MAP_EVENT          table id → schema/table + column types      for parallel apply, Ch.9)
WRITE_ROWS_EVENT         before/after row images (binary)
UPDATE_ROWS_EVENT ...
XID_EVENT                commit marker = InnoDB XID    ← the 2PC commit point of Ch. 7
```

`binlog_format` (`ROW` default, `STATEMENT`, `MIXED`): row events (`Rows_log_event`,
generated at `ha_write_row` time via `THD::binlog_write_row`, `sql/binlog.cc:11531`) replay
deterministically; statement events are compact but non-deterministic statements (UUID(),
LIMIT without ORDER BY...) made STATEMENT mode a long tale of replication drift — the reason
ROW won.

During execution, events accumulate in a per-THD **binlog cache** (`binlog_cache_data`,
`sql/binlog.cc:718` — memory up to `binlog_cache_size`, then a temp file). Nothing touches
the real binlog until commit, so transactions appear in the binlog **atomically and in
commit order**.

## 8.3 Group commit: the three-stage pipeline

At commit, the transaction's cache must be flushed to the binlog file and fsynced — naively,
one fsync per commit, on top of InnoDB's. MySQL 5.6+ solved this with a staged pipeline
(`MYSQL_BIN_LOG::ordered_commit`, `sql/binlog.cc:9203`; stages in
`Commit_stage_manager`, `rpl_commit_stage_manager.h:166`):

```
committing sessions queue up; the FIRST in each stage's queue becomes the LEADER,
processes EVERYONE queued, while the others sleep:

FLUSH stage   leader writes all queued caches to the binlog file
              (+ assigns GTIDs to the whole group, binlog.cc:8813)
SYNC stage    ONE fsync covers the whole group (sync_binlog=1)
              [optional wait: binlog_group_commit_sync_delay to grow the group]
COMMIT stage  engine commits run in queue order (binlog_order_commits)
```

This is the same group-commit idea the embedded InnoDB had for its redo log (InnoDB series
Ch. 5) — reinvented at the server layer, with an explicit leader/follower queue instead of
"piggyback on the running write". One fsync amortized over N transactions; `sync_binlog=1` +
`innodb_flush_log_at_trx_commit=1` ("dual-1") became affordable because of this pipeline.
Commit *order* is preserved end-to-end, which Chapter 9's parallel replication depends on.

## 8.4 GTIDs: names for transactions

A file:offset position is server-local and fragile. **GTIDs** name every transaction
globally: `server_uuid:sequence` (`Gtid`, `sql/rpl_gtid.h:1066`), assigned to the whole
group during the flush stage (`Gtid_state::generate_automatic_gtid`,
`sql/rpl_gtid_state.cc:488`). The server tracks sets of them (`Gtid_set` — interval lists,
so `uuid:1-4000000` is two words): `gtid_executed`, `gtid_purged`, persisted in the
`mysql.gtid_executed` table (`rpl_gtid_persist.h`).

The payoff is in Chapter 9: a replica says "I have executed *this set*" and the source
streams everything else — no positions, safe failover, idempotent reapply. Auto-positioning
replication, group replication, and every modern MySQL HA tool stand on this.

## 8.5 What to remember

1. Redo log = physical, engine, crash recovery; binlog = logical, server, replication &
   point-in-time recovery. Two logs, married by the XID 2PC.
2. A transaction = GTID + TABLE_MAP + row events + XID, buffered per-session and appended
   atomically in commit order.
3. Group commit = leader/follower FLUSH→SYNC→COMMIT pipeline; one fsync per *group* makes
   full durability practical.
4. GTID sets turn replication state from "file+offset" into set algebra.

**Try it:** `SHOW BINLOG EVENTS`, then `mysqlbinlog --hexdump --verbose <file>` — find the
19-byte header fields and the row images; `SELECT @@gtid_executed` to see the set notation.

---
**Previous:** [Chapter 7 — MDL & Transactions](./07-mdl-and-transactions.md) · **Next:** [Chapter 9 — Replication](./09-replication.md)
