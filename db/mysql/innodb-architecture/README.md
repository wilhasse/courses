# InnoDB Architecture Deep-Dive

A 13-chapter, step-by-step study guide to how InnoDB was architected — from the bytes on disk
up to ACID transactions — based on the source code of **Embedded InnoDB 1.0.6** (the
Innobase Oy era, before deep MySQL integration). Every claim is grounded in real source with
`file.c:line` references, plus Mermaid diagrams and byte-layout tables.

**Source repository studied:** [wilhasse/oss-embedded-innodb](https://github.com/wilhasse/oss-embedded-innodb)
— all file references in the chapters (e.g. `btr/btr0cur.c:345`) point into that tree. Clone it
alongside to read the code as you go; it builds and has runnable test programs (`tests/`) for
the hands-on exercises at the end of each chapter.

## Start here

➡️ **[Chapter 0 — Overview: How InnoDB Is Architected](./00-overview.md)** — the big-picture
map, a five-layer mental model, and how to study with this series.

📖 Also available as a [website](https://wilhasse.github.io/courses/innodb-architecture/)
and a [PDF](./innodb-architecture.pdf) for offline reading.

## Chapters

The order is bottom-up: each layer only makes sense in terms of the one below it.

| # | Chapter | Question it answers |
|---|---------|--------------------|
| [00](./00-overview.md) | Overview & Roadmap | How do all the pieces fit together? |
| [01](./01-file-storage.md) | Files, Tablespaces & Space Management | Where do bytes live? How are pages allocated? |
| [02](./02-page-format.md) | The 16KB Page & Record Format | What exactly is inside a page? How is a row encoded? |
| [03](./03-buffer-pool.md) | The Buffer Pool | How are pages cached, evicted, and safely written back? |
| [04](./04-mini-transactions.md) | Mini-Transactions & Latching | How is a page change made atomic and logged? |
| [05](./05-redo-log-recovery.md) | Redo Log & Crash Recovery | How do commits survive any crash? |
| [06](./06-btree.md) | The B+Tree | How are rows indexed, found, split, merged? |
| [07](./07-transactions-mvcc.md) | Transactions, Undo & MVCC | How do commit, rollback, and snapshot reads work? |
| [08](./08-locking.md) | The Lock Manager | How are conflicts, phantoms, and deadlocks handled? |
| [09](./09-row-operations.md) | Row Operations | What happens end-to-end on INSERT/SELECT/UPDATE/DELETE? |
| [10](./10-data-dictionary.md) | The Data Dictionary | Where does InnoDB keep its own schema? |
| [11](./11-startup-api.md) | Startup, Shutdown & the Embedded API | What happens between `ib_init()` and a usable database? |
| [12](./12-background-threads.md) | Background Threads & the Insert Buffer | Who does the work transactions defer? |

## Continue the journey

This series covers the storage engine. The companion
**[MySQL Server Architecture Deep-Dive](../mysql-server-architecture/README.md)** covers
everything *above* it — connections, parser, optimizer, executor, binlog, replication —
based on Percona Server 8.0, and meets this series at the handler API.

## Why this codebase for studying MySQL's early days

This is InnoDB as a **pure storage engine** — no MySQL parser, replication, or optimizer —
in ~250k lines of readable C. Nearly every mechanism here still exists in MySQL 8.x InnoDB
(`innodb_flush_log_at_trx_commit`, the midpoint LRU, the doublewrite buffer, next-key locking,
history list length…), just with more layers on top. The chapters point out these connections
as they appear, so the folklore of two decades of MySQL tuning maps back to the code that
created it.
