# Chapter 12 — Capstone: One Statement, Every Layer

> The final exam for both series: narrate an UPDATE from TCP packet to replica, touching
> every mechanism studied.

## 12.1 The statement

```sql
UPDATE accounts SET balance = balance - 100 WHERE id = 42;
```

Autocommit on, binlog on (ROW), one replica attached, InnoDB table with PK `id`.

## 12.2 The full trace

**On the source:**

1. **Wire** (Server Ch. 1): the connection's thread returns from `do_command`'s blocking
   read with a `COM_QUERY` packet → `dispatch_command` → `dispatch_sql_command`.
2. **Parse** (Ch. 2): lexer + Bison build `PT_*` nodes; contextualization fills a
   `Query_block` (one table, one assignment, WHERE `id=42`); digest recorded;
   `lex->sql_command = SQLCOM_UPDATE`.
3. **Resolve** (Ch. 3): MDL `SHARED_WRITE` on `accounts` + IX on GLOBAL/SCHEMA/COMMIT
   (Ch. 7); `TABLE_SHARE` from cache, `TABLE` instance bound, `ha_innobase` handler ready
   (Ch. 6); `fix_fields` binds `id` and `balance`.
4. **Optimize** (Ch. 4): range analysis on the PK finds `id=42` → single-row `EQ_REF`-style
   plan; AccessPath tree: `UPDATE_ROWS ← FILTER? ← INDEX lookup`.
5. **Execute** (Ch. 5): the update iterator pulls one row — `ha_index_read_map`
   (Ch. 6) → `row_search_mvcc`.
6. **Into InnoDB** (InnoDB series): B+tree descent to the leaf (Ch. 6 there), buffer pool
   fetch if needed (Ch. 3), row lock — `LOCK_X | LOCK_REC_NOT_GAP` on the PK entry
   (Ch. 8); MVCC not needed for a locking read. Row copied to `record[0]` via the row
   template.
7. The server computes the new row into `record[0]` (old in `record[1]`), calls
   `ha_update_row` → `row_update_for_mysql`: **undo record** written with the old balance
   (Ch. 7 there), row updated **in place** on the page inside a **mini-transaction** that
   emits **redo** (Ch. 4-5 there), page marked dirty in the flush list (Ch. 3). The `ha_*`
   wrapper appends a `Rows_log_event` update image to the THD's **binlog cache** (Ch. 8).
8. **Commit — the 2PC** (Ch. 7-8): `ha_commit_trans` → InnoDB **prepare**
   (`TRX_PREPARED`, redo flushed) → binlog group commit: FLUSH (GTID assigned, cache
   written), SYNC (one fsync covers the group; semi-sync waits for a replica ack here),
   COMMIT (InnoDB flips the undo state, releases the row lock — Ch. 7 there). OK packet to
   the client.
9. **Sometime later** (both series' background chapters): the page cleaner writes the dirty
   page through the doublewrite buffer; the checkpoint advances; purge reclaims the undo
   record once no read view needs the old balance.

**On the replica** (Ch. 9): the dump thread ships GTID + TABLE_MAP + UPDATE_ROWS + XID; the
I/O thread appends to the relay log; a worker (scheduled by `last_committed` — safe because
the transaction group-committed with its neighbors) applies it as a raw `ha_update_row`,
committing the new position atomically with the data.

**If anything crashes anywhere:** InnoDB redo replays committed page changes and rolls back
unprepared transactions (InnoDB Ch. 5); prepared-but-undecided transactions are resolved by
the binlog XID scan (Ch. 7); the replica resumes from its transactional position (Ch. 9).
No step in this trace can be half-done after recovery.

## 12.3 The five ideas you now own

Across ~25 chapters and two codebases, the same handful of ideas did all the work:

1. **Write-ahead + commit points.** Redo before pages, binlog XID before engine commit,
   DDL log before file ops, undo state flip as *the* commit — every durability story is
   "log the intention, then one atomic switch."
2. **MVCC: versions + visibility.** Undo chains and read views made readers lock-free in
   2005; the same undo powers rollback, purge, and even INSTANT DDL's row versioning.
3. **Two-level concurrency.** Short physical latches (pages, mtr) under long logical locks
   (rows, metadata) — each level with its own deadlock strategy (ordering vs detection).
4. **Amortize the fsync.** Group commit in the embedded redo log, reinvented as the binlog
   stage pipeline, then reused as replication's parallelism oracle. One idea, three payoffs.
5. **Interfaces make evolution possible.** The handler API let B+trees and LSM trees
   coexist; the AccessPath tree let two optimizers coexist; the dictionary-in-itself
   pattern recurs in InnoDB, the DD, MyRocks, and replication positions.

## 12.4 Where to go next

- **Read code with a debugger, not a browser** — one breakpoint per chapter of this trace
  and a single UPDATE will show you everything in an afternoon.
- **Modern frontiers**, each an extension of a chapter here: the hypergraph optimizer
  (Ch. 4), Group Replication/Paxos (Ch. 9), redo-log and buffer-pool scalability work in
  8.x InnoDB (InnoDB Ch. 3-5), HeatWave's secondary-engine hook (`override_executor_func`,
  Ch. 5's execution loop).
- **The history you now recognize**: when you read a MySQL changelog item, you can usually
  name the file it touches.

*Fim do curso — you can now narrate a database from the socket to the sector.*

---
**Previous:** [Chapter 11 — What Percona Adds](./11-percona-additions.md) · **Series index:** [README](./README.md) · **Companion:** [InnoDB Architecture Deep-Dive](../innodb-architecture/README.md)
