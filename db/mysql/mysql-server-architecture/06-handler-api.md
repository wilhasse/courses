# Chapter 6 — The Handler API: Where SQL Meets the Storage Engine

> The pluggable-engine interface — and the exact point where this series hands off to the
> InnoDB deep-dive.
> Source: `sql/handler.h`, `sql/handler.cc`, `storage/innobase/handler/ha_innodb.cc`

## 6.1 Two objects define an engine

MySQL's storage-engine pluggability rests on one pair (`sql/handler.h`):

- **`handlerton`** (`:2663`) — "handler singleton": one per engine, a struct of function
  pointers for everything *not* tied to one table: `commit`, `rollback`, `prepare` (the 2PC
  surface — Chapter 7), `create` (the handler factory), DDL hooks, and the data-dictionary
  bridge (`ddse_dict_init` — InnoDB serving as dictionary engine, Chapter 10).
- **`class handler`** (`:4506`) — one per *open table per session*: a cursor-style abstract
  class the executor drives.

The core `handler` virtuals form a tiny, 1990s-flavored ISAM interface:

```
lifecycle   open() close() external_lock() start_stmt()
full scan   rnd_init() rnd_next(buf) rnd_pos(buf, rowid)
index scan  index_init(idx) index_read_map(key) index_next() index_next_same()
writes      write_row(buf) update_row(old,new) delete_row(buf)
metadata    info(flags) records_in_range(idx, min, max) position(record)
```

Every iterator from Chapter 5 bottoms out here: `TableScanIterator::Read()` calls
`ha_rnd_next`; `RefIterator` calls `ha_index_read_map`; the row-apply path of replication
(Ch. 9) calls `ha_write_row`. The public `ha_*` wrappers (e.g. `handler::ha_write_row`,
`sql/handler.cc:8353`) add PSI instrumentation, then invoke the engine's virtual — and on
success call `binlog_log_row()`: **row-based binlogging is a side effect of the handler
API**, which is why it works identically for every engine.

Rows cross the boundary in **MySQL row format**: `TABLE::record[0]` (current row) and
`record[1]` (old row for updates) — `table.h:1486`. The engine converts to its own format
internally.

An engine registers by declaring a plugin (`mysql_declare_plugin(innobase)`,
`ha_innodb.cc:24586`) whose init function fills in the handlerton
(`innodb_init`, `:5526`: `->commit = innobase_commit`, `->create = innobase_create_handler`,
…). "Storage engine" and "plugin" are the same mechanism.

## 6.2 Crossing the bridge into InnoDB

`ha_innobase` (`storage/innobase/handler/ha_innodb.cc`) implements the interface, and if you
did the InnoDB series, everything behind it is familiar — the same machinery, 20 years after
the embedded version:

| handler call | InnoDB internal | InnoDB series |
|---|---|---|
| `open()` (`:7716`) | `row_create_prebuilt()` → **`row_prebuilt_t`** (`row0mysql.h:578`) | the `ib_crsr_t` prebuilt (Ch. 11) |
| `write_row()` (`:9587`) | `row_insert_for_mysql()` → insert graph → `row_ins` | `ib_cursor_insert_row` (Ch. 9.2) |
| `index_read()` (`:10802`) | **`row_search_mvcc()`** (`row/row0sel.cc:4297`) | `row_search_for_client` (Ch. 9.3) |
| `rnd_next()` (`:11473`) | `general_fetch` → `row_search_mvcc` | same |
| `update_row()` / `delete_row()` (`:10348`/`:10514`) | `row_update_for_mysql()` | `row_upd`, delete-mark (Ch. 9.4) |
| `records_in_range()` (`:17403`) | `btr_estimate_n_rows_in_range` — B+tree dives | the index dives feeding Ch. 4's optimizer |
| `info()` (`:18232`) | sampled statistics → `stats.records`, `rec_per_key` | `dict_index_t` stats (Ch. 10) |

The **row template** mechanism (`build_template()`, `ha_innodb.cc:8957`) precompiles the
column mapping between InnoDB records and `record[0]` into `mysql_row_templ_t[]`, so
`row_sel_store_mysql_rec()` (`row/row0sel.cc:2678`) can convert each fetched row with
memcpy-level work rather than re-interpreting metadata per row. The `row_prebuilt_t` caching
this — along with the persistent cursor it embeds — is the *same struct by the same name* as
in Embedded InnoDB: the handler API is, in a real sense, the embedded API with MySQL as the
only client.

## 6.3 The impedance mismatches

The interface's simplicity has costs, and much of MySQL's evolution is visible as patches
around them:

- **Row-at-a-time**: one virtual call (and engine latch dance) per row. Mitigations:
  MRR (`multi_range_read_*` — batch ranges, sort rowids), batched key access, and the
  handler's ICP hook (`index_read_pushed`) that pushes WHERE fragments *into* the engine's
  index scan, filtering before row materialization.
- **Opaque statistics**: the optimizer sees only `info()`/`records_in_range()` — estimate
  quality is engine-private (InnoDB's sampled dives), invisible to the planner.
- **Transaction coordination**: each engine commits independently, so multi-engine (or
  engine + binlog) atomicity needs the server-level 2PC — the next chapter.

Compare MyRocks (`storage/rocksdb/ha_rocksdb.cc` — Chapter 11): a completely different
storage architecture (LSM tree) behind the identical interface. That's the payoff of the
abstraction, mismatches and all.

## 6.4 What to remember

1. `handlerton` = engine-level ops (commit/prepare/create); `handler` = per-open-table
   cursor with an ISAM-shaped API. Iterators above, pages and B+trees below.
2. Rows cross in MySQL record format; InnoDB's prebuilt + row template make the conversion
   cheap. `ha_*` wrappers add instrumentation and row-based binlogging.
3. `ha_innobase` maps almost 1:1 onto the row operations of the InnoDB deep-dive —
   `row_search_mvcc` is `row_search_for_client` grown up.
4. The API's row-at-a-time, stats-opaque design explains ICP, MRR, BKA, and the whole
   "pushdown" genre of MySQL features.

**Try it:** in gdb, `break ha_innobase::index_read` and run a point SELECT; walk the stack
up (RefIterator → executor loop) and down (`row_search_mvcc`) — the two halves of both
series meet in that one backtrace.

---
**Previous:** [Chapter 5 — The Iterator Executor](./05-executor.md) · **Next:** [Chapter 7 — MDL & Transaction Coordination](./07-mdl-and-transactions.md)
