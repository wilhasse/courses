# Chapter 10 — The Data Dictionary & Atomic DDL

> Where the schema lives since MySQL 8.0 — and how DDL finally became crash-safe.
> Source: `sql/dd/`, `sql/sql_table.cc`, `storage/innobase/log/log0ddl.cc`,
> `storage/innobase/handler/handler0alter.cc`

## 10.1 The 8.0 revolution: no more .frm files

Before 8.0, MySQL's schema was scattered: `.frm` files for table definitions, `.par` for
partitions, `.trg` for triggers, plus the engine's *own* dictionary (InnoDB's `SYS_TABLES` —
the very tables you studied in the InnoDB series, Ch. 10). Two sources of truth, updated
non-transactionally: a crash mid-DDL could leave an orphaned `.frm`, an InnoDB table with no
`.frm`, or both halves disagreeing.

MySQL 8.0 deleted all of it. Now there is **one data dictionary, stored in InnoDB tables**
(in the `mysql.ibd` tablespace), updated transactionally. The design InnoDB pioneered in 2000 —
"the schema is just rows in my own B+trees" — was promoted to the whole server.

```
                     5.7                                  8.0
        ┌─────────────────────────┐         ┌──────────────────────────────┐
        │ .frm  .par  .trg files  │         │  dd::Table, dd::Column, ...  │
        │ mysql.* MyISAM tables   │   ──►   │  = rows in DD tables inside  │
        │ InnoDB SYS_* tables     │         │    mysql.ibd (InnoDB, ACID)  │
        └─────────────────────────┘         └──────────────────────────────┘
```

## 10.2 The dd:: object model

The dictionary is exposed to server code as C++ objects (`sql/dd/types/`):
`dd::Schema` → `dd::Table` (→ `dd::Column`, `dd::Index`, `dd::Foreign_key`,
`dd::Partition`, `dd::Check_constraint`), plus `dd::Tablespace`, `dd::View`, `dd::Routine`,
`dd::Trigger`, `dd::Event`… All descend from `dd::Entity_object` (id + name). Each type maps
to a physical DD table declared in `sql/dd/impl/tables/` (`tables.cc`, `columns.cc`,
`indexes.cc`, `schemata.cc`…). The dictionary's own version is a row in `dd_properties`
(`DD_VERSION` = 80023, `sql/dd/dd_version.h`).

Access goes through a per-session **`Dictionary_client`**
(`sql/dd/cache/dictionary_client.h:149`) backed by a three-tier cache:

```
THD ── Dictionary_client ── local Object_registry
                              │ miss
                              ▼
              Shared_dictionary_cache (process-wide, LRU)
                              │ miss
                              ▼
              Storage_adapter → SELECT from DD tables (InnoDB)
```

The API is deliberately transactional-looking: `acquire()` (shared, read-only),
`acquire_for_modification()` (returns a mutable clone), `store()`/`update()`/`drop()`
(`:1168-1224`) — dictionary writes are just InnoDB row changes inside the DDL's transaction.
An RAII `Auto_releaser` (`:177`) unpins acquired objects at scope exit.

Bootstrap solves the same chicken-and-egg the InnoDB series met at page 7: on first start,
`dd::bootstrap::initialize()` (`sql/dd/impl/bootstrap/bootstrapper.cc:918`) creates the DD
tablespace and tables from compiled-in definitions, then stores *the dictionary's own
definition* into it (`populate_tables`, `:595`).

**INFORMATION_SCHEMA fell out for free**: in 8.0 most I_S tables are just **SQL views over the
DD tables** (`sql/dd/impl/system_views/` — one file per view). No more special-purpose C++
fill functions; `SELECT FROM information_schema.tables` is a plain query the optimizer can
optimize. Percona adds two views of its own here (`compression_dictionary*.cc`).

## 10.3 Atomic DDL: the two-dictionary problem, solved properly

Even with one dictionary, DDL touches things a transaction cannot roll back: files created,
B+trees built, tablespaces renamed. 8.0's answer has two coordinated halves:

1. **All metadata changes are one InnoDB transaction** — the DD rows (via
   `Dictionary_client`) and InnoDB's internal dictionary updates commit or roll back together.
2. **Physical file operations go through the InnoDB DDL log** — the `mysql.innodb_ddl_log`
   table (`storage/innobase/log/log0ddl.cc`; `DDL_Record` `:78`, `Log_DDL` manager `:408`).
   Before each irreversible physical step (create/free an index tree, delete a file, rename a
   tablespace), a compensating record is logged *in the same transactional world*:

```
CREATE TABLE t:                                on crash:
  ddl_log += "FREE index tree X"               │ trx did not commit →
  build index tree X                           │ replay ddl_log entries of the trx:
  ddl_log += "DELETE file t.ibd"               │   free tree X, delete t.ibd
  create t.ibd                                 │ → as if CREATE never happened
  insert DD rows (dd::Table, columns...)       │ trx committed →
  COMMIT  ──►  post_ddl: purge the ddl_log     │   nothing to do; ddl_log purged
```

So `DROP TABLE` on a crashing server can no longer strand half a table: recovery replays or
discards the DDL log entries according to whether the owning transaction committed
(`Log_DDL::replay_*`, `post_ddl`). This is the server-wide generalization of the
undo-then-redo layering from the InnoDB series (Ch. 5).

The SQL-layer flow for `CREATE TABLE` ties it together: `mysql_create_table`
(`sql/sql_table.cc:10239`) → `create_table_impl` (`:8901`) → `rea_create_base_table`
(`:1084`) → build the `dd::Table` object (`dd::create_dd_user_table`,
`sql/dd/dd_table.h:88`) → `handler::create()` in the engine → one commit.

## 10.4 ALTER TABLE: COPY, INPLACE, INSTANT

Three algorithms, negotiated between server and engine
(`handler::check_if_supported_inplace_alter`,
`storage/innobase/handler/handler0alter.cc`):

| algorithm | what happens | blocking |
|-----------|--------------|----------|
| **COPY** | server-level: create temp table, copy every row, swap (`mysql_alter_table`, `sql/sql_table.cc`) | writes blocked |
| **INPLACE** | engine rebuilds/builds indexes itself, concurrent DML journaled and replayed (`prepare/inplace/commit_inplace_alter_table`; parallel builders in `storage/innobase/ddl/`) | mostly online |
| **INSTANT** | metadata-only — e.g. ADD COLUMN just records a new row version in the DD (`Instant_Type`, `handler0alter.cc:913`; version keys `DD_INSTANT_VERSION_ADDED/DROPPED`, `:515-581`) | none |

INSTANT is the endpoint of an old idea: since rows carry format metadata, new columns can
exist only in the dictionary until a row is next rewritten — schema change in O(1). (The
row-versioning bookkeeping lives in the same DD private data that Ch. 2 of the InnoDB series
would have called "the record format's evolution problem".)

## 10.5 What to remember

1. Since 8.0 the schema is **rows in InnoDB tables** (`mysql.ibd`), accessed through the
   `dd::` object model + a three-tier cache, versioned and bootstrapped like any dictionary.
2. Crash-safe DDL = transactional DD rows + the **InnoDB DDL log** for compensating physical
   operations. Recovery decides by transaction outcome — the same commit-point discipline as
   row data.
3. I_S became views; `.frm` files, and an entire genre of "orphaned table" bugs, are gone.
4. ALTER negotiates COPY / INPLACE / INSTANT with the engine — read
   `check_if_supported_inplace_alter` once and MySQL's online-DDL documentation becomes
   self-evident.

**Try it:** `SELECT * FROM information_schema.innodb_tables WHERE name LIKE 'mysql/%';` shows
the DD tables living inside InnoDB; `SHOW CREATE VIEW information_schema.tables\G` reveals the
I_S-as-view mechanism.

---
**Previous:** [Chapter 9 — Replication](./09-replication.md) · **Next:** [Chapter 11 — What Percona Adds](./11-percona-additions.md)
