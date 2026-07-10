# Chapter 10 — The Data Dictionary & the Internal SQL Interpreter

> Where InnoDB keeps its own schema — in itself — and the miniature SQL engine it uses to
> do so.
> Source: `dict/dict0boot.c`, `dict/dict0dict.c`, `dict/dict0load.c`, `dict/dict0crea.c`,
> `pars/`, `que/`, `eval/`

## 10.1 The bootstrapping trick

InnoDB stores table metadata *in InnoDB tables*. That's circular: to read a table you need its
metadata, which lives in a table. The cycle is broken with one well-known page — the
**dictionary header**, page 7 of the system tablespace (`DICT_HDR_PAGE_NO`,
`include/dict0boot.h:96`):

```
DICT_HDR page (space 0, page 7)              include/dict0boot.h:118-132
┌────────────────────────────────────────────────────────┐
│ DICT_HDR_ROW_ID    next DB_ROW_ID to assign            │
│ DICT_HDR_TABLE_ID  next table id                       │
│ DICT_HDR_INDEX_ID  next index id                       │
│ DICT_HDR_TABLES    root page # of SYS_TABLES tree   ───┼─► fixed entry points
│ DICT_HDR_TABLE_IDS root page # of SYS_TABLE_IDS tree   │   into four B+trees
│ DICT_HDR_COLUMNS   root page # of SYS_COLUMNS tree     │   whose schemas are
│ DICT_HDR_INDEXES   root page # of SYS_INDEXES tree     │   HARD-CODED in C
│ DICT_HDR_FIELDS    root page # of SYS_FIELDS tree      │
└────────────────────────────────────────────────────────┘
```

The four **system tables** are ordinary B+trees; what's special is that their column layouts
are compiled into the engine (`dict_boot()`, `dict/dict0boot.c:226`, builds their `dict_table_t`
descriptors from literal C code):

| table | one row per | key columns |
|-------|-------------|-------------|
| `SYS_TABLES` | table | NAME → (ID, N_COLS, TYPE, SPACE, …) |
| `SYS_COLUMNS` | column | (TABLE_ID, POS) → (NAME, MTYPE, PRTYPE, LEN, …) |
| `SYS_INDEXES` | index | (TABLE_ID, ID) → (NAME, N_FIELDS, TYPE, SPACE, **PAGE_NO**) |
| `SYS_FIELDS` | index column | (INDEX_ID, POS) → COL_NAME |

`SYS_INDEXES.PAGE_NO` is the payoff: it stores each index's **root page number** — the single
number that connects a name like `t1(PRIMARY)` to a B+tree you can descend (Chapter 6). At
first database creation, `dict_hdr_create()` (`dict/dict0boot.c:124`) `btr_create()`s the four
trees and writes their roots into the header; on every later startup `dict_boot()` just reads
them back. Bootstrap complete.

## 10.2 The in-memory cache

Parsing SYS_ tables on every query would be absurd, so `dict_sys`
(`include/dict0dict.h:1133`) caches loaded tables: hash by name, hash by id, an LRU list, all
under one `dict_sys->mutex`. The cached objects are the structs every other chapter has been
using implicitly:

- **`dict_table_t`** (`include/dict0mem.h:359`) — name, space id, columns, the list of its
  indexes, foreign-key lists, lock list.
- **`dict_index_t`** (`:233`) — **`page` (root page number)**, `space`, `type`
  (`DICT_CLUSTERED`…), the fields, per-index statistics (`stat_n_diff_key_vals` — fed by
  random B+tree dives, this era's whole "optimizer statistics"), the `rw_lock_t lock` — yes,
  *the* tree latch of Chapters 4/6 lives in the dictionary cache entry.

Loading a table by name (`dict_load_table`, `dict/dict0load.c:847`) is plain Chapter 6
machinery: build a search tuple for NAME, descend SYS_TABLES with a persistent cursor, then
walk SYS_COLUMNS / SYS_INDEXES / SYS_FIELDS to assemble the object
(`:876-1035`).

## 10.3 A SQL interpreter inside a storage engine?

Surprise: this "no-SQL storage engine" contains a lexer, a yacc grammar, and a query-graph
executor (`pars/pars0grm.y`, `pars/pars0lex.l`, `que/que0que.c`, `eval/`). It exists because
InnoDB's *own* maintenance operations — creating dictionary rows, foreign-key cascades, purge
traversals — are expressed as little SQL procedures executed by this interpreter
(`pars_sql()` → `que_run_threads()`, `que/que0que.c:1390` is the canonical entry —
`que_eval_sql`).

DDL shows the pattern (`dict/dict0crea.c`): `CREATE TABLE` builds a graph whose nodes insert
rows into SYS_TABLES/SYS_COLUMNS (`tab_create_graph_create`, `:867`), and `CREATE INDEX`
inserts into SYS_INDEXES, then calls `btr_create()` and *updates PAGE_NO in the SYS_INDEXES
row* (`dict_create_index_tree_step`, `:610`). Schema changes are therefore **transactional row
operations on the dictionary tables** — they write undo, they take locks, they can roll back
on crash (which is exactly what recovery does with half-finished DDL, Chapter 5's undo phase).

The executor model (`que0que.c:61-80`) — fork/thread/step nodes pumped by `que_run_threads` —
is also the scaffolding under every row operation of Chapter 9 (`row_ins_step`,
`row_upd_step`…): the embedded API drives the same graph machinery, just with graphs built
directly in C rather than parsed from text.

## 10.4 What to remember

1. The dictionary bootstraps from one fixed page (7) holding root page numbers of four
   hard-coded system tables; everything else about a schema is ordinary B+tree rows.
2. `dict_index_t.page` is the bridge from names to trees; `dict_index_t.lock` is the tree
   latch — the dictionary cache is load-bearing for concurrency, not just metadata.
3. DDL = transactional DML on SYS_ tables, executed by InnoDB's private SQL interpreter —
   crash-safe schema changes fall out of the same undo/redo machinery as everything else.
4. When MySQL 8.0 finally moved *its* data dictionary into InnoDB tables (2018), it was
   re-adopting the design you're looking at here from ~2000.

**Try it:** `tests/.libs/ib_ddl` creates and drops tables; break in `dict_create_index_tree_step`
to watch a root page get allocated and recorded.

---
**Previous:** [Chapter 9 — Row Operations](./09-row-operations.md) · **Next:** [Chapter 11 — Startup, Shutdown & the Embedded API](./11-startup-api.md)
