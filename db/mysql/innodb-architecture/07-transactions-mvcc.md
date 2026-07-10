# Chapter 7 — Transactions, Undo Logs & MVCC

> **Layer 5 of 5 — Transactions.** Commit, rollback, and the trick that lets readers never
> block writers: multi-version concurrency control built on undo logs.
> Source: `trx/trx0trx.c`, `trx/trx0sys.c`, `trx/trx0undo.c`, `trx/trx0rec.c`,
> `trx/trx0roll.c`, `trx/trx0purge.c`, `read/read0read.c`, `row/row0vers.c`

## 7.1 The transaction object and system

A transaction is a `trx_t` (`include/trx0trx.h:463-696`). The fields that matter most:

| field | meaning |
|-------|---------|
| `id` | assigned at start, from a global counter |
| `no` | *serialization number*, assigned at **commit** — commit order |
| `conc_state` | `TRX_NOT_STARTED` → `TRX_ACTIVE` → (`TRX_PREPARED`) → `TRX_COMMITTED_IN_MEMORY` |
| `isolation_level` | `TRX_ISO_*`; default REPEATABLE READ (`include/trx0trx.h:717-749`) |
| `read_view` | the MVCC snapshot (below) |
| `insert_undo`, `update_undo` | the trx's two undo logs |
| `undo_no` | count of undo records written — the rollback "stack pointer" |

Shared state lives in `trx_sys` (`include/trx0sys.h:527-554`): the list of open transactions,
the `max_trx_id` counter, the rollback segments, and the read views. Almost all of it is
protected by one giant **kernel mutex** (`SYNC_KERNEL`, Chapter 4) — the coarse-grained choice
that later versions of InnoDB spent a decade breaking apart.

Persistence lives on the **TRX_SYS page** — page 5 of the system tablespace
(`TRX_SYS_PAGE_NO`, `include/trx0sys.h:402`): the high-water `max_trx_id` (written every 256
ids so a restart can never reuse one, `TRX_SYS_TRX_ID_WRITE_MARGIN`, `include/trx0sys.h:559`),
256 rollback-segment slots, and — as seen in Chapter 3 — the doublewrite buffer's location.

## 7.2 Undo logs: the before-image store

Redo (Chapters 4-5) makes changes durable; **undo makes them revocable and multi-versioned**.
Undo records live in ordinary 16KB pages (`FIL_PAGE_UNDO_LOG`) inside **rollback segments** —
so they are buffered, latched, and *redo-logged* like everything else.

```
TRX_SYS page (space 0, page 5)
 └─ 256 rollback segment slots → rseg header page (page 6 = the first)
      └─ rseg: TRX_RSEG_UNDO_SLOTS (1024 slots) + TRX_RSEG_HISTORY list
           └─ undo log segment (one per active trx & type)
                └─ undo pages → undo records
```

Each transaction that writes gets a rollback segment (round-robin,
`trx_assign_rseg`, `trx/trx0trx.c:608`) and up to two undo logs in it
(`include/trx0undo.h:346-360`):

- **insert undo** — records for freshly inserted rows. After commit these are worthless (the
  row is fully visible; no older version exists), so they're freed immediately.
- **update undo** — records for updates and delete-marks: the **before image**. These must
  outlive commit, because other transactions' snapshots may still need the old version.

An update-undo record (`trx/trx0rec.c`; types `TRX_UNDO_UPD_EXIST_REC` etc.,
`include/trx0rec.h:309-330`) stores: table id, the primary key of the row, the old
`DB_TRX_ID` + `DB_ROLL_PTR`, and the old values of changed columns.

### The version chain

Recall the hidden columns (Chapter 2). Every update rewrites them:

```
row (current, in clustered index leaf)
  DB_TRX_ID = 105, DB_ROLL_PTR ──► undo rec (before image, trx 105)
                                     old DB_TRX_ID = 92, old DB_ROLL_PTR ──► undo rec (trx 92)
                                                                               old DB_TRX_ID = 71 ...
```

`DB_ROLL_PTR` is a packed 7-byte (rseg id, page no, offset) pointer
(`trx_undo_build_roll_ptr`, `include/trx0undo.ic:35`), with its top bit flagging
insert-vs-update undo. Follow the chain and you time-travel backwards through the row's
history. **This chain is the entire basis of MVCC.**

## 7.3 Read views: deciding what a transaction may see

A **read view** (`read_view_t`, `include/read0read.h:126-162`) is a snapshot of the
transaction state at one instant, built by `read_view_open_now()` (`read/read0read.c:252`)
under the kernel mutex:

- `low_limit_id` — `max_trx_id` at snapshot time: any trx id ≥ this started *after* the
  snapshot → invisible.
- `up_limit_id` — the smallest active id: any trx id < this had committed *before* the
  snapshot → visible.
- `trx_ids[]` — ids active (or prepared) at snapshot time → invisible (in-flight).

Visibility of a row version is then `read_view_sees_trx_id(view, DB_TRX_ID)`
(`include/read0read.ic:61-98`):

```
             visible ◄─┤ up_limit_id        low_limit_id ├─► invisible
   ─────────────────────┼───────────────────────┼──────────────────────► trx id
                        │   in trx_ids[]?       │
                        │   yes → invisible     │
                        │   no  → visible       │
```

When a reader lands on a record whose `DB_TRX_ID` it cannot see, it rebuilds an older version:
`row_vers_build_for_consistent_read()` (`row/row0vers.c:484`) walks the `DB_ROLL_PTR` chain,
applying before-images one step at a time, until it reaches a version the view sees (or the
chain's start — meaning the row didn't exist for this reader). **Readers take no locks and
block nothing**; they pay with CPU and undo-page reads proportional to how far behind they are.

Isolation levels fall out of *when the view is created*:

- **REPEATABLE READ** (default): one view per transaction, created at the first read
  (`trx_assign_read_view`, `trx/trx0trx.c:976`) and reused — every statement sees the same
  snapshot.
- **READ COMMITTED**: the view is closed after each statement
  (`read_view_close_for_read_committed`, `read/read0read.c:340`), so the next statement builds
  a fresh one — it sees everything committed so far.
- **READ UNCOMMITTED**: skip version building entirely (read the newest version, dirty).
- **SERIALIZABLE**: reads become locking reads (Chapter 8); MVCC steps aside.

## 7.4 Commit and rollback

### Commit is cheap by design

`trx_commit_off_kernel()` (`trx/trx0trx.c:730`) does **no work proportional to the
transaction's size** — the changes are already in the pages, already redo-logged:

1. In one mini-transaction: mark the undo logs finished, assign the serialization number
   `trx->no`, and hand the update-undo log to the **history list** of its rollback segment
   (`trx_undo_update_cleanup` → `trx_purge_add_update_undo_to_history`,
   `trx/trx0purge.c:308`). The mtr's commit is *the* durable commit point.
2. Set `conc_state = TRX_COMMITTED_IN_MEMORY` and release all locks
   (`lock_release_off_kernel`, Chapter 8).
3. Flush the log up to the commit LSN per `flush_log_at_trx_commit` (Chapter 5's group
   commit).

Note what makes this correct: after step 1, crash recovery will find the undo log in
"finished" state and treat the transaction as committed; before it, recovery rolls it back.
A transaction's fate is decided by one atomic mtr on one undo page.

### Rollback replays undo in reverse

Rollback (`trx_general_rollback`, `trx/trx0roll.c:66`) pops undo records in strictly
descending `undo_no` order across both undo logs
(`trx_roll_pop_top_rec_of_trx`, `trx/trx0roll.c:677`) and dispatches each to the row layer
(`row_undo`, `row/row0undo.c:232`): an insert-undo record removes the inserted row; an
update-undo record restores the before image. Because `undo_no` counts records per
transaction, **partial rollback is free**: rolling back to a savepoint (`trx_savept_take`,
`trx/trx0roll.c:173`) just stops popping at the saved `undo_no` — the same mechanism rolls
back a single failed statement (`trx->last_sql_stat_start`).

Crash recovery's undo phase (Chapter 5) is this very code, run against transactions found
`TRX_ACTIVE` after redo. And XA two-phase commit (`trx_prepare_off_kernel`,
`trx/trx0trx.c:1792`) is a small variation: persist the state `TRX_PREPARED` (+XID) in the
undo header, flush the log, and let the coordinator decide later
(`trx_recover`, `:1930`, lists in-doubt transactions after a crash).

## 7.5 Purge: taking out the MVCC garbage

Two kinds of garbage accumulate: update-undo logs of committed transactions (the history
list), and delete-marked records still physically sitting in indexes (Chapter 6). Neither can
be removed while *any* read view might still need them.

The **purge system** (`trx_purge_t`, `include/trx0purge.h:131-186`, driven from the master
thread — Chapter 12) computes its horizon the elegant way: it *clones the oldest existing read
view* (`read_view_oldest_copy_or_open_new`, `read/read0read.c:168`). Whatever is invisible to
that view is invisible to everyone — safe to destroy. Purge then walks the history lists in
commit order (`trx_purge`, `trx/trx0purge.c:1099`), physically removing delete-marked records
from indexes (`row/row0purge.c`, Chapter 9) and freeing undo pages
(`trx_purge_truncate_history`, `:599`).

The failure mode is famous: **a long-running read view stalls purge**, the history list grows
(`trx_sys->rseg_history_len`), version chains lengthen, reads slow down. This codebase already
has the pressure valve — `max_purge_lag` (`srv_max_purge_lag`, checked in `trx_purge`
`:1137-1151`), which artificially delays DML when history grows too long. Twenty years later,
"HISTORY LIST LENGTH" is still the first thing a DBA checks on a struggling MySQL server.

## 7.6 What to remember

1. Undo = before-images in regular, redo-protected pages; each row heads a **version chain**
   through `DB_ROLL_PTR`.
2. A read view is three numbers and a list; visibility is a comparison against them, and
   isolation levels differ only in *when views are (re)built*.
3. Commit is O(1): flip undo state, assign commit number, release locks, flush log. Rollback
   is O(changes): pop undo records in reverse. Recovery reuses both.
4. Purge is a garbage collector whose horizon is the oldest read view; the history list is
   its backlog gauge.

**Try it:** `tests/.libs/ib_mt_stress` runs concurrent transactions; break in
`read_view_open_now` and inspect `trx_ids` to watch a snapshot being taken.

---
**Previous:** [Chapter 6 — The B+Tree](./06-btree.md) · **Next:** [Chapter 8 — The Lock Manager](./08-locking.md)
